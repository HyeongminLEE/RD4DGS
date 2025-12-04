import math

import torch
from torch.nn import functional as F
from diff_gaussian_rasterization_df import GaussianRasterizationSettings, GaussianRasterizer
from utils.sh_utils import eval_sh
from utils.interpolations import quat_slerp_interp_uniiterval

index_cache_global_dict = {
    'scale': None,
    'rot': None,
    'dc': None,
    'sh1': None,
    'sh2': None,
    'sh3': None
}

def render(viewpoint_camera, pc, pipe, bg_color : torch.Tensor, timestamp=None, scaling_modifier=1.0, override_color=None, subpixel_offset=None, mode=0, training=False, near=0.2, far=100.0,
          # RDO-related parameters
          activate_shprune=False, activate_gsprune=False, activate_vq=False, activate_wavelet=False,
          vq_cfg=None, update_index=False,
          lst_prune_mask=[None, None, None], lst_extract_dynamic_mask=[None, None]):
    """
    Render the scene.

    Background tensor (bg_color) must be on GPU!
    """
    # prune index cache if nan_prune_mask is not None
    if index_cache_global_dict['scale'] is not None:
        if lst_extract_dynamic_mask[0] is not None:
            ex_static_prune_mask = lst_extract_dynamic_mask[0]
            static_num = ex_static_prune_mask.shape[0]
            for key in index_cache_global_dict.keys():
                if key == 'rot':
                    index_cache_global_dict[key] = torch.cat([index_cache_global_dict[key][:static_num][~ex_static_prune_mask], index_cache_global_dict[key][static_num:], torch.repeat_interleave(index_cache_global_dict[key][:static_num][ex_static_prune_mask], pc.keyframe_num, dim=0)])
                else:
                    index_cache_global_dict[key] = torch.cat([index_cache_global_dict[key][:static_num][~ex_static_prune_mask], index_cache_global_dict[key][static_num:], index_cache_global_dict[key][:static_num][ex_static_prune_mask]])

        for msk in lst_prune_mask:
            if msk is not None:
                msk_static, msk_dynamic = msk
                for key in index_cache_global_dict.keys():
                    if key == 'rot':
                        index_cache_global_dict[key] = index_cache_global_dict[key][~torch.cat([msk_static, torch.repeat_interleave(msk_dynamic, pc.keyframe_num, dim=0)])]
                    else:
                        index_cache_global_dict[key] = index_cache_global_dict[key][~torch.cat([msk_static, msk_dynamic])]

        if lst_extract_dynamic_mask[1] is not None:
            ex_static_prune_mask = lst_extract_dynamic_mask[1]
            static_num = ex_static_prune_mask.shape[0]
            for key in index_cache_global_dict.keys():
                if key == 'rot':
                    index_cache_global_dict[key] = torch.cat([index_cache_global_dict[key][:static_num][~ex_static_prune_mask], index_cache_global_dict[key][static_num:], torch.repeat_interleave(index_cache_global_dict[key][:static_num][ex_static_prune_mask], pc.keyframe_num, dim=0)])
                else:
                    index_cache_global_dict[key] = torch.cat([index_cache_global_dict[key][:static_num][~ex_static_prune_mask], index_cache_global_dict[key][static_num:], index_cache_global_dict[key][:static_num][ex_static_prune_mask]])

    timestamp = timestamp if timestamp is not None else viewpoint_camera.timestamp
    # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
    screenspace_points = torch.zeros_like(pc.get_xyz_at_t(timestamp, mode=mode), dtype=pc._xyz.dtype, requires_grad=True, device="cuda") + 0
    try:
        screenspace_points.retain_grad()
    except:
        pass

    # Set up rasterization configuration
    tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
    tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)

    if subpixel_offset is None:
        subpixel_offset = torch.zeros((int(viewpoint_camera.image_height), int(viewpoint_camera.image_width), 2), dtype=torch.float32, device="cuda")

    raster_settings = GaussianRasterizationSettings(
        image_height=int(viewpoint_camera.image_height),
        image_width=int(viewpoint_camera.image_width),
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        kernel_size=pc.kernel_size,
        subpixel_offset=subpixel_offset,
        bg=bg_color,
        scale_modifier=scaling_modifier,
        viewmatrix=viewpoint_camera.world_view_transform,
        projmatrix=viewpoint_camera.full_proj_transform,
        sh_degree=pc.active_sh_degree,
        campos=viewpoint_camera.camera_center,
        prefiltered=False,
        min_depth=near,
        max_depth=far,
        debug=pipe.debug
    )

    rasterizer = GaussianRasterizer(raster_settings=raster_settings)
    means3D = pc.get_xyz_at_t(timestamp, mode=mode, training=training)
    means2D = screenspace_points
    opacity = pc.get_opacity_at_t(timestamp, mode=mode, training=training)

    num_static = pc._xyz.shape[0]
    num_dynamic = pc._xyz_motion.shape[0]
    num_gs = num_static + num_dynamic
    if activate_wavelet and mode != 1 and num_dynamic > 0:
        xyz_motion = pc.apply_wv_mask_at_t(timestamp)['xyz_motion']
        means3D = torch.cat([pc._xyz, xyz_motion], dim=0)

    flow = torch.zeros_like(means3D, requires_grad=True) + 0
    try:
        flow.retain_grad()
    except:
        pass

    scales = None
    rotations = None
    # If precomputed 3d covariance is provided, use it. If not, then it will be computed from
    # scaling / rotation by the rasterizer.
    cov3D_precomp = None
    if pipe.compute_cov3D_python:
        # warning: this may not work correctly with the video model
        cov3D_precomp = pc.get_covariance_at_t(timestamp, scaling_modifier, mode=mode)
    else:
        scales = pc.get_scaling(mode=mode)
        rotations = pc.get_rotation_at_t(timestamp, mode=mode)

    # If precomputed colors are provided, use them. Otherwise, if it is desired to precompute colors
    # from SHs in Python, do it. If not, then SH -> RGB conversion will be done by rasterizer.
    shs = None
    colors_precomp = None
    if override_color is None:
        if pipe.convert_SHs_python:
            shs_view = pc.get_features(motion=mode).transpose(1, 2).view(-1, 3, (pc.max_sh_degree+1)**2)
            dir_pp = (pc.get_xyz_at_t(timestamp, motion=mode) - viewpoint_camera.camera_center.repeat(pc.get_features(mode=mode).shape[0], 1))
            dir_pp_normalized = dir_pp/dir_pp.norm(dim=1, keepdim=True)
            sh2rgb = eval_sh(pc.active_sh_degree, shs_view, dir_pp_normalized)
            colors_precomp = torch.clamp_min(sh2rgb + 0.5, 0.0)
        else:
            shs = pc.get_features(mode=mode)
    else:
        colors_precomp = override_color

    # Process Gaussian pruning mask if activated
    if activate_gsprune:
        gs_prune_result_dict = pc.apply_gs_mask_at_t(timestamp, mode=mode, training=training)
        scales = gs_prune_result_dict['scale']
        opacity = gs_prune_result_dict['opacity']
        gs_soft_mask = gs_prune_result_dict['gs_soft_mask']
        gs_hard_mask = gs_prune_result_dict['gs_hard_mask']
        gs_bitmask = gs_prune_result_dict['gs_bitmask']
        gs_mask_loss = torch.mean(gs_soft_mask)
        # calculate mask percentage
        gs_mask_percent = 1 - gs_hard_mask.sum() / max(gs_hard_mask.numel(), 1)
        gs_mask_percent_static = 1 - gs_hard_mask[:num_static].sum() / max(gs_hard_mask[:num_static].numel(), 1)
        gs_mask_percent_dynamic = 1 - gs_hard_mask[num_static:].sum() / max(gs_hard_mask[num_static:].numel(), 1)
    else:
        gs_mask_loss = torch.tensor(0.0, device="cuda")
        gs_mask_percent = 0.0
        gs_mask_percent_static = 0.0
        gs_mask_percent_dynamic = 0.0

    # Process RDO masks if activated
    if activate_shprune:
        sh_prune_result_dict = pc.apply_sh_mask(mode=mode)
        features_rest = sh_prune_result_dict['features_rest']
        sh_soft_mask = sh_prune_result_dict['sh_soft_mask']
        sh_hard_mask = sh_prune_result_dict['sh_hard_mask']
        sh1_bitmask = sh_prune_result_dict['sh1_bitmask']
        sh2_bitmask = sh_prune_result_dict['sh2_bitmask']
        sh3_bitmask = sh_prune_result_dict['sh3_bitmask']
        if activate_gsprune:
            sh_hard_mask = sh_hard_mask[gs_bitmask]
            sh_soft_mask = sh_soft_mask[gs_bitmask]
            sh1_bitmask = torch.logical_and(sh1_bitmask, gs_bitmask)
            sh2_bitmask = torch.logical_and(sh2_bitmask, gs_bitmask)
            sh3_bitmask = torch.logical_and(sh3_bitmask, gs_bitmask)
        sh_mask_loss = torch.mean(sh_soft_mask)
        # calculate mask percentage
        sh_mask_percent = 1 - sh_hard_mask.sum() / sh_hard_mask.numel()
        sh_mask_percent_static = 1 - sh_hard_mask[:num_static].sum() / sh_hard_mask[:num_static].numel()
        sh_mask_percent_dynamic = 1 - sh_hard_mask[num_static:].sum() / sh_hard_mask[num_static:].numel()
        shs[:, 1:, :] = features_rest
    else:
        sh_mask_loss = torch.tensor(0.0, device="cuda")
        sh_mask_percent = 0.0
        sh_mask_percent_static = 0.0
        sh_mask_percent_dynamic = 0.0

    def ste(y_hat, y):
        return (y_hat - y).detach() + y

    # Initialize RDO-related variables
    rate_loss = [torch.tensor(0.0, device="cuda")]
    vq_loss = [torch.tensor(0.0, device="cuda")]
    bits_dict = {}
    index_cache_dict = {}
    vq_dim = 0

    # Vector Quantization if activated
    if activate_vq:
        vq_inputs = pc.vector_dict
        vq_out = {}
        vq_inputs['sh1'] = features_rest[sh1_bitmask,:3,:].flatten(start_dim=1)
        vq_inputs['sh2'] = features_rest[sh2_bitmask,3:8,:].flatten(start_dim=1)
        vq_inputs['sh3'] = features_rest[sh3_bitmask,8:,:].flatten(start_dim=1)
        vq_inputs['scale'] = scales[gs_bitmask,:]

        static_num = pc._xyz.shape[0]
        dynamic_num = pc._xyz_motion.shape[0]
        keyframe_num = pc.keyframe_num
        masked_static_num = gs_bitmask[:static_num].sum()
        masked_dynamic_num = gs_bitmask[static_num:].sum()
        rot_static = vq_inputs['rot'][:static_num,0,:][gs_bitmask[:static_num]]
        rot_dynamic = vq_inputs['rot'][static_num:,:,:][gs_bitmask[static_num:]].reshape(masked_dynamic_num * keyframe_num, 4)
        rot_bitmask = torch.cat([gs_bitmask[:static_num], torch.repeat_interleave(gs_bitmask[static_num:], keyframe_num, dim=0)])
        vq_inputs['rot'] = torch.cat([rot_static, rot_dynamic], dim=0)

        vq_inputs['dc'] = vq_inputs['dc'][gs_bitmask,:]
        if not update_index:
            # use cached indexes
            index_cache_dict['scale'] = index_cache_global_dict['scale'][gs_bitmask]
            index_cache_dict['rot'] = index_cache_global_dict['rot'][rot_bitmask]
            index_cache_dict['dc'] = index_cache_global_dict['dc'][gs_bitmask]
            index_cache_dict['sh1'] = index_cache_global_dict['sh1'][sh1_bitmask]
            index_cache_dict['sh2'] = index_cache_global_dict['sh2'][sh2_bitmask]
            index_cache_dict['sh3'] = index_cache_global_dict['sh3'][sh3_bitmask]

        for vq_key in vq_cfg['keys']:
            result = pc._quantizer[vq_key](vq_inputs[vq_key], index_cache=None if update_index else index_cache_dict[vq_key])
            vq_out[vq_key] = ste(y_hat=result['x_hat'], y=vq_inputs[vq_key])
            if vq_key != 'rot':
                vq_out[vq_key] = pc.vq_post_process(vq_key, vq_out[vq_key])
            bits_dict[vq_key] = result['bits']
            rate_loss.append(bits_dict[vq_key] / vq_cfg['lmbda'][vq_key])
            index_cache_dict[vq_key] = result['x_index']
            vq_loss.append(torch.sum(torch.norm(result['x_hat'] - vq_inputs[vq_key], dim=-1)))
            vq_dim += vq_inputs[vq_key].shape[-1]

        scales[gs_bitmask,:] = vq_out['scale']

        rot_q_static = vq_out['rot'][:masked_static_num,:]
        rot_q_dynamic = vq_out['rot'][masked_static_num:,:].reshape(masked_dynamic_num, keyframe_num, 4)
        t = timestamp + pc.time_shift
        t_idx = t // pc.interval
        delta_t = (t % pc.interval) / pc.interval
        t_idx = int(t_idx)
        rot_q_dynamic = quat_slerp_interp_uniiterval(rot_q_dynamic[:,t_idx,:], rot_q_dynamic[:,t_idx+1,:], delta_t).squeeze(1)
        rot_q = torch.cat([rot_q_static, rot_q_dynamic], dim=0)
        rot_q = pc.vq_post_process('rot', rot_q)
        rotations[gs_bitmask,:] = rot_q

        shs[gs_bitmask, 0:1, :] = vq_out['dc']
        shs[sh1_bitmask, 1:4, :] = vq_out['sh1']
        shs[sh2_bitmask, 4:9, :] = vq_out['sh2']
        shs[sh3_bitmask, 9:, :] = vq_out['sh3']

        if index_cache_global_dict['scale'] is None:
            for key in index_cache_global_dict.keys():
                if key == 'rot':
                    index_cache_global_dict[key] = torch.zeros(num_static + num_dynamic * keyframe_num, 1, device='cuda', dtype=torch.long)
                else:
                    index_cache_global_dict[key] = torch.zeros(num_gs, 1, device='cuda', dtype=torch.long)

        index_cache_global_dict['scale'][gs_bitmask] = index_cache_dict['scale']
        index_cache_global_dict['rot'][rot_bitmask] = index_cache_dict['rot']
        index_cache_global_dict['dc'][gs_bitmask] = index_cache_dict['dc']
        index_cache_global_dict['sh1'][sh1_bitmask] = index_cache_dict['sh1']
        index_cache_global_dict['sh2'][sh2_bitmask] = index_cache_dict['sh2']
        index_cache_global_dict['sh3'][sh3_bitmask] = index_cache_dict['sh3']

        # Set final VQ losses
        if vq_dim != 0:
            rate_loss = sum(rate_loss) / num_gs / vq_dim
            vq_loss = sum(vq_loss) / num_gs / vq_dim
        else:
            rate_loss = sum(rate_loss) / num_gs
            vq_loss = sum(vq_loss) / num_gs

    # Rasterize visible Gaussians to image, obtain their radii (on screen).
    rendered_image, radii, rendered_depth, out_flow, acc, idxs = rasterizer(
        means3D = means3D,
        means2D = means2D,
        dir3D = flow,
        shs = shs,
        colors_precomp = colors_precomp,
        opacities = opacity,
        scales = scales,
        rotations = rotations,
        cov3D_precomp = cov3D_precomp)

    torch.cuda.synchronize()

    # Create result dictionary with all outputs
    result = {
        "render": rendered_image,
        "depth": rendered_depth,
        "opticalflow": out_flow,
        "acc": acc,
        "viewspace_points": screenspace_points,
        "viewspace_l1points": flow,
        "dominent_idxs": idxs,
        "visibility_filter": radii > 0,
        "radii": radii,
        # RDO-related outputs
        "vq_loss": vq_loss,
        "rate_loss": rate_loss,
        "bits": bits_dict,
        "sh_mask_loss": sh_mask_loss,
        "sh_mask_percent": sh_mask_percent * 100,
        "sh_mask_percent_static": sh_mask_percent_static * 100,
        "sh_mask_percent_dynamic": sh_mask_percent_dynamic * 100,
        "gs_mask_loss": gs_mask_loss,
        "gs_mask_percent": gs_mask_percent * 100,
        "gs_mask_percent_static": gs_mask_percent_static * 100,
        "gs_mask_percent_dynamic": gs_mask_percent_dynamic * 100
    }

    # Those Gaussians that were frustum culled or had a radius of 0 were not visible.
    # They will be excluded from value updates used in the splitting criteria.
    return result
