import os
import json
import sys
import uuid
import math
from random import randint
import gc

import torch
from tqdm import tqdm
from PIL import Image
import joblib
import datetime

from utils.image_utils import psnr
from utils.loss_utils import l1_loss, ssim
from utils.general_utils import safe_state, PILtoTorch
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, OptimizationParams
from vq.ecvq import get_vq_cfg

from gaussian_renderer_wavelet import render, network_gui
from scene_wavelet import Scene, getmodel

# RDO-related imports
import numpy as np

torch.set_default_dtype(torch.float32)


def training(dataset, opt, pipe, testing_iterations, saving_iterations, checkpoint_iterations, checkpoint, debug_from, args):
    first_iter = 0
    activate_vq = False
    activate_gsprune = False
    activate_shprune = False
    activate_wavelet = False
    prepare_output_and_logger(args)
    vq_cfg = get_vq_cfg(opt)
    GaussianModel = getmodel(dataset.model) # gmodel, gmodelrgbonly
    gaussians = GaussianModel(dataset.sh_degree, dataset.start_duration, dataset.time_interval, dataset.time_pad,
                              interp_type=dataset.interp_type, rot_interp_type=dataset.rot_interp_type,
                              time_pad_type=dataset.time_pad_type, var_pad=dataset.var_pad, kernel_size=dataset.kernel_size,
                              vq_cfg=vq_cfg)
    scene = Scene(dataset, gaussians, use_timepad=True)

    # Initialize VQ quantizers structure first
    gaussians.initialize_quantizer() # Initialize the structure

    # Setup optimizers (including VQ and mask optimizers if RDO enabled)
    gaussians.training_setup(opt)
    args.duration = dataset.duration

    if checkpoint:
        (model_params, first_iter) = torch.load(checkpoint, weights_only=False)
        gaussians.restore(model_params, opt)

    gaussians._opacity_duration_var = torch.nan_to_num(gaussians._opacity_duration_var, nan=0.0)

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
    iter_start = torch.cuda.Event(enable_timing = True)
    iter_end = torch.cuda.Event(enable_timing = True)

    # variables for progressive training
    scene.set_sampling_len(dataset.start_duration, sample_every=dataset.sample_every)
    expanded = gaussians.expand_duration(dataset.start_duration)

    sample_len = dataset.start_duration
    g_sample_len = dataset.start_duration

    mark_extract = False
    need_extract = True
    mark_last = False
    viewpoint_stack = None
    prune_inv = False
    train_images = None
    e_count = args.extract_every

    progress_bar = tqdm(range(first_iter, opt.iterations), desc="Training progress") # Include last iteration

    first_iter += 1
    lst_prune_mask = [None, None, None]
    lst_extract_dynamic_mask = [None, None]
    for iteration in range(first_iter, opt.iterations + 1):
        if network_gui.conn == None:
            network_gui.try_connect()
        while network_gui.conn != None:
            try:
                net_image_bytes = None
                custom_cam, do_training, pipe.convert_SHs_python, pipe.compute_cov3D_python, keep_alive, scaling_modifer = network_gui.receive()
                if custom_cam != None:
                    net_image = render(custom_cam, gaussians, pipe, background, scaling_modifer, far=dataset.far)["render"]
                    net_image_bytes = memoryview((torch.clamp(net_image, min=0, max=1.0) * 255).byte().permute(1, 2, 0).contiguous().cpu().numpy())
                network_gui.send(net_image_bytes, dataset.source_path)
                if do_training and ((iteration < int(opt.iterations)) or not keep_alive):
                    break
            except Exception as e:
                network_gui.conn = None
        if iteration == opt.vq_start_iter:
            # kmeans initialization
            for key in vq_cfg["keys"]:
                if key == 'rot':
                    rot = gaussians.vector_dict[key]
                    static_num = gaussians._xyz.shape[0]
                    rot_static = rot[:static_num][:, 0, :]
                    rot_dynamic = rot[static_num:].reshape(-1, rot.shape[2])
                    gaussians._quantizer[key].kmeans_initialize(torch.cat([rot_static, rot_dynamic], dim=0))
                else:
                    gaussians._quantizer[key].kmeans_initialize(gaussians.vector_dict[key].data)
            activate_vq = True
        if activate_vq and iteration == opt.rate_constrain_iter:
            print(f"\n[ITER {iteration}] Set rate_constrain = True")
            for key in vq_cfg["keys"]:
                gaussians._quantizer[key].rate_constrain = True
        if iteration == opt.gs_prune_start_iter:
            activate_gsprune = True
        if iteration == opt.sh_prune_start_iter:
            activate_shprune = True
        if iteration == opt.wv_prune_start_iter:
            activate_wavelet = True

        iter_start.record()
        gaussians.update_learning_rate(iteration)

        # Every 1000 its we increase the levels of SH up to a maximum degree
        if iteration % 1000 == 0:
            gaussians.oneupSHdegree()

        # Pick a random Camera
        if not viewpoint_stack:
            viewpoint_stack, train_images = scene.getTrainCameras(return_as='generator', shuffle=True)
            viewpoint_stack = viewpoint_stack.copy()

            if iteration > opt.prune_invisible_interval:
                prune_inv = True

        viewpoint_cam = viewpoint_stack.pop(0)
        gt_image = next(train_images).cuda()

        if mark_last:
            if viewpoint_cam.timestamp >= scene.sample_len - gaussians.interval:
                mark_extract = True
                mark_last = False

        # Render
        if (iteration - 1) == debug_from:
            pipe.debug = True

        bg = torch.rand((3), device="cuda") if opt.random_background else background

        # Updated render call with RDO parameters
        render_pkg = render(
            viewpoint_cam, gaussians, pipe, bg,
            near=dataset.near, far=dataset.far,
            # RDO parameters
            activate_shprune=activate_shprune,
            activate_gsprune=activate_gsprune,
            activate_vq=activate_vq,
            activate_wavelet=activate_wavelet,
            vq_cfg=vq_cfg if activate_vq else None,
            update_index=(iteration-1) % 10 == 0,
            lst_prune_mask=lst_prune_mask,
            lst_extract_dynamic_mask=lst_extract_dynamic_mask
        )

        lst_prune_mask = [None, None, None]
        lst_extract_dynamic_mask = [None, None]
        image, viewspace_point_tensor, visibility_filter, radii = render_pkg["render"], render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"]

        viewspace_point_error_tensor = render_pkg["viewspace_l1points"]
        depth = render_pkg["depth"]
        flow = render_pkg["opticalflow"]
        acc = render_pkg["acc"]
        idxs = render_pkg["dominent_idxs"]

        rate_loss = render_pkg["rate_loss"]
        vq_loss = render_pkg["vq_loss"]
        sh_mask_loss = render_pkg["sh_mask_loss"]
        sh_mask_percent = render_pkg["sh_mask_percent"]
        sh_mask_percent_static = render_pkg["sh_mask_percent_static"]
        sh_mask_percent_dynamic = render_pkg["sh_mask_percent_dynamic"]
        gs_mask_loss = render_pkg["gs_mask_loss"]
        gs_mask_percent = render_pkg["gs_mask_percent"]
        gs_mask_percent_static = render_pkg["gs_mask_percent_static"]
        gs_mask_percent_dynamic = render_pkg["gs_mask_percent_dynamic"]
        bits_dict = render_pkg["bits"]

        # Loss Calculation
        Ll1 = l1_loss(image, gt_image)
        loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim(image, gt_image))

        # backtrack register
        if opt.l1_accum:
            l1_errors = (image - gt_image).abs().mean(dim=0)
            ssim_errors = ssim(image, gt_image, reduce=False).mean(dim=0)
            hook_tensor = torch.stack([acc[0], l1_errors, ssim_errors])
            flow_h = flow.register_hook(lambda grad: hook_tensor)
            loss += flow.mean() * 0

        # Regularization
        if opt.static_reg > 0 and iteration > opt.progressive_growing_steps + opt.make_dynamic_interval:
            loss += opt.static_reg * torch.log(gaussians._xyz_disp.norm(dim=-1)+0.001).mean()

        if opt.motion_reg > 0 and iteration > opt.progressive_growing_steps * opt.extract_every + opt.make_dynamic_interval and gaussians._xyz_motion.shape[0] > 0:
            diff1 = (gaussians._xyz_motion[:, :1] - gaussians._xyz_motion[:, 1:])
            loss += opt.motion_reg * diff1.norm(dim=-1).mean()

        if opt.rot_reg > 0 and iteration > opt.progressive_growing_steps * opt.extract_every + opt.make_dynamic_interval and gaussians._xyz_motion.shape[0] > 0:
            r1 = gaussians._rotation_motion[:, 1:]
            r2 = gaussians._rotation_motion[:, :-1]

            r_i = 1 - (r1 * r2).sum(dim=-1) / r1.norm(dim=-1).clamp_min(1e-6) / r2.norm(dim=-1).clamp_min(1e-6)
            loss += opt.rot_reg * r_i.mean()

        if activate_vq:
            loss += vq_loss + rate_loss

        if activate_shprune:
            loss += opt.sh_mask_lambda * sh_mask_loss

        if activate_gsprune:
            loss += opt.gs_mask_lambda * gs_mask_loss

        loss.backward()

        if opt.l1_accum:
            flow_h.remove()

        iter_end.record()
        torch.cuda.synchronize()
        gaussians.mark_error(loss.item(), viewpoint_cam.timestamp)

        with torch.no_grad():
            # Progress bar update
            psnr_log = psnr(image.unsqueeze(0), gt_image.unsqueeze(0)).mean().item()

            if iteration % 10 == 0:
                # Standard losses and PSNR for terminal output
                log_message_parts = []
                if not torch.isnan(loss) and not math.isnan(loss.item()):
                    log_message_parts.append(f"Loss: {loss.item():.6f}")
                if not torch.isnan(Ll1) and not math.isnan(Ll1.item()):
                    log_message_parts.append(f"L1: {Ll1.item():.6f}")
                if not math.isnan(psnr_log):
                     log_message_parts.append(f"PSNR: {psnr_log:.2f}")

                # RDO Losses for terminal output
                if activate_vq:
                    if not torch.isnan(vq_loss) and not math.isnan(vq_loss.item()):
                        log_message_parts.append(f"VQ Loss: {vq_loss.item():.6f}")
                    if not torch.isnan(rate_loss) and not math.isnan(rate_loss.item()):
                        log_message_parts.append(f"Rate Loss: {rate_loss.item():.6f}")
                if activate_shprune:
                    if not torch.isnan(sh_mask_loss) and not math.isnan(sh_mask_loss.item()):
                        log_message_parts.append(f"SH Mask Loss: {sh_mask_loss.item():.6f}")
                if activate_gsprune:
                    if not torch.isnan(gs_mask_loss) and not math.isnan(gs_mask_loss.item()):
                        log_message_parts.append(f"GS Mask Loss: {gs_mask_loss.item():.6f}")

                # RDO Bits (converted to MB) for terminal output
                if activate_vq and bits_dict:
                    total_bits = 0
                    valid_bits_exist = False
                    for key, bits_val in bits_dict.items():
                        if not math.isnan(bits_val):
                            total_bits += bits_val
                            valid_bits_exist = True
                            log_message_parts.append(f"Bits/{key}: {bits_val / (8 * 1024 * 1024):.4f}MB")
                    if valid_bits_exist and not math.isnan(total_bits):
                         log_message_parts.append(f"Bits/Total: {total_bits / (8 * 1024 * 1024):.4f}MB")

                # RDO Mask Percentages for terminal output
                if activate_shprune:
                    if not math.isnan(sh_mask_percent):
                        log_message_parts.append(f"SH Mask %: {sh_mask_percent:.2f}")
                    if not math.isnan(sh_mask_percent_static):
                         log_message_parts.append(f"SH Mask Static %: {sh_mask_percent_static:.2f}")
                    if not math.isnan(sh_mask_percent_dynamic):
                         log_message_parts.append(f"SH Mask Dynamic %: {sh_mask_percent_dynamic:.2f}")
                if activate_gsprune:
                    if not math.isnan(gs_mask_percent):
                        log_message_parts.append(f"GS Mask %: {gs_mask_percent:.2f}")
                    if not math.isnan(gs_mask_percent_static):
                         log_message_parts.append(f"GS Mask Static %: {gs_mask_percent_static:.2f}")
                    if not math.isnan(gs_mask_percent_dynamic):
                         log_message_parts.append(f"GS Mask Dynamic %: {gs_mask_percent_dynamic:.2f}")

                # Log static/dynamic point counts for terminal output
                static_points = gaussians._xyz.shape[0]
                dynamic_points = gaussians._xyz_motion.shape[0]
                total_points_val = static_points + dynamic_points
                log_message_parts.append(f"Static Points: {static_points}")
                log_message_parts.append(f"Dynamic Points: {dynamic_points}")
                log_message_parts.append(f"Total Points: {total_points_val}")

                if total_points_val > 0:
                    log_message_parts.append(f"Static Ratio: {static_points / total_points_val:.2f}")
                    log_message_parts.append(f"Dynamic Ratio: {dynamic_points / total_points_val:.2f}")

                if log_message_parts:
                    print(f"[ITER {iteration}] " + ", ".join(log_message_parts))


                # Update progress bar
                progress_info = {"PSNR": f"{psnr_log:.{2}f}", "Loss": f"{loss.item():.{6}f}"}
                if activate_shprune: progress_info["SH%"] = f"{sh_mask_percent:.{2}f}"
                if activate_gsprune: progress_info["GS%"] = f"{gs_mask_percent:.{2}f}"

                if activate_vq and bits_dict:
                    total_bits_val = sum(bits_dict.values())
                    if not math.isnan(total_bits_val):
                         progress_info["Bits"] = f"{total_bits_val / (8*1024*1024):.{2}f}MB"
                progress_bar.set_postfix(progress_info)
                progress_bar.update(10)

            if iteration == opt.iterations:
                progress_bar.close()

            if iteration in checkpoint_iterations:
                print(f"[ITER {iteration}] Saving Checkpoint")
                removed_nan = gaussians.handle_nan_values()
                if removed_nan:
                    print(f"[ITER {iteration}] Pruned {removed_nan[0].sum()} static and {removed_nan[1].sum()} dynamic NaN points before saving.")
                torch.save((gaussians.capture(), iteration), scene.model_path + "/chkpnt" + str(iteration) + ".pth")

            # Call training_report (which will now also just print to console)
            training_report(iteration, Ll1, loss, l1_loss, iter_start.elapsed_time(iter_end), testing_iterations, scene, (pipe, background), dataset.near, dataset.far,
                            vq_cfg=vq_cfg, activate_vq=activate_vq, activate_shprune=activate_shprune, activate_gsprune=activate_gsprune)

            # Final Pruning and Compressed Model Saving
            if iteration == opt.iterations:
                print("Number of Gaussians before pruning: ", gaussians._xyz.shape[0] + gaussians._xyz_motion.shape[0])
                final_sh_mask_percent = None
                if activate_shprune:
                    sh_prune_final_dict = gaussians.apply_sh_mask()
                    gaussians._features_rest = sh_prune_final_dict['features_rest']
                    if sh_prune_final_dict['sh_hard_mask'] is not None and sh_prune_final_dict['sh_hard_mask'].numel() > 0:
                        final_sh_mask_percent = (1 - sh_prune_final_dict['sh_hard_mask'].sum() / sh_prune_final_dict['sh_hard_mask'].numel()) * 100
                        print(f'SHs Masked percent {final_sh_mask_percent:.4f}')
                    else:
                        print("Could not calculate final SH mask percentage.")

                final_gs_mask_percent_proxy = None
                if activate_gsprune and not math.isnan(gs_mask_percent):
                    final_gs_mask_percent_proxy = gs_mask_percent
                    print(f'Gaussians Masked percent (from last iteration) {final_gs_mask_percent_proxy:.4f}')

                vq_path = os.path.join(scene.model_path, "bitstream")
                if not args.no_rdo:
                    print("Handling NaNs before final VQ save...")
                    gaussians.handle_nan_values()
                    gaussians.apply_and_save_vq(vq_path)
                final_static_points = gaussians._xyz.shape[0]
                final_dynamic_points = gaussians._xyz_motion.shape[0]
                final_total_points = final_static_points + final_dynamic_points
                print("Number of Gaussians after pruning/VQ: ", final_total_points)

                summary_message_parts = [
                    f'Final Static Points: {final_static_points}',
                    f'Final Dynamic Points: {final_dynamic_points}',
                    f'Final Total Points: {final_total_points}'
                ]
                if final_sh_mask_percent is not None and not math.isnan(final_sh_mask_percent):
                    summary_message_parts.append(f'Final SH Mask %: {final_sh_mask_percent:.4f}')
                if final_gs_mask_percent_proxy is not None:
                     summary_message_parts.append(f'Final GS Mask % (Proxy): {final_gs_mask_percent_proxy:.4f}')
                print(", ".join(summary_message_parts))


            if iteration in saving_iterations:
                print(f"[ITER {iteration}] Saving Gaussians")
                scene.save(iteration)

            if activate_vq and (iteration % opt.reactivate_codeword_period == 0 and iteration < opt.rate_constrain_iter):
                print(f"\n[ITER {iteration}] Reactivate codeword {vq_cfg['keys']}")
                for key in vq_cfg["keys"]:
                    gaussians._quantizer[key].reactivate_codeword(prob_threshold=1e-5)

            if activate_vq and (iteration % opt.reactivate_codeword_period == 0 and iteration >= opt.rate_constrain_iter):
                print(f"\n[ITER {iteration}] Codebook info: {vq_cfg['keys']}")
                for key in vq_cfg["keys"]:
                    gaussians._quantizer[key].print_codebook_info(prob_threshold=1e-5)

            if iteration < opt.densify_until_iter:
                static_num = gaussians._xyz.shape[0]
                static_vis_filter = visibility_filter[:static_num]
                static_radii = radii[:static_num]
                dynamic_vis_filter = visibility_filter[static_num:]
                dynamic_radii = radii[static_num:]

                gaussians.max_radii2D[static_vis_filter] = torch.max(gaussians.max_radii2D[static_vis_filter], static_radii[static_vis_filter])
                gaussians.motion_max_radii2D[dynamic_vis_filter] = torch.max(gaussians.motion_max_radii2D[dynamic_vis_filter], dynamic_radii[dynamic_vis_filter])
                gaussians.add_densification_stats(viewspace_point_tensor, static_vis_filter, dynamic_vis_filter, static_num)

                if opt.l1_accum:
                    gaussians.add_l1_ssim_stats(viewspace_point_error_tensor, static_vis_filter, dynamic_vis_filter, static_num, viewpoint_cam.timestamp)

                if iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0:
                    size_threshold, dynamic_size_threshold = None, None
                    s_max_ssim = opt.s_max_ssim  if iteration > opt.error_base_prune_steps and iteration % (opt.densification_interval * opt.ssim_prune_every) == 0 else 0
                    s_l1_thres = opt.s_l1_thres if iteration > opt.error_base_prune_steps and iteration % (opt.densification_interval * opt.l1_prune_every) == 0 else 100

                    d_max_ssim = opt.d_max_ssim  if iteration > opt.error_base_prune_steps and iteration % (opt.densification_interval * opt.ssim_prune_every) == 0 else 0
                    d_l1_thres = opt.d_l1_thres if iteration > opt.error_base_prune_steps and iteration % (opt.densification_interval * opt.l1_prune_every) == 0 else 100
                    gaussians.densify_and_prune(opt.densify_grad_threshold,
                                                opt.densify_dgrad_threshold,
                                                0.01, 0.01, scene.cameras_extent, size_threshold, dynamic_size_threshold,
                                                s_max_ssim=s_max_ssim, s_l1_thres=s_l1_thres, d_max_ssim=d_max_ssim, d_l1_thres=d_l1_thres)
                elif iteration > opt.extract_from_iter and iteration % opt.extracton_interval == 0:
                    static_num = gaussians._xyz.shape[0]
                    candidate = gaussians.get_errorneous_timestamp()
                    if not candidate is None:
                        extract_dynamic_mask = gaussians.extract_dynamic_points_from_static(torch.tensor(viewpoint_cam.T).unsqueeze(0), candidate, static_vis_filter, scene.cameras_extent, percentile=opt.extract_percentile, max_dur=sample_len)
                        lst_extract_dynamic_mask[1] = extract_dynamic_mask
            if iteration % (opt.densification_interval*4) == 0 and iteration < opt.densify_until_iter - 3000:
                gaussians.adjust_temp_opa(max_dur=sample_len)

            invisible_prune_mask = None
            small_prune_mask = None
            if prune_inv and iteration < opt.iterations and iteration > 3000:
                invisible_prune_mask = gaussians.prune_invisible()
                if opt.l1_accum:
                    small_prune_mask = gaussians.prune_small()
                prune_inv = False
            lst_prune_mask[0] = invisible_prune_mask
            lst_prune_mask[1] = small_prune_mask

            if iteration < opt.iterations:
                gaussians.optimizer.step()
                gaussians.optimizer.zero_grad(set_to_none = True)
                if activate_vq:
                    gaussians.optimizer_vq.step()
                    gaussians.scheduler_vq.step()
                gaussians.optimizer_vq.zero_grad(set_to_none = True)
                if activate_shprune:
                    gaussians.optimizer_sh_mask.step()
                gaussians.optimizer_sh_mask.zero_grad(set_to_none = True)
                if activate_gsprune:
                    gaussians.optimizer_gs_mask.step()
                gaussians.optimizer_gs_mask.zero_grad(set_to_none = True)
                nan_prune_mask = gaussians.handle_nan_values()
                lst_prune_mask[2] = nan_prune_mask
            if iteration > opt.extract_from_iter and (iteration % opt.progressive_growing_steps == opt.make_dynamic_interval) and need_extract :
                mark_last = True
                need_extract = False

            if iteration > opt.extract_from_iter and iteration % opt.progressive_growing_steps == 0 and iteration > opt.progressive_growing_steps and ~need_extract:
                sample_len = min(dataset.duration + gaussians.time_shift, int(dataset.time_interval * dataset.progressive_step) + scene.sample_len)

                scene.set_sampling_len(sample_len, sample_every=dataset.sample_every)
                g_sample_len = min(dataset.duration + gaussians.time_shift, sample_len)
                expanded = gaussians.expand_duration(g_sample_len)

                if expanded:
                    e_count += 1
                    if e_count >= opt.extract_every:
                        mark_last = True
                        need_extract = True
                        e_count = 0
            if mark_extract:
                static_num = gaussians._xyz.shape[0]
                static_vis_filter = visibility_filter[:static_num]
                extract_dynamic_mask = gaussians.extract_dynamic_points_from_static(torch.tensor(viewpoint_cam.T).unsqueeze(0), viewpoint_cam.timestamp,
                                                                                     static_vis_filter, scene.cameras_extent, percentile=opt.extract_percentile, max_dur=sample_len)
                lst_extract_dynamic_mask[1] = extract_dynamic_mask
                mark_extract = False
        torch.cuda.empty_cache()


def prepare_output_and_logger(args):
    if not args.model_path:
        if os.getenv('OAR_JOB_ID'):
            unique_str=os.getenv('OAR_JOB_ID')
        else:
            unique_str = str(uuid.uuid4())
        args.model_path = os.path.join("./output/", unique_str[0:10])

    print("Output folder: {}".format(args.model_path))
    os.makedirs(args.model_path, exist_ok = True)
    with open(os.path.join(args.model_path, "cfg_args"), 'w') as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))

    print("Output directory prepared. Logging will be to console.")
    return None


def training_report(iteration, Ll1, loss, l1_loss_fn, elapsed, testing_iterations, scene, renderArgs, near, far,
                    vq_cfg, activate_vq=False, activate_shprune=False, activate_gsprune=False):

    # Basic train metrics for console
    report_message_parts = []
    if not torch.isnan(Ll1) and not math.isnan(Ll1.item()):
        report_message_parts.append(f"Batch L1: {Ll1.item():.6f}")
    if not torch.isnan(loss) and not math.isnan(loss.item()):
         report_message_parts.append(f"Batch Total Loss: {loss.item():.6f}")
    if not math.isnan(elapsed):
        report_message_parts.append(f"Iter Time (ms): {elapsed:.2f}")
    if report_message_parts:
        print(f"[ITER {iteration} Report] " + ", ".join(report_message_parts))


    if iteration in testing_iterations:
        torch.cuda.empty_cache()

        test_viewpoint_stack, test_images = scene.getTestCameras(shuffle=False, return_as='generator',  n_job=1)
        test_viewpoint_stack = test_viewpoint_stack.copy()

        train_viewpoint_stack, train_images = scene.getTrainCameras(shuffle=False, return_as='generator', n_job=1)
        train_viewpoint_stack = train_viewpoint_stack.copy()

        validation_configs = ({'name': 'test', 'cameras': test_viewpoint_stack,
                                                'images': test_images},
                              {'name': 'train', 'cameras': train_viewpoint_stack,
                                                 'images': train_images})
        for config in validation_configs:
            if config['cameras'] and len(config['cameras']) > 0:
                l1_test = 0.0
                psnr_test = 0.0
                # ssim_test = 0.0 # Add SSIM calculation if desired
                count = 0
                # Removed images_to_log for wandb

                for idx, viewpoint in enumerate(config['cameras']):
                    log_this_view = False
                    if config['name'] == 'test':
                        log_this_view = (idx < 5) # Still useful to see some renders even if not logging to wandb
                    elif config['name'] == 'train':
                         sampled_idx_list = [i % len(train_viewpoint_stack) for i in range(5)]
                         if idx in sampled_idx_list:
                             log_this_view = True
                         else:
                            _ = next(config['images'])
                            continue

                    gt_image = next(config['images']).cuda()
                    with torch.no_grad():
                        rend_pkg = render(viewpoint, scene.gaussians, near=near, far=far, *renderArgs,
                                          activate_shprune=activate_shprune,
                                          activate_gsprune=activate_gsprune,
                                          activate_vq=activate_vq,
                                          vq_cfg=vq_cfg if activate_vq else None,
                                          update_index=False)

                    image = torch.clamp(rend_pkg["render"], 0.0, 1.0).cuda()

                    if log_this_view:
                         # Here you could save images locally if needed, e.g., using torchvision.utils.save_image
                         # For now, we'll just print a message that a view was rendered for eval.
                         print(f"[ITER {iteration} Eval] Rendered {config['name']} view: {viewpoint.image_name}")
                         pass

                    current_l1 = l1_loss_fn(image, gt_image).mean().double().detach().item()
                    current_psnr = psnr(image.unsqueeze(0), gt_image.unsqueeze(0)).mean().double().detach().item()

                    if not math.isnan(current_l1): l1_test += current_l1
                    if not math.isnan(current_psnr): psnr_test += current_psnr
                    count += 1

                    del(rend_pkg)
                    del(image)
                    del(gt_image)
                    del(viewpoint)
                    torch.cuda.empty_cache()

                if count > 0:
                    psnr_test /= count
                    l1_test /= count

                eval_message = f"[ITER {iteration}] Evaluating {config['name']}: L1 {l1_test:.6f} PSNR {psnr_test:.2f}"
                print(eval_message)

                # Removed eval_log_data and wandb.log

        # Removed histogram logging
        torch.cuda.empty_cache()


if __name__ == "__main__":
    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument('--ip', type=str, default="127.0.0.1")
    parser.add_argument('--port', type=int, default=6009)
    # parser.add_argument('--wandb_name', type=str, default="tmp") # Removed
    parser.add_argument('--debug_from', type=int, default=-1)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[7_000_000]) # Adjusted for clarity
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[30_000, 40_000])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[30_000, 40_000])
    parser.add_argument("--start_checkpoint", type=str, default = None)
    parser.add_argument("--configpath", type=str, default = "None") # Changed from config to configpath for consistency
    parser.add_argument("--no_rdo", action="store_true", default=False)
    args = parser.parse_args(sys.argv[1:])
    args.save_iterations.append(args.iterations)
    args.checkpoint_iterations.append(args.iterations)

    print("Optimizing " + args.model_path)

    safe_state(args.quiet)

    if os.path.exists(args.configpath) and args.configpath != "None":
        print("Overloading config from " + args.configpath)
        config_data = json.load(open(args.configpath)) # Renamed to avoid conflict
        for k in config_data.keys():
            try:
                # value = getattr(args, k) # Not needed
                newvalue = config_data[k]
                setattr(args, k, newvalue)
            except AttributeError: # Catch if arg doesn't exist
                print(f"Warning: Config key '{k}' not found in script arguments.")
            except Exception as e:
                print(f"Failed to set config: {k} due to {e}")
        print("Finished loading config from " + args.configpath)
    # Removed the 'else: raise ValueError' as config might not always be provided
    # and defaults in argparse should handle it.
    # If a config is *required* for some runs, that logic should be more specific.

    network_gui_initialized = False
    while not network_gui_initialized:
        try:
            network_gui.init(args.ip, args.port)
            network_gui_initialized = True
        except OSError as e:
            if "Address already in use" in str(e):
                print(f"Port {args.port} already in use, trying next port.")
                args.port += 1
            else:
                raise e
        except Exception as e:
             print(f"Error initializing GUI server: {e}")
             raise e

    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    try:
        training(lp.extract(args), op.extract(args), pp.extract(args), args.test_iterations, args.save_iterations, args.checkpoint_iterations, args.start_checkpoint, args.debug_from, args)
    finally:
        # Removed wandb.finish()
        print("Training process finished or encountered an error.")

    print("\nTraining complete.")
