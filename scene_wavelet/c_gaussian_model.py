import math
import os

import numpy as np
import torch
from torch import nn
from plyfile import PlyData, PlyElement

from utils.sh_utils import RGB2SH
from simple_knn._C import distCUDA2
from utils.general_utils import inverse_sigmoid, get_expon_lr_func, build_rotation
from utils.graphics_utils import BasicPointCloud, BasicPointCloud_pc
from utils.general_utils import strip_symmetric, build_scaling_rotation, MultipleOptimizer
from utils.system_utils import mkdir_p
from utils.interpolations import linear_interp_uniiterval, quat_slerp_interp_uniiterval, time_bigaussian, pchip_interpolate, cube_interpolate, quad_diff_interpolate

# RDO imports
from torch.optim.lr_scheduler import MultiStepLR
from vq.ecvq import ECVQ
from utils.encode_utils import *

# Wavelet imports
from utils.wavelet_utils import generate_haar_matrix, points_to_wavelet_coeffs, mask_wavelet_coeffs, restore_masked_wavelet_coeffs, wavelet_coeffs_to_points


class CGaussianModel:
    def setup_functions(self):
        def build_covariance_from_scaling_rotation(scaling, scaling_modifier, rotation):
            L = build_scaling_rotation(scaling_modifier * scaling, rotation)
            actual_covariance = L @ L.transpose(1, 2)
            symm = strip_symmetric(actual_covariance)
            return symm

        self.scaling_activation = torch.exp
        self.scaling_inverse_activation = torch.log

        self.covariance_activation = build_covariance_from_scaling_rotation

        self.opacity_activation = torch.sigmoid
        self.inverse_opacity_activation = inverse_sigmoid

        self.rotation_activation = torch.nn.functional.normalize

    def __init__(self, sh_degree : int, duration : int, interval : int, time_pad : int = 1, interp_type='linear', rot_interp_type='slerp', time_pad_type=0, var_pad=3, kernel_size=0.1, vq_cfg=None, **kwargs):
        self.active_sh_degree = 0
        self.max_sh_degree = sh_degree
        self._xyz = torch.empty(0).cuda()
        self._features_dc = torch.empty(0).cuda()
        self._features_rest = torch.empty(0).cuda()
        self._scaling = torch.empty(0).cuda()
        self._rotation = torch.empty(0).cuda()
        self._opacity = torch.empty(0).cuda()
        self.max_radii2D = torch.empty(0).cuda()
        self.min_radii2D = torch.empty(0).cuda()
        self.xyz_gradient_accum = torch.empty(0).cuda()
        self.denom = torch.empty(0).cuda()
        self.xyz_error_accum = torch.empty(0).cuda()
        self.xyz_error_min = torch.empty(0).cuda()
        self.xyz_error_min_timestamp = torch.empty(0).cuda()
        self.xyz_ssim_error_accum = torch.empty(0).cuda()
        self.error_denom = torch.empty(0).cuda()
        self.optimizer = None
        self.percent_dense = 0
        self.spatial_lr_scale = 0
        self.var_pad = var_pad
        self.setup_functions()

        # properties for motion handling
        self.kernel_size = kernel_size
        self.duration = max(duration, 1) # prevent zero div
        self.interval = interval
        self.time_pad = time_pad
        self.time_pad_type = time_pad_type
        self.time_shift = time_pad
        self.keyframe_num = 0
        self._xyz_disp = torch.empty(0).cuda()
        self._xyz_motion = torch.empty(0).cuda() # [b, t, 3]
        self._features_dc_motion = torch.empty(0).cuda()
        self._features_rest_motion = torch.empty(0).cuda()
        self._scaling_motion = torch.empty(0).cuda()
        self._opacity_motion = torch.empty(0).cuda() # [b, 1]
        self._opacity_duration_center = torch.empty(0).cuda() # [b, 1]
        self._opacity_duration_var = torch.empty(0).cuda() # [b, 1]
        self.opacity_degree = 2
        self._rotation_motion = torch.empty(0).cuda() # [b, t, 4]
        self.motion_max_radii2D = torch.empty(0).cuda()
        self.motion_min_radii2D = torch.empty(0).cuda()
        self.motion_xyz_gradient_accum = torch.empty(0).cuda()
        self.motion_denom = torch.empty(0).cuda()
        self.motion_xyz_error_min = torch.empty(0).cuda()
        self.motion_xyz_error_mean = torch.empty(0).cuda()
        self.motion_xyz_error_min_timestamp = torch.empty(0).cuda()
        self.motion_xyz_ssim_error_accum = torch.empty(0).cuda()
        self.motion_error_denom = torch.empty(0).cuda()

        self.interp_type = interp_type
        # set interpolations
        if interp_type == "linear":
            self.motion_degree = 1 # points
            def interpolation(y, t_idx, delta_t):
                if type(t_idx) == torch.Tensor:
                    return linear_interp_uniiterval(torch.gather(y, 1, t_idx.repeat(1, 1, 3)),
                                                torch.gather(y, 1, t_idx.repeat(1, 1, 3) + 1), delta_t).squeeze(1)
                else:
                    return linear_interp_uniiterval(y[...,t_idx, :], y[...,t_idx+1, :], delta_t).squeeze(1)
        elif interp_type == "cube":
            self.motion_degree = 1 # points, diff
            def interpolation(y, t_idx, delta_t):
                if type(t_idx) == torch.Tensor:
                    y0 = torch.gather(y, 1, t_idx.repeat(1, 1, 3 * self.motion_degree) - 1)
                    y1 = torch.gather(y, 1, t_idx.repeat(1, 1, 3 * self.motion_degree))
                    y2 = torch.gather(y, 1, t_idx.repeat(1, 1, 3 * self.motion_degree) + 1)
                    y3 = torch.gather(y, 1, t_idx.repeat(1, 1, 3 * self.motion_degree) + 2)
                    return cube_interpolate(y0[..., :3], y1[...,:3],y2[..., :3], y3[...,:3], delta_t).squeeze(1)
                else:
                    return cube_interpolate(y[:, t_idx-1, :3], y[:, t_idx, :3],y[:, t_idx+1, :3], y[:, t_idx+2, :3], delta_t).squeeze(1)
            self.time_shift += interval

        elif interp_type == "cubic_diff":
            self.motion_degree = 1 # points, diff
            def interpolation(y, y_d, t_idx, delta_t):
                if type(t_idx) == torch.Tensor:
                    y1 = torch.gather(y, 1, t_idx.repeat(1, 1, 3 * self.motion_degree))
                    y2 = torch.gather(y, 1, t_idx.repeat(1, 1, 3 * self.motion_degree) + 1)
                    y_d1 = torch.gather(y_d, 1, t_idx.repeat(1, 1, 3 * self.motion_degree))
                    y_d2 = torch.gather(y_d, 1, t_idx.repeat(1, 1, 3 * self.motion_degree) + 1)
                    return quad_diff_interpolate(y1, y2, y_d1, y_d2, delta_t).squeeze(1)
                else:
                    return quad_diff_interpolate(y[:,t_idx], y[:,t_idx+1], y_d[:,t_idx], y_d[:,t_idx+1], delta_t).squeeze(1)

        elif interp_type == "pchip":
            self.motion_degree = 1 # points
            def interpolation(y, t_idx, delta_t):
                if type(t_idx) == torch.Tensor:
                    y0 = torch.gather(y, 1, t_idx.repeat(1, 1, 3 * self.motion_degree) - 1)
                    y1 = torch.gather(y, 1, t_idx.repeat(1, 1, 3 * self.motion_degree))
                    y2 = torch.gather(y, 1, t_idx.repeat(1, 1, 3 * self.motion_degree) + 1)
                    y3 = torch.gather(y, 1, t_idx.repeat(1, 1, 3 * self.motion_degree) + 2)
                    return pchip_interpolate(y0[..., :3], y1[...,:3],y2[..., :3], y3[...,:3], delta_t).squeeze(1)
                else:
                    return pchip_interpolate(y[...,t_idx-1, :3], y[...,t_idx, :3],y[...,t_idx+1, :3], y[...,t_idx+2, :3], delta_t).squeeze(1)
            self.time_shift += interval
        else:
            raise NotImplementedError

        self.interpolator = interpolation

        if rot_interp_type == 'lerp':
            def interpolation(y, t_idx, delta_t):
                if type(t_idx) == torch.Tensor:
                    return linear_interp_uniiterval(torch.gather(y, 1, t_idx.repeat(1, 1, 4)),
                                                torch.gather(y, 1, t_idx.repeat(1, 1, 4) + 1), delta_t).squeeze(1)
                else:
                    return linear_interp_uniiterval(y[...,t_idx, :], y[...,t_idx+1, :], delta_t).squeeze(1)
        elif rot_interp_type == 'slerp':
            # rotation interpolation
            def quat_interpolation(y, t_idx, delta_t):
                if type(t_idx) == torch.Tensor:
                    return quat_slerp_interp_uniiterval(torch.gather(y, 1, t_idx.repeat(1, 1, 4)),
                                                        torch.gather(y, 1, t_idx.repeat(1, 1, 4) + 1), delta_t).squeeze(1)
                else:
                    return quat_slerp_interp_uniiterval(y[...,t_idx, :], y[...,t_idx+1, :], delta_t).squeeze(1)
        else:
            raise NotImplementedError
        self.quat_interpolator = quat_interpolation
        self.error_dict = {}

        # RDO properties
        self.vq_cfg = vq_cfg
        self.gs_mask_thres = 0.1
        self.sh_mask_thres = 0.1
        self._quantizer = nn.ModuleDict()
        self.gs_mask = None
        self.sh_mask = None
        self.gs_mask_motion = None
        self.sh_mask_motion = None

    def get_xyz_at_t(self, t, mode=0, training=True):
        assert t >= -self.time_shift and t <= self.duration+self.time_shift
        if self._xyz_motion.shape[0] == 0 or mode == 1:
            return self.get_static_xyz_at_t(t)
        elif mode == 2:
            return self.get_dynamic_xyz_at_t(t)
        return torch.cat([self.get_static_xyz_at_t(t, training), self.get_dynamic_xyz_at_t(t)], dim=0).contiguous()

    def get_static_xyz_at_t(self, t, training=True):
        # static points
        return self._xyz + self._xyz_disp * t / self.duration

    def get_dynamic_xyz_at_t(self, t):
        # dynamic points
        t = t + self.time_shift
        t_idx = t // self.interval
        delta_t = (t % self.interval) / self.interval

        if type(t) == torch.Tensor:
            t_idx = t_idx.view(-1, 1, 1).long()
            delta_t = delta_t.view(-1, 1, 1)
        else:
            t_idx = int(t_idx)
        return self.interpolator(self._xyz_motion, t_idx, delta_t)

    def get_rotation_at_t(self, t, mode=0):
        assert t >= -self.time_shift and t <= self.duration+self.time_shift
        if self._rotation_motion.shape[0] == 0 or mode == 1:
            return self._rotation
        elif mode == 2:
            return self.get_dynamic_rotation_at_t(t)
        return torch.cat([self._rotation, self.get_dynamic_rotation_at_t(t)], dim=0).contiguous()

    def get_dynamic_rotation_at_t(self, t):
        # dynamic points
        t = t + self.time_shift
        t_idx = t // self.interval
        delta_t = (t % self.interval) / self.interval

        if type(t) == torch.Tensor:
            t_idx = t_idx.view(-1, 1, 1).long()
            delta_t = delta_t.view(-1, 1, 1)
        else:
            t_idx = int(t_idx)

        return self.quat_interpolator(self._rotation_motion, t_idx, delta_t)

    def capture(self):
        return (
            self.active_sh_degree,
            self._xyz,
            self._features_dc,
            self._features_rest,
            self._scaling,
            self._rotation,
            self._opacity,
            self.max_radii2D,
            self.min_radii2D,
            self.xyz_gradient_accum,
            self.denom,
            self.xyz_error_accum,
            self.xyz_error_min,
            self.xyz_error_min_timestamp,
            self.xyz_ssim_error_accum,
            self.error_denom,
            self.optimizer.state_dict(),
            self.spatial_lr_scale,

            self._xyz_disp,
            self.duration,
            self.interval,
            self.time_shift,
            self.keyframe_num,
            self._xyz_motion,
            self._features_dc_motion,
            self._features_rest_motion,
            self._scaling_motion,
            self._opacity_motion,
            self._opacity_duration_center,
            self._opacity_duration_var,
            self._rotation_motion,
            self.motion_max_radii2D,
            self.motion_min_radii2D,
            self.motion_xyz_gradient_accum,
            self.motion_denom,
            self.motion_xyz_error_min,
            self.motion_xyz_error_mean,
            self.motion_xyz_error_min_timestamp,
            self.motion_xyz_ssim_error_accum,
            self.motion_error_denom,

            # RDO-related parameters
            self.gs_mask,
            self.sh_mask,
            self.gs_mask_motion,
            self.sh_mask_motion,
            self._quantizer.state_dict(),
        )

    def restore_partially(self, model_args, training_args):
        (_, _, _features_dc, _features_rest, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _features_dc_motion, _features_rest_motion, _, _, _, _, _, _, _, _, _, _, _, _, _, _, gs_mask, sh_mask, gs_mask_motion, sh_mask_motion) = model_args

        gs_mask = torch.cat([gs_mask, gs_mask_motion], dim=0)
        gs_soft_mask = torch.sigmoid(gs_mask)
        gs_bitmask = gs_soft_mask.flatten() > self.gs_mask_thres

        # Get the expected static and dynamic counts
        static_count = self._xyz.shape[0]
        dynamic_count = self._xyz_motion.shape[0]

        # Calculate separate masks for static and dynamic parts
        static_gs_bitmask = gs_bitmask[:len(gs_bitmask)-gs_mask_motion.shape[0]]
        dynamic_gs_bitmask = gs_bitmask[len(gs_bitmask)-gs_mask_motion.shape[0]:]

        # Create masks for each SH band
        sh_bitmasks = [
            torch.cat([torch.sigmoid(sh_mask[:, 0]) > self.sh_mask_thres, torch.sigmoid(sh_mask_motion[:, 0]) > self.sh_mask_thres], dim=0),
            torch.cat([torch.sigmoid(sh_mask[:, 1]) > self.sh_mask_thres, torch.sigmoid(sh_mask_motion[:, 1]) > self.sh_mask_thres], dim=0),
            torch.cat([torch.sigmoid(sh_mask[:, 2]) > self.sh_mask_thres, torch.sigmoid(sh_mask_motion[:, 2]) > self.sh_mask_thres], dim=0),
        ]

        sh1_bitmask = sh_bitmasks[0]
        sh2_bitmask = sh_bitmasks[1]
        sh3_bitmask = sh_bitmasks[2]

        # Combine with gaussian mask
        sh1_bitmask = torch.logical_and(sh1_bitmask, gs_bitmask)
        sh2_bitmask = torch.logical_and(sh2_bitmask, gs_bitmask)
        sh3_bitmask = torch.logical_and(sh3_bitmask, gs_bitmask)

        # Apply static masks
        filtered_static_features_dc = torch.zeros_like(_features_dc)
        filtered_static_features_rest = torch.zeros_like(_features_rest)

        # Apply dynamic masks
        filtered_dynamic_features_dc = torch.zeros_like(_features_dc_motion)
        filtered_dynamic_features_rest = torch.zeros_like(_features_rest_motion)

        # Keep only features with corresponding true in mask
        static_indices = torch.nonzero(static_gs_bitmask).squeeze(1)
        if len(static_indices) > 0:
            filtered_static_features_dc = _features_dc[static_indices]
            filtered_static_features_rest = _features_rest[static_indices]

        dynamic_indices = torch.nonzero(dynamic_gs_bitmask).squeeze(1)
        if len(dynamic_indices) > 0:
            filtered_dynamic_features_dc = _features_dc_motion[dynamic_indices]
            filtered_dynamic_features_rest = _features_rest_motion[dynamic_indices]

        # Debug info
        print(f"Static mask count: {static_gs_bitmask.sum()}, Dynamic mask count: {dynamic_gs_bitmask.sum()}")
        print(f"Static feature counts: DC={filtered_static_features_dc.shape[0]}, Rest={filtered_static_features_rest.shape[0]}")
        print(f"Dynamic feature counts: DC={filtered_dynamic_features_dc.shape[0]}, Rest={filtered_dynamic_features_rest.shape[0]}")

        # Final assignment
        self._features_dc = filtered_static_features_dc
        self._features_rest = filtered_static_features_rest
        self._features_dc_motion = filtered_dynamic_features_dc
        self._features_rest_motion = filtered_dynamic_features_rest

        # Verify shapes
        assert self._features_dc.shape[0] == static_count, f"Static DC shape mismatch: {self._features_dc.shape[0]} vs {static_count}"
        assert self._features_dc_motion.shape[0] == dynamic_count, f"Dynamic DC shape mismatch: {self._features_dc_motion.shape[0]} vs {dynamic_count}"
        assert self._features_rest.shape[0] == static_count, f"Static rest shape mismatch: {self._features_rest.shape[0]} vs {static_count}"
        assert self._features_rest_motion.shape[0] == dynamic_count, f"Dynamic rest shape mismatch: {self._features_rest_motion.shape[0]} vs {dynamic_count}"

    def restore(self, model_args, training_args):
        if len(model_args) < 45:
            model_args += tuple(None for _ in range(45 - len(model_args)))

        (self.active_sh_degree,
        self._xyz,
        self._features_dc,
        self._features_rest,
        self._scaling,
        self._rotation,
        self._opacity,
        self.max_radii2D,
        self.min_radii2D,
        xyz_gradient_accum,
        denom,
        xyz_error_accum,
        xyz_error_min,
        xyz_error_min_timestamp,
        xyz_ssim_error_accum,
        error_denom,
        opt_dict,
        self.spatial_lr_scale,

        self._xyz_disp,
        self.duration,
        self.interval,
        self.time_shift,
        self.keyframe_num,
        self._xyz_motion,
        self._features_dc_motion,
        self._features_rest_motion,
        self._scaling_motion,
        self._opacity_motion,
        self._opacity_duration_center,
        self._opacity_duration_var,
        self._rotation_motion,
        self.motion_max_radii2D,
        self.motion_min_radii2D,
        motion_xyz_gradient_accum,
        motion_denom,
        motion_xyz_error_min,
        motion_xyz_error_mean,
        motion_xyz_error_min_timestamp,
        motion_xyz_ssim_error_accum,
        motion_error_denom,

        # RDO-related parameters
        gs_mask,
        sh_mask,
        gs_mask_motion,
        sh_mask_motion,
        quant_dict
        ) = model_args

        if len(model_args) > 44:
            quant_dict = model_args[-1]

        if gs_mask is None:
            self.gs_mask = nn.Parameter(torch.zeros(self._xyz.shape[0], 1, device="cuda").requires_grad_(True))
        else:
            self.gs_mask = gs_mask

        if gs_mask_motion is None:
            self.gs_mask_motion = nn.Parameter(torch.zeros(self._xyz_motion.shape[0], 1, device="cuda").requires_grad_(True))
        else:
            self.gs_mask_motion = gs_mask_motion

        if sh_mask is None:
            self.sh_mask = nn.Parameter(torch.zeros(self._xyz.shape[0], 3, device="cuda").requires_grad_(True))
        else:
            self.sh_mask = sh_mask

        if sh_mask_motion is None:
            self.sh_mask_motion = nn.Parameter(torch.zeros(self._xyz_motion.shape[0], 3, device="cuda").requires_grad_(True))
        else:
            self.sh_mask_motion = sh_mask_motion

        self.training_setup(training_args)
        self.xyz_gradient_accum = xyz_gradient_accum
        self.denom = denom
        self.motion_xyz_gradient_accum = motion_xyz_gradient_accum
        self.motion_denom = motion_denom
        self.xyz_error_accum = xyz_error_accum
        self.xyz_error_min = xyz_error_min
        self.xyz_error_min_timestamp = xyz_error_min_timestamp
        self.xyz_ssim_error_accum = xyz_ssim_error_accum
        self.error_denom = error_denom
        self.motion_xyz_error_min = motion_xyz_error_min
        self.motion_xyz_error_mean = motion_xyz_error_mean
        self.motion_xyz_error_min_timestamp = motion_xyz_error_min_timestamp
        self.motion_xyz_ssim_error_accum = motion_xyz_ssim_error_accum
        self.motion_error_denom = motion_error_denom
        self.optimizer.load_state_dict(opt_dict)
        if quant_dict is not None:
            self._quantizer.load_state_dict(quant_dict)

    @property
    def get_static_scaling(self):
        return  self.scaling_activation(self._scaling)

    @property
    def get_motion_scaling(self):
        return self.scaling_activation(self._scaling_motion)

    def get_scaling(self, mode=0):
        if self._scaling_motion.shape[0] == 0 or mode == 1:
            return self.get_static_scaling
        elif mode == 2:
            return self.get_motion_scaling
        return self.scaling_activation(torch.cat([self._scaling, self._scaling_motion], dim=0))

    def get_features(self, mode=0):
        if self._features_dc_motion.shape[0] == 0 or mode == 1:
            features_dc = self._features_dc
            features_rest = self._features_rest
            return torch.cat((features_dc, features_rest), dim=1)
        elif mode == 2:
            features_dc_motion = self._features_dc_motion
            features_rest_motion = self._features_rest_motion
            return torch.cat((features_dc_motion, features_rest_motion), dim=1)

        features_dc = self._features_dc
        features_rest = self._features_rest
        features_dc_motion = self._features_dc_motion
        features_rest_motion = self._features_rest_motion
        return torch.cat((
                torch.cat((features_dc, features_rest), dim=1),
                torch.cat((features_dc_motion, features_rest_motion), dim=1)), dim=0).contiguous()

    # Add masking-related methods
    def apply_sh_mask(self, mode=0):
        """Apply learned mask to SH coefficients"""
        self._check_mask_shape()

        if self._xyz_motion.shape[0] == 0 or mode == 1:
            sh_soft_mask = torch.sigmoid(torch.cat([
                self.sh_mask[:, 0:1].unsqueeze(2).expand(-1, 3, 3),
                self.sh_mask[:, 1:2].unsqueeze(2).expand(-1, 5, 3),
                self.sh_mask[:, 2:3].unsqueeze(2).expand(-1, 7, 3)], dim=1)
            )
            features_rest = self._features_rest
            sh_bitmasks = [
                torch.sigmoid(self.sh_mask[:, 0]) > self.sh_mask_thres,
                torch.sigmoid(self.sh_mask[:, 1]) > self.sh_mask_thres,
                torch.sigmoid(self.sh_mask[:, 2]) > self.sh_mask_thres,
            ]
        elif mode == 2:
            sh_soft_mask = torch.sigmoid(torch.cat([
                self.sh_mask_motion[:, 0:1].unsqueeze(2).expand(-1, 3, 3),
                self.sh_mask_motion[:, 1:2].unsqueeze(2).expand(-1, 5, 3),
                self.sh_mask_motion[:, 2:3].unsqueeze(2).expand(-1, 7, 3)], dim=1)
            )
            features_rest = self._features_rest_motion
            sh_bitmasks = [
                torch.sigmoid(self.sh_mask_motion[:, 0]) > self.sh_mask_thres,
                torch.sigmoid(self.sh_mask_motion[:, 1]) > self.sh_mask_thres,
                torch.sigmoid(self.sh_mask_motion[:, 2]) > self.sh_mask_thres,
            ]
        else:
            sh_soft_mask_static = torch.sigmoid(torch.cat([
                self.sh_mask[:, 0:1].unsqueeze(2).expand(-1, 3, 3),
                self.sh_mask[:, 1:2].unsqueeze(2).expand(-1, 5, 3),
                self.sh_mask[:, 2:3].unsqueeze(2).expand(-1, 7, 3)], dim=1)
            )
            sh_soft_mask_dynamic = torch.sigmoid(torch.cat([
                self.sh_mask_motion[:, 0:1].unsqueeze(2).expand(-1, 3, 3),
                self.sh_mask_motion[:, 1:2].unsqueeze(2).expand(-1, 5, 3),
                self.sh_mask_motion[:, 2:3].unsqueeze(2).expand(-1, 7, 3)], dim=1)
            )
            sh_soft_mask = torch.cat([sh_soft_mask_static, sh_soft_mask_dynamic], dim=0)
            features_rest = torch.cat([self._features_rest, self._features_rest_motion], dim=0)
            sh_bitmasks = [
                torch.cat([torch.sigmoid(self.sh_mask[:, 0]) > self.sh_mask_thres, torch.sigmoid(self.sh_mask_motion[:, 0]) > self.sh_mask_thres], dim=0),
                torch.cat([torch.sigmoid(self.sh_mask[:, 1]) > self.sh_mask_thres, torch.sigmoid(self.sh_mask_motion[:, 1]) > self.sh_mask_thres], dim=0),
                torch.cat([torch.sigmoid(self.sh_mask[:, 2]) > self.sh_mask_thres, torch.sigmoid(self.sh_mask_motion[:, 2]) > self.sh_mask_thres], dim=0),
            ]

        # Create binary mask with straight-through estimator for gradients
        sh_hard_mask = ((sh_soft_mask > self.sh_mask_thres).float() - sh_soft_mask).detach() + sh_soft_mask
        features_rest = torch.mul(features_rest, sh_hard_mask)

        # Return masked features and other information
        return {
            'features_rest': features_rest,
            'sh_soft_mask': sh_soft_mask,
            'sh_hard_mask': sh_hard_mask,
            'sh1_bitmask': sh_bitmasks[0],
            'sh2_bitmask': sh_bitmasks[1],
            'sh3_bitmask': sh_bitmasks[2],
        }

    def apply_wv_mask_at_t(self, t):
        """Apply learned mask to wavelet coefficients"""
        self._check_mask_shape()

        if self._xyz_motion.shape[0] == 0:
            raise ValueError("No dynamic points to apply wavelet mask to")

        is_t_even = (self.keyframe_num % 2 == 0)
        t_half = self.keyframe_num // 2 if is_t_even else (self.keyframe_num + 1) // 2
        wavelet_matrix = generate_haar_matrix(self.keyframe_num if is_t_even else self.keyframe_num + 1).to('cuda')
        points = self._xyz_motion

        if not is_t_even:
            points = torch.cat([points, points[:,-1,:]], dim=1)

        # Get wavelet coefficients
        coeffs = points_to_wavelet_coeffs(points, wavelet_matrix)
        coeffs_approximation = coeffs[:, :t_half, :]
        coeffs_detail_masked = torch.zeros_like(coeffs_approximation)

        # apply mask
        coeffs_masked = torch.cat([coeffs_approximation, coeffs_detail_masked], dim=1)

        # restore points
        restored_points = wavelet_coeffs_to_points(coeffs_masked, wavelet_matrix)
        if not is_t_even:
            restored_points = restored_points[:, :-1, :]

        # dynamic points
        t = t + self.time_shift
        t_idx = t // self.interval
        delta_t = (t % self.interval) / self.interval

        if type(t) == torch.Tensor:
            t_idx = t_idx.view(-1, 1, 1).long()
            delta_t = delta_t.view(-1, 1, 1)
        else:
            t_idx = int(t_idx)
        return {
            'xyz_motion': self.interpolator(restored_points, t_idx, delta_t),
        }

    @property
    def get_opacity(self):
        return self.opacity_activation(self._opacity)

    @property
    def get_motion_opacity(self):
        return self.opacity_activation(self._opacity_motion)

    def get_motion_opacity_at_t(self, t, training=False):
        t = (t + self.time_shift) / self.interval
        return (time_bigaussian(self._opacity_duration_center, self._opacity_duration_var, t, training=training, var_min=self.var_pad/self.interval) \
                            * self.get_motion_opacity)

    def get_opacity_at_t(self, t, mode=0, training=False):
        assert t >= -self.time_shift and t <= self.duration+self.time_shift
        if self._opacity_motion.shape[0] == 0 or mode == 1:
            return self.get_opacity
        elif mode == 2:
            return self.get_motion_opacity_at_t(t, training=training)
        return torch.cat([self.get_opacity,
                          self.get_motion_opacity_at_t(t, training=training)], dim=0).contiguous()

    def apply_gs_mask_at_t(self, t, mode=0, training=False):
        """Apply learned mask to Gaussian parameters"""
        self._check_mask_shape()

        if self.gs_mask_motion.shape[0] == 0 or mode == 1:
            gs_soft_mask = torch.sigmoid(self.gs_mask)
        elif mode == 2:
            gs_soft_mask = torch.sigmoid(self.gs_mask_motion)
        else:
            gs_mask = torch.cat([self.gs_mask, self.gs_mask_motion], dim=0)
            gs_soft_mask = torch.sigmoid(gs_mask)
        gs_hard_mask = ((gs_soft_mask > self.gs_mask_thres).float() - gs_soft_mask).detach() + gs_soft_mask

        # Apply mask
        scales = torch.mul(self.get_scaling(mode=mode), gs_hard_mask)
        opacity = torch.mul(self.get_opacity_at_t(t, mode=mode, training=training), gs_hard_mask)

        return {
            'scale': scales,
            'opacity': opacity,
            'gs_soft_mask': gs_soft_mask,
            'gs_hard_mask': gs_hard_mask,
            'gs_bitmask': gs_soft_mask.flatten() > self.gs_mask_thres
        }

    def get_covariance_at_t(self, t, scaling_modifier = 1, mode=0):
        # pad not required here
        return self.covariance_activation(self.get_scaling(mode=mode), scaling_modifier, self.get_rotation_at_t(t, mode=mode)).contiguous()

    # RDO-related properties and methods
    @property
    def vector_dict(self):
        """Returns a dictionary of flattened vectors for quantization"""
        static_dict = {
            'scale': self.get_static_scaling.flatten(start_dim=1),
            'rot': self._rotation.unsqueeze(1).expand(-1, self.keyframe_num, -1).flatten(start_dim=2),
            'dc': self._features_dc.flatten(start_dim=1),
            'sh1': self._features_rest.tensor_split((3,8), dim=1)[0].flatten(start_dim=1),
            'sh2': self._features_rest.tensor_split((3,8), dim=1)[1].flatten(start_dim=1),
            'sh3': self._features_rest.tensor_split((3,8), dim=1)[2].flatten(start_dim=1)
        }

        # Add dynamic vectors if they exist
        if self._xyz_motion.shape[0] > 0:
            dynamic_dict = {
                'scale': self.get_motion_scaling.flatten(start_dim=1),
                'rot': self._rotation_motion.flatten(start_dim=2),
                'dc': self._features_dc_motion.flatten(start_dim=1),
                'sh1': self._features_rest_motion.tensor_split((3,8), dim=1)[0].flatten(start_dim=1),
                'sh2': self._features_rest_motion.tensor_split((3,8), dim=1)[1].flatten(start_dim=1),
                'sh3': self._features_rest_motion.tensor_split((3,8), dim=1)[2].flatten(start_dim=1)
            }
            # Merge dictionaries (concatenate tensors)
            for key in static_dict:
                static_dict[key] = torch.cat([static_dict[key], dynamic_dict[key]], dim=0)

        return static_dict

    def vq_post_process(self, key, value):
        """Post-process the quantized vectors"""
        if key == 'scale':
            return value
        if key == 'rot':
            return self.rotation_activation(value)
        if key == 'dc':
            return torch.unsqueeze(value, 1)
        if key == 'sh1':
            return torch.reshape(value, (-1, 3, 3))
        if key == 'sh2':
            return torch.reshape(value, (-1, 5, 3))
        if key == 'sh3':
            return torch.reshape(value, (-1, 7, 3))

    def initialize_quantizer(self):
        """Initialize vector quantizers for compression"""
        if not self.vq_cfg:
            return

        vq_cfg = self.vq_cfg
        x_dim = {
            'scale': self._scaling.flatten(start_dim=1).shape[-1],
            'rot': self._rotation.flatten(start_dim=1).shape[-1],
            'dc': self._features_dc.flatten(start_dim=1).shape[-1],
            'sh1': self._features_rest.tensor_split((3,8), dim=1)[0].flatten(start_dim=1).shape[-1],
            'sh2': self._features_rest.tensor_split((3,8), dim=1)[1].flatten(start_dim=1).shape[-1],
            'sh3': self._features_rest.tensor_split((3,8), dim=1)[2].flatten(start_dim=1).shape[-1]
        }

        for key in vq_cfg['keys']:
            self._quantizer[key] = ECVQ(
                x_dim=x_dim[key], cb_dim=x_dim[key], cb_size=vq_cfg['cb_size'][key],
                lmbda=vq_cfg['lmbda'][key], patch_size=vq_cfg['patch_size'],
                rate_constrain=False,
            ).cuda()

    def oneupSHdegree(self):
        if self.active_sh_degree < self.max_sh_degree:
            self.active_sh_degree += 1

    def create_from_pcd(self, pcd, spatial_lr_scale : float):
        self.spatial_lr_scale = spatial_lr_scale
        fused_point_cloud = torch.tensor(np.asarray(pcd.points)).float().cuda()
        fused_color = RGB2SH(torch.tensor(np.asarray(pcd.colors)).float().cuda())
        features = torch.zeros((fused_color.shape[0], 3, (self.max_sh_degree + 1) ** 2)).float().cuda()
        features[:, :3, 0 ] = fused_color
        features[:, 3:, 1:] = 0.0 # why?

        print("Number of points at initialisation : ", fused_point_cloud.shape[0])

        dist2 = torch.clamp_min(distCUDA2(torch.from_numpy(np.asarray(pcd.points)).float().cuda()), 0.0000001)
        scales = torch.log(torch.sqrt(dist2))[...,None].repeat(1, 3)
        rots = torch.zeros((fused_point_cloud.shape[0], 4), device="cuda")
        rots[:, 0] = 1

        opacities = inverse_sigmoid(0.1 * torch.ones((fused_point_cloud.shape[0], 1), dtype=torch.float, device="cuda"))

        self._xyz = nn.Parameter(fused_point_cloud.requires_grad_(True))
        self._features_dc = nn.Parameter(features[:,:,0:1].transpose(1, 2).contiguous().requires_grad_(True))
        self._features_rest = nn.Parameter(features[:,:,1:].transpose(1, 2).contiguous().requires_grad_(True))
        self._scaling = nn.Parameter(scales.requires_grad_(True))
        self._rotation = nn.Parameter(rots.requires_grad_(True))
        self._opacity = nn.Parameter(opacities.requires_grad_(True))
        self.max_radii2D = torch.zeros((self._xyz.shape[0]), device="cuda")
        self.min_radii2D = torch.ones((self._xyz.shape[0]), device="cuda") * 1000
        self._xyz_disp = nn.Parameter(torch.zeros_like(self._xyz).requires_grad_(True))

        self.sh_mask = nn.Parameter(torch.zeros(features.shape[0], 3, device="cuda").requires_grad_(True))
        self.gs_mask = nn.Parameter(torch.zeros(features.shape[0], 1, device="cuda").requires_grad_(True))

    def training_setup(self, training_args):
        self.percent_dense = training_args.percent_dense
        self.xyz_gradient_accum = torch.zeros((self._xyz.shape[0], 1), device="cuda")
        self.denom = torch.zeros((self._xyz.shape[0], 1), device="cuda")
        self.xyz_error_accum = torch.zeros((self._xyz.shape[0], 1), device="cuda")
        self.xyz_error_min = torch.ones((self._xyz.shape[0], 1), device="cuda") * 1000
        self.xyz_error_min_timestamp = torch.ones((self._xyz.shape[0], 1), device="cuda") * -1
        self.xyz_ssim_error_accum = torch.zeros((self._xyz.shape[0], 1), device="cuda")
        self.error_denom = torch.zeros((self._xyz.shape[0], 1), device="cuda")

        self.motion_xyz_gradient_accum = torch.zeros((self._xyz_motion.shape[0], 1), device="cuda")
        self.motion_denom = torch.zeros((self._xyz_motion.shape[0], 1), device="cuda")
        self.motion_xyz_error_min = torch.ones((self._xyz_motion.shape[0], 1), device="cuda") * 1000
        self.motion_xyz_error_mean = torch.zeros((self._xyz_motion.shape[0], 1), device="cuda")
        self.motion_xyz_error_min_timestamp = torch.ones((self._xyz_motion.shape[0], 1), device="cuda") * -1
        self.motion_xyz_ssim_error_accum = torch.zeros((self._xyz_motion.shape[0], 1), device="cuda")
        self.motion_error_denom = torch.zeros((self._xyz_motion.shape[0], 1), device="cuda")
        #######################################################################################################
        l = [
            {'params': [self._xyz], 'lr': training_args.position_lr_init * self.spatial_lr_scale, "name": "xyz"},
            {'params': [self._features_dc], 'lr': training_args.feature_lr, "name": "f_dc"},
            {'params': [self._features_rest], 'lr': training_args.feature_lr / 20.0, "name": "f_rest"},
            {'params': [self._opacity], 'lr': training_args.opacity_lr, "name": "opacity"},
            {'params': [self._scaling], 'lr': training_args.scaling_lr, "name": "scaling"},
            {'params': [self._rotation], 'lr': training_args.rotation_lr, "name": "rotation"},
            {'params': [self._xyz_disp], 'lr': training_args.disp_lr, "name": "xyz_disp"},

            {'params': [self._xyz_motion], 'lr': training_args.dynamic_position_lr_init * self.spatial_lr_scale, "name": "motion_xyz"},
            {'params': [self._features_dc_motion], 'lr': training_args.feature_motion_lr, "name": "motion_f_dc"}, # use the same lr as static points
            {'params': [self._features_rest_motion], 'lr': training_args.feature_motion_lr / 20.0, "name": "motion_f_rest"},
            {'params': [self._scaling_motion], 'lr': training_args.scaling_lr, "name": "motion_scaling"},
            {'params': [self._opacity_motion], 'lr': training_args.opacity_motion_lr, "name": "motion_opacity"},
            {'params': [self._opacity_duration_center], 'lr': training_args.opacity_motion_center_lr, "name": "motion_opacity_center"},
            {'params': [self._opacity_duration_var], 'lr': training_args.opacity_motion_var_lr, "name": "motion_opacity_var"},
            {'params': [self._rotation_motion], 'lr': training_args.rotation_motion_lr, "name": "motion_rotation"}
        ]

        self.optimizer = torch.optim.RAdam(l, lr=0.001)

        self.xyz_scheduler_args = get_expon_lr_func(lr_init=training_args.position_lr_init*self.spatial_lr_scale,
                                                    lr_final=training_args.position_lr_final*self.spatial_lr_scale,
                                                    lr_delay_mult=training_args.position_lr_delay_mult,
                                                    max_steps=training_args.position_lr_max_steps)

        self.xyz_motion_scheduler_args = get_expon_lr_func(lr_init=training_args.dynamic_position_lr_init*self.spatial_lr_scale,
                                                    lr_final=training_args.dynamic_position_lr_final*self.spatial_lr_scale,
                                                    lr_delay_mult=training_args.dynamic_position_lr_delay_mult,
                                                    max_steps=training_args.dynamic_position_lr_max_steps)

        # Initialize masks if they don't exist
        if self.gs_mask is None:
            self.gs_mask = nn.Parameter(torch.zeros(self._xyz.shape[0], 1, device="cuda").requires_grad_(True))

        if self.gs_mask_motion is None:
            self.gs_mask_motion = nn.Parameter(torch.zeros(self._xyz_motion.shape[0], 1, device="cuda").requires_grad_(True))

        if self.sh_mask is None:
            self.sh_mask = nn.Parameter(torch.zeros(self._xyz.shape[0], 3, device="cuda").requires_grad_(True))

        if self.sh_mask_motion is None:
            self.sh_mask_motion = nn.Parameter(torch.zeros(self._xyz_motion.shape[0], 3, device="cuda").requires_grad_(True))


        # Setup VQ optimizers if VQ config is available
        l = [
            {'params': [self._quantizer[key].codebook for key in self._quantizer.keys()], 'lr': self.vq_cfg['cb_lr']},
            {'params': [self._quantizer[key].logits for key in self._quantizer.keys()], 'lr': self.vq_cfg['logits_lr']}
        ]
        self.optimizer_vq = torch.optim.Adam(l, lr=0.0, eps=1e-15)
        self.scheduler_vq = MultiStepLR(self.optimizer_vq, milestones=[8000], gamma=0.1)

        # Setup mask optimizers if training_args has the required attributes
        l = [
            {'params': [self.sh_mask], 'lr': training_args.sh_mask_lr, "name": "sh_mask"},
            {'params': [self.sh_mask_motion], 'lr': training_args.sh_mask_lr, "name": "sh_mask_motion"}
        ]
        self.optimizer_sh_mask = torch.optim.Adam(l, lr=training_args.sh_mask_lr, eps=1e-15)

        l = [
            {'params': [self.gs_mask], 'lr': training_args.gs_mask_lr, "name": "gs_mask"},
            {'params': [self.gs_mask_motion], 'lr': training_args.gs_mask_lr, "name": "gs_mask_motion"}
        ]
        self.optimizer_gs_mask = torch.optim.Adam(l, lr=training_args.gs_mask_lr, eps=1e-15)

    def update_learning_rate(self, iteration):
        ''' Learning rate scheduling per step '''
        for param_group in self.optimizer.param_groups:
            if param_group["name"] == "xyz":
                lr = self.xyz_scheduler_args(iteration)
                param_group['lr'] = lr
                return lr
            elif param_group["name"] == "motion_xyz":
                lr = self.xyz_motion_scheduler_args(iteration)
                param_group['lr'] = lr
                return lr

    def construct_list_of_static_attributes(self):
        l = ['x', 'y', 'z', 'nx', 'ny', 'nz']
        # All channels except the 3 DC
        for i in range(self._features_dc.shape[1]*self._features_dc.shape[2]):
            l.append('f_dc_{}'.format(i))
        for i in range(self._features_rest.shape[1]*self._features_rest.shape[2]):
            l.append('f_rest_{}'.format(i))
        l.append('opacity')
        for i in range(self._scaling.shape[1]):
            l.append('scale_{}'.format(i))
        for i in range(self._rotation.shape[1]):
            l.append('rot_{}'.format(i))
        for i in range(self._xyz_disp.shape[1]):
            l.append('xyz_disp_{}'.format(i))

        return l

    def construct_list_of_dynamic_attributes(self):
        l2 = []
        # attributes for dynamic points.
        for i in range(self._xyz_motion.shape[1]):
            for j in range(self._xyz_motion.shape[2]):
                l2.append('motion_xyz_{}_{}'.format(i, j))
        for i in range(self._features_dc_motion.shape[1]*self._features_dc_motion.shape[2]):
            l2.append('motion_f_dc_{}'.format(i))
        for i in range(self._features_rest_motion.shape[1]*self._features_rest_motion.shape[2]):
            l2.append('motion_f_rest_{}'.format(i))
        for i in range(self._scaling_motion.shape[1]):
            l2.append('motion_scale_{}'.format(i))
        l2.append('motion_opacity')
        for i in range(self._opacity_duration_center.shape[1]):
            l2.append('motion_opacity_c_{}'.format(i))
        for i in range(self._opacity_duration_var.shape[1]):
            l2.append('motion_opacity_v_{}'.format(i))
        for i in range(self._rotation_motion.shape[1]):
            for j in range(self._rotation_motion.shape[2]):
                l2.append('motion_rot_{}_{}'.format(i, j))

        return l2

    def save_ply(self, path):
        mkdir_p(os.path.dirname(path))

        xyz = self._xyz.detach().cpu().numpy()
        normals = np.zeros_like(xyz)
        f_dc = self._features_dc.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        f_rest = self._features_rest.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        opacities = self._opacity.detach().cpu().numpy()
        scale = self._scaling.detach().cpu().numpy()
        rotation = self._rotation.detach().cpu().numpy()
        xyz_disp = self._xyz_disp.detach().cpu().numpy()

        static_dtype_full = [(attribute, 'f4') for attribute in self.construct_list_of_static_attributes()]

        static_elements = np.empty(xyz.shape[0], dtype=static_dtype_full)
        static_attributes = np.concatenate((xyz, normals, f_dc, f_rest, opacities, scale, rotation, xyz_disp), axis=1)
        static_elements[:] = list(map(tuple, static_attributes))
        s_el = PlyElement.describe(static_elements, 'vertex')
        PlyData([s_el]).write(path)

        xyz_motion = self._xyz_motion.detach().flatten(start_dim=1).contiguous().cpu().numpy()
        f_dc_motion = self._features_dc_motion.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        f_rest_motion = self._features_rest_motion.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        scaling_motion = self._scaling_motion.detach().cpu().numpy()
        opacity_motion = self._opacity_motion.detach().cpu().numpy()
        opacity_duration_center = self._opacity_duration_center.detach().flatten(start_dim=1).cpu().numpy()
        opacity_duration_var = self._opacity_duration_var.detach().flatten(start_dim=1).cpu().numpy()
        rotation_motion = self._rotation_motion.detach().flatten(start_dim=1).contiguous().cpu().numpy()

        dynamic_dtype_full = [(attribute, 'f4') for attribute in self.construct_list_of_dynamic_attributes()]
        dynamic_elements = np.empty(xyz_motion.shape[0], dtype=dynamic_dtype_full)
        dynamic_attributes = np.concatenate((xyz_motion, f_dc_motion, f_rest_motion, scaling_motion, opacity_motion, opacity_duration_center, opacity_duration_var, rotation_motion), axis=1)
        dynamic_elements[:] = list(map(tuple, dynamic_attributes))
        d_el = PlyElement.describe(dynamic_elements, 'vertex')
        PlyData([d_el]).write(path.replace('point_cloud.ply', 'dynamic_point_cloud.ply'))

    def reset_opacity(self):
        opacities_new = inverse_sigmoid(torch.min(self.get_opacity, torch.ones_like(self.get_opacity)*0.85))
        optimizable_tensors = self.replace_tensor_to_optimizer({"opacity": opacities_new})
        self._opacity = optimizable_tensors["opacity"]

        if self._opacity_motion.shape[0] == 0:
            return
        motion_opacities_new = inverse_sigmoid(torch.min(self.get_motion_opacity, torch.ones_like(self.get_motion_opacity)*0.95))
        optimizable_tensors = self.replace_tensor_to_optimizer({"motion_opacity": motion_opacities_new})
        self._opacity_motion = optimizable_tensors["motion_opacity"]

    def load_ply(self, path):
        plydata = PlyData.read(path)

        xyz = np.stack((np.asarray(plydata.elements[0]["x"]),
                        np.asarray(plydata.elements[0]["y"]),
                        np.asarray(plydata.elements[0]["z"])),  axis=1)
        opacities = np.asarray(plydata.elements[0]["opacity"])[..., np.newaxis]

        features_dc = np.zeros((xyz.shape[0], 3, 1))
        features_dc[:, 0, 0] = np.asarray(plydata.elements[0]["f_dc_0"])
        features_dc[:, 1, 0] = np.asarray(plydata.elements[0]["f_dc_1"])
        features_dc[:, 2, 0] = np.asarray(plydata.elements[0]["f_dc_2"])

        extra_f_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("f_rest_")]
        extra_f_names = sorted(extra_f_names, key = lambda x: int(x.split('_')[-1]))
        assert len(extra_f_names)==3*(self.max_sh_degree + 1) ** 2 - 3
        features_extra = np.zeros((xyz.shape[0], len(extra_f_names)))
        for idx, attr_name in enumerate(extra_f_names):
            features_extra[:, idx] = np.asarray(plydata.elements[0][attr_name])
        # Reshape (P,F*SH_coeffs) to (P, F, SH_coeffs except DC)
        features_extra = features_extra.reshape((features_extra.shape[0], 3, (self.max_sh_degree + 1) ** 2 - 1))

        scale_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("scale_")]
        scale_names = sorted(scale_names, key = lambda x: int(x.split('_')[-1]))
        scales = np.zeros((xyz.shape[0], len(scale_names)))
        for idx, attr_name in enumerate(scale_names):
            scales[:, idx] = np.asarray(plydata.elements[0][attr_name])

        rot_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("rot_")]
        rot_names = sorted(rot_names, key = lambda x: int(x.split('_')[-1]))
        rots = np.zeros((xyz.shape[0], len(rot_names)))
        for idx, attr_name in enumerate(rot_names):
            rots[:, idx] = np.asarray(plydata.elements[0][attr_name])

        xyz_disp = np.zeros((xyz.shape[0], 3))
        xyz_disp[:, 0] = np.asarray(plydata.elements[0]["xyz_disp_0"])
        xyz_disp[:, 1] = np.asarray(plydata.elements[0]["xyz_disp_1"])
        xyz_disp[:, 2] = np.asarray(plydata.elements[0]["xyz_disp_2"])

        # dynamic points
        plydata_dynamic = PlyData.read(path.replace('point_cloud.ply', 'dynamic_point_cloud.ply'))
        motion_opacities = np.asarray(plydata_dynamic.elements[0]["motion_opacity"])[..., np.newaxis]
        num_points = motion_opacities.shape[0]
        self.keyframe_num = math.ceil((self.duration + self.time_shift + self.time_pad*2 + 1) / self.interval) + 1 + 4

        motion_xyz_names = [p.name for p in plydata_dynamic.elements[0].properties if p.name.startswith("motion_xyz_")]
        motion_xyz_names = sorted(motion_xyz_names, key = lambda x: ( int(x.split('_')[-2]), int(x.split('_')[-1]) ))

        motion_xyz = np.zeros((num_points, self.keyframe_num * 3 * self.motion_degree))
        for idx, attr_name in enumerate(motion_xyz_names):
            motion_xyz[:, idx] = np.asarray(plydata_dynamic.elements[0][attr_name])
        motion_xyz = motion_xyz.reshape((num_points, self.keyframe_num, 3 * self.motion_degree))

        motion_features_dc = np.zeros((num_points, 3, 1))
        motion_features_dc[:, 0, 0] = np.asarray(plydata_dynamic.elements[0]["motion_f_dc_0"])
        motion_features_dc[:, 1, 0] = np.asarray(plydata_dynamic.elements[0]["motion_f_dc_1"])
        motion_features_dc[:, 2, 0] = np.asarray(plydata_dynamic.elements[0]["motion_f_dc_2"])

        motion_f_rest_names = [p.name for p in plydata_dynamic.elements[0].properties if p.name.startswith("motion_f_rest_")]
        motion_f_rest_names = sorted(motion_f_rest_names, key = lambda x: int(x.split('_')[-1]))
        assert len(motion_f_rest_names)==3*(self.max_sh_degree + 1) ** 2 - 3
        motion_features_extra = np.zeros((num_points, len(motion_f_rest_names)))
        for idx, attr_name in enumerate(motion_f_rest_names):
            motion_features_extra[:, idx] = np.asarray(plydata_dynamic.elements[0][attr_name])
        # Reshape (P,F*SH_coeffs) to (P, F, SH_coeffs except DC)
        motion_features_extra = motion_features_extra.reshape((motion_features_extra.shape[0], 3, (self.max_sh_degree + 1) ** 2 - 1))

        motion_scale_names = [p.name for p in plydata_dynamic.elements[0].properties if p.name.startswith("motion_scale_")]
        motion_scale_names = sorted(motion_scale_names, key = lambda x: int(x.split('_')[-1]))
        motion_scales = np.zeros((num_points, 3))
        for idx, attr_name in enumerate(motion_scale_names):
            motion_scales[:, idx] = np.asarray(plydata_dynamic.elements[0][attr_name])

        motion_opacity_c_names = [p.name for p in plydata_dynamic.elements[0].properties if p.name.startswith("motion_opacity_c_")]
        motion_opacity_c_names = sorted(motion_opacity_c_names, key = lambda x: int(x.split('_')[-1] ))
        motion_opacities_c = np.zeros((num_points, self.opacity_degree, 1))
        for idx, attr_name in enumerate(motion_opacity_c_names):
            motion_opacities_c[:, idx, 0] = np.asarray(plydata_dynamic.elements[0][attr_name])

        motion_opacity_v_names = [p.name for p in plydata_dynamic.elements[0].properties if p.name.startswith("motion_opacity_v_")]
        motion_opacity_v_names = sorted(motion_opacity_v_names, key = lambda x: int(x.split('_')[-1] ))
        motion_opacities_v = np.zeros((num_points, self.opacity_degree, 1))
        for idx, attr_name in enumerate(motion_opacity_v_names):
            motion_opacities_v[:, idx, 0] = np.asarray(plydata_dynamic.elements[0][attr_name])

        motion_rot_names = [p.name for p in plydata_dynamic.elements[0].properties if p.name.startswith("motion_rot_")]
        motion_rot_names = sorted(motion_rot_names, key = lambda x: ( int(x.split('_')[-2]), int(x.split('_')[-1]) ))
        motion_rots = np.zeros((num_points, self.keyframe_num * 4))
        for idx, attr_name in enumerate(motion_rot_names):
            motion_rots[:, idx] = np.asarray(plydata_dynamic.elements[0][attr_name])
        motion_rots = motion_rots.reshape((num_points, self.keyframe_num, 4))

        self._xyz = nn.Parameter(torch.tensor(xyz, dtype=torch.float, device="cuda").requires_grad_(True))
        self._features_dc = nn.Parameter(torch.tensor(features_dc, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(True))
        self._features_rest = nn.Parameter(torch.tensor(features_extra, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(True))
        self._opacity = nn.Parameter(torch.tensor(opacities, dtype=torch.float, device="cuda").requires_grad_(True))
        self._scaling = nn.Parameter(torch.tensor(scales, dtype=torch.float, device="cuda").requires_grad_(True))
        self._rotation = nn.Parameter(torch.tensor(rots, dtype=torch.float, device="cuda").requires_grad_(True))
        self._xyz_disp = nn.Parameter(torch.tensor(xyz_disp, dtype=torch.float, device="cuda").requires_grad_(True))

        # dynamic points
        self._xyz_motion = nn.Parameter(torch.tensor(motion_xyz, dtype=torch.float, device="cuda").requires_grad_(True))
        self._features_dc_motion = nn.Parameter(torch.tensor(motion_features_dc, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(True))
        self._features_rest_motion = nn.Parameter(torch.tensor(motion_features_extra, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(True))
        self._scaling_motion = nn.Parameter(torch.tensor(motion_scales, dtype=torch.float, device="cuda").requires_grad_(True))
        self._opacity_motion = nn.Parameter(torch.tensor(motion_opacities, dtype=torch.float, device="cuda").requires_grad_(True))
        self._opacity_duration_center = nn.Parameter(torch.tensor(motion_opacities_c, dtype=torch.float, device="cuda").requires_grad_(True))
        self._opacity_duration_var = nn.Parameter(torch.tensor(motion_opacities_v, dtype=torch.float, device="cuda").requires_grad_(True))
        self._rotation_motion = nn.Parameter(torch.tensor(motion_rots, dtype=torch.float, device="cuda").requires_grad_(True))

        # masks
        self.sh_mask = nn.Parameter(torch.zeros(xyz.shape[0], 3, device="cuda").requires_grad_(True))
        self.gs_mask = nn.Parameter(torch.zeros(xyz.shape[0], 1, device="cuda").requires_grad_(True))
        self.sh_mask_motion = nn.Parameter(torch.zeros(motion_xyz.shape[0], 3, device="cuda").requires_grad_(True))
        self.gs_mask_motion = nn.Parameter(torch.zeros(motion_xyz.shape[0], 1, device="cuda").requires_grad_(True))
        self.active_sh_degree = self.max_sh_degree

    def replace_tensor_to_optimizer(self, tensor_dict):
        optimizable_tensors = {}
        for name, tensor in tensor_dict.items():
            for group in self.optimizer.param_groups:
                if group["name"] == name:
                    stored_state = self.optimizer.state.get(group['params'][0], None)
                    if stored_state is not None:
                        stored_state["exp_avg"] = torch.zeros_like(tensor)
                        stored_state["exp_avg_sq"] = torch.zeros_like(tensor)

                        del self.optimizer.state[group['params'][0]]
                        group["params"][0] = nn.Parameter(tensor.requires_grad_(True))
                        self.optimizer.state[group['params'][0]] = stored_state

                        optimizable_tensors[group["name"]] = group["params"][0]
                    else:
                        group["params"][0] = nn.Parameter(group["params"][0].requires_grad_(True))
                        optimizable_tensors[group["name"]] = group["params"][0]

        return optimizable_tensors

    def _prune_optimizer(self, static_mask, dynamic_mask):
        self._check_mask_shape()

        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            if group["name"].startswith("motion_"):
                mask = dynamic_mask
            else:
                mask = static_mask
            stored_state = self.optimizer.state.get(group['params'][0], None)
            if stored_state is not None:
                stored_state["exp_avg"] = stored_state["exp_avg"][mask]
                stored_state["exp_avg_sq"] = stored_state["exp_avg_sq"][mask]

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter((group["params"][0][mask].requires_grad_(True)))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(group["params"][0][mask].requires_grad_(True))
                optimizable_tensors[group["name"]] = group["params"][0]

        for group in self.optimizer_gs_mask.param_groups:
            if group["name"] == "gs_mask_motion":
                mask = dynamic_mask
            else:
                mask = static_mask
            stored_state = self.optimizer_gs_mask.state.get(group['params'][0], None)
            if stored_state is not None:
                stored_state["exp_avg"] = stored_state["exp_avg"][mask]
                stored_state["exp_avg_sq"] = stored_state["exp_avg_sq"][mask]

                del self.optimizer_gs_mask.state[group['params'][0]]
                group["params"][0] = nn.Parameter((group["params"][0][mask].requires_grad_(True)))
                self.optimizer_gs_mask.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(group["params"][0][mask].requires_grad_(True))
                optimizable_tensors[group["name"]] = group["params"][0]

        for group in self.optimizer_sh_mask.param_groups:
            if group["name"] == "sh_mask_motion":
                mask = dynamic_mask
            else:
                mask = static_mask
            stored_state = self.optimizer_sh_mask.state.get(group['params'][0], None)
            if stored_state is not None:
                stored_state["exp_avg"] = stored_state["exp_avg"][mask]
                stored_state["exp_avg_sq"] = stored_state["exp_avg_sq"][mask]

                del self.optimizer_sh_mask.state[group['params'][0]]
                group["params"][0] = nn.Parameter((group["params"][0][mask].requires_grad_(True)))
                self.optimizer_sh_mask.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(group["params"][0][mask].requires_grad_(True))
                optimizable_tensors[group["name"]] = group["params"][0]

        self._check_mask_shape()
        return optimizable_tensors

    def _check_mask_shape(self):
        if self._xyz.shape[0] != self.gs_mask.shape[0] or self._xyz.shape[0] != self.sh_mask.shape[0]:
            raise ValueError(f"xyz shape {self._xyz.shape[0]} does not match the mask shape {self.gs_mask.shape[0]} or {self.sh_mask.shape[0]}")
        if self._xyz_motion.shape[0] != self.gs_mask_motion.shape[0] or self._xyz_motion.shape[0] != self.sh_mask_motion.shape[0]:
            raise ValueError(f"xyz_motion shape {self._xyz_motion.shape[0]} does not match the mask shape {self.gs_mask_motion.shape[0]} or {self.sh_mask_motion.shape[0]}")

    def prune_points(self, static_mask, dynamic_mask):
        self._check_mask_shape()

        vaild_static_mask = ~static_mask
        if dynamic_mask.shape[0] == 0:
            dynamic_mask = torch.empty(0, dtype=torch.bool, device=static_mask.device)
        vaild_dynamic_mask = ~dynamic_mask

        optimizable_tensors = self._prune_optimizer(vaild_static_mask, vaild_dynamic_mask)

        self._xyz = optimizable_tensors["xyz"]
        self._features_dc = optimizable_tensors["f_dc"]
        self._features_rest = optimizable_tensors["f_rest"]
        self._opacity = optimizable_tensors["opacity"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]
        self._xyz_disp = optimizable_tensors["xyz_disp"]

        self.xyz_gradient_accum = self.xyz_gradient_accum[vaild_static_mask]
        self.denom = self.denom[vaild_static_mask]
        self.xyz_error_accum = self.xyz_error_accum[vaild_static_mask]
        self.xyz_error_min = self.xyz_error_min[vaild_static_mask]
        self.xyz_error_min_timestamp = self.xyz_error_min_timestamp[vaild_static_mask]
        self.xyz_ssim_error_accum = self.xyz_ssim_error_accum[vaild_static_mask]
        self.error_denom = self.error_denom[vaild_static_mask]
        self.max_radii2D = self.max_radii2D[vaild_static_mask]
        self.min_radii2D = self.min_radii2D[vaild_static_mask]

        self.gs_mask = optimizable_tensors["gs_mask"]
        self.sh_mask = optimizable_tensors["sh_mask"]

        self._check_mask_shape()

        # dynamic points
        if vaild_dynamic_mask.shape[0] == 0:
            return

        self._xyz_motion = optimizable_tensors["motion_xyz"]
        self._features_dc_motion = optimizable_tensors["motion_f_dc"]
        self._features_rest_motion = optimizable_tensors["motion_f_rest"]
        self._scaling_motion = optimizable_tensors["motion_scaling"]
        self._opacity_motion = optimizable_tensors["motion_opacity"]
        self._opacity_duration_center = optimizable_tensors["motion_opacity_center"]
        self._opacity_duration_var = optimizable_tensors["motion_opacity_var"]
        self._rotation_motion = optimizable_tensors["motion_rotation"]

        self.motion_xyz_gradient_accum = self.motion_xyz_gradient_accum[vaild_dynamic_mask]
        self.motion_denom = self.motion_denom[vaild_dynamic_mask]
        self.motion_xyz_error_min = self.motion_xyz_error_min[vaild_dynamic_mask]
        self.motion_xyz_error_mean = self.motion_xyz_error_mean[vaild_dynamic_mask]
        self.motion_xyz_error_min_timestamp = self.motion_xyz_error_min_timestamp[vaild_dynamic_mask]
        self.motion_xyz_ssim_error_accum = self.motion_xyz_ssim_error_accum[vaild_dynamic_mask]
        self.motion_error_denom = self.motion_error_denom[vaild_dynamic_mask]
        self.motion_max_radii2D = self.motion_max_radii2D[vaild_dynamic_mask]
        self.motion_min_radii2D = self.motion_min_radii2D[vaild_dynamic_mask]

        self.gs_mask_motion = optimizable_tensors["gs_mask_motion"]
        self.sh_mask_motion = optimizable_tensors["sh_mask_motion"]

        self._check_mask_shape()

    def cat_tensors_to_optimizer(self, tensors_dict):
        self._check_mask_shape()
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            assert len(group["params"]) == 1
            # warning this may skip invaild paramgroups
            if not group["name"] in tensors_dict.keys():
                continue
            extension_tensor = tensors_dict[group["name"]]
            stored_state = self.optimizer.state.get(group['params'][0], None)
            if stored_state is not None:
                stored_state["exp_avg"] = torch.cat((stored_state["exp_avg"], torch.zeros_like(extension_tensor)), dim=0)
                stored_state["exp_avg_sq"] = torch.cat((stored_state["exp_avg_sq"], torch.zeros_like(extension_tensor)), dim=0)

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter(torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
                optimizable_tensors[group["name"]] = group["params"][0]

        for group in self.optimizer_gs_mask.param_groups:
            assert len(group["params"]) == 1
            if not group["name"] in tensors_dict.keys():
                continue
            extension_tensor = tensors_dict[group["name"]]
            stored_state = self.optimizer_gs_mask.state.get(group['params'][0], None)
            if stored_state is not None:

                stored_state["exp_avg"] = torch.cat((stored_state["exp_avg"], torch.zeros_like(extension_tensor)), dim=0)
                stored_state["exp_avg_sq"] = torch.cat((stored_state["exp_avg_sq"], torch.zeros_like(extension_tensor)), dim=0)

                del self.optimizer_gs_mask.state[group['params'][0]]
                group["params"][0] = nn.Parameter(torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
                self.optimizer_gs_mask.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
                optimizable_tensors[group["name"]] = group["params"][0]

        for group in self.optimizer_sh_mask.param_groups:
            assert len(group["params"]) == 1
            if not group["name"] in tensors_dict.keys():
                continue
            extension_tensor = tensors_dict[group["name"]]
            stored_state = self.optimizer_sh_mask.state.get(group['params'][0], None)
            if stored_state is not None:

                stored_state["exp_avg"] = torch.cat((stored_state["exp_avg"], torch.zeros_like(extension_tensor)), dim=0)
                stored_state["exp_avg_sq"] = torch.cat((stored_state["exp_avg_sq"], torch.zeros_like(extension_tensor)), dim=0)

                del self.optimizer_sh_mask.state[group['params'][0]]
                group["params"][0] = nn.Parameter(torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
                self.optimizer_sh_mask.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
                optimizable_tensors[group["name"]] = group["params"][0]

        self._check_mask_shape()
        return optimizable_tensors

    def densification_postfix(self, new_xyz, new_features_dc, new_features_rest, new_opacities, new_scaling, new_rotation, new_xyz_disp,
                              new_xyz_motion, new_features_dc_motion, new_features_rest_motion, new_scaling_motion, new_opacity_motion,
                              new_opacity_duration_center, new_opacity_duration_var, new_rotation_motion, new_gs_mask, new_sh_mask, new_gs_mask_motion, new_sh_mask_motion
                              ):
        self._check_mask_shape()
        d = {"xyz": new_xyz,
        "f_dc": new_features_dc,
        "f_rest": new_features_rest,
        "opacity": new_opacities,
        "scaling" : new_scaling,
        "rotation" : new_rotation,
        "xyz_disp" : new_xyz_disp,

        "motion_xyz" : new_xyz_motion,
        "motion_f_dc" : new_features_dc_motion,
        "motion_f_rest" : new_features_rest_motion,
        "motion_scaling" : new_scaling_motion,
        "motion_opacity" : new_opacity_motion,
        "motion_opacity_center" : new_opacity_duration_center,
        "motion_opacity_var" : new_opacity_duration_var,
        "motion_rotation" : new_rotation_motion,

        "gs_mask" : new_gs_mask,
        "gs_mask_motion" : new_gs_mask_motion,
        "sh_mask" : new_sh_mask,
        "sh_mask_motion" : new_sh_mask_motion
        }

        optimizable_tensors = self.cat_tensors_to_optimizer(d)
        self._xyz = optimizable_tensors["xyz"]
        self._features_dc = optimizable_tensors["f_dc"]
        self._features_rest = optimizable_tensors["f_rest"]
        self._opacity = optimizable_tensors["opacity"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]
        self._xyz_disp = optimizable_tensors["xyz_disp"]

        # dynamic points
        self._xyz_motion = optimizable_tensors["motion_xyz"]
        self._features_dc_motion = optimizable_tensors["motion_f_dc"]
        self._features_rest_motion = optimizable_tensors["motion_f_rest"]
        self._scaling_motion = optimizable_tensors["motion_scaling"]
        self._opacity_motion = optimizable_tensors["motion_opacity"]
        self._opacity_duration_center = optimizable_tensors["motion_opacity_center"]
        self._opacity_duration_var = optimizable_tensors["motion_opacity_var"]
        self._rotation_motion = optimizable_tensors["motion_rotation"]

        self.xyz_gradient_accum = torch.zeros((self._xyz.shape[0], 1), device="cuda")
        self.denom = torch.zeros((self._xyz.shape[0], 1), device="cuda")
        self.xyz_error_accum = torch.zeros((self._xyz.shape[0], 1), device="cuda")
        self.xyz_ssim_error_accum = torch.zeros((self._xyz.shape[0], 1), device="cuda")
        self.error_denom = torch.zeros((self._xyz.shape[0], 1), device="cuda")
        self.max_radii2D = torch.zeros((self._xyz.shape[0]), device="cuda")
        self.min_radii2D = torch.ones((self._xyz.shape[0]), device="cuda") * 1000

        self.motion_xyz_gradient_accum = torch.zeros((self._xyz_motion.shape[0], 1), device="cuda")
        self.motion_denom = torch.zeros((self._xyz_motion.shape[0], 1), device="cuda")
        self.motion_xyz_error_mean = torch.zeros((self._xyz_motion.shape[0], 1), device="cuda")
        self.motion_xyz_ssim_error_accum = torch.zeros((self._xyz_motion.shape[0], 1), device="cuda")
        self.motion_error_denom = torch.zeros((self._xyz_motion.shape[0], 1), device="cuda")
        self.motion_max_radii2D = torch.zeros((self._xyz_motion.shape[0]), device="cuda")
        self.motion_min_radii2D = torch.ones((self._xyz_motion.shape[0]), device="cuda") * 1000

        self.gs_mask = optimizable_tensors["gs_mask"]
        self.sh_mask = optimizable_tensors["sh_mask"]
        self.gs_mask_motion = optimizable_tensors["gs_mask_motion"]
        self.sh_mask_motion = optimizable_tensors["sh_mask_motion"]
        self._check_mask_shape()

    def densification_postfix_onlystatic(self, new_xyz, new_features_dc, new_features_rest, new_opacities, new_scaling, new_rotation, new_xyz_disp, new_gs_mask, new_sh_mask
                              ):
        self._check_mask_shape()
        d = {"xyz": new_xyz,
        "f_dc": new_features_dc,
        "f_rest": new_features_rest,
        "opacity": new_opacities,
        "scaling" : new_scaling,
        "rotation" : new_rotation,
        "xyz_disp" : new_xyz_disp,

        "gs_mask" : new_gs_mask,
        "sh_mask" : new_sh_mask
        }

        optimizable_tensors = self.cat_tensors_to_optimizer(d)
        self._xyz = optimizable_tensors["xyz"]
        self._features_dc = optimizable_tensors["f_dc"]
        self._features_rest = optimizable_tensors["f_rest"]
        self._opacity = optimizable_tensors["opacity"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]
        self._xyz_disp = optimizable_tensors["xyz_disp"]

        self.xyz_gradient_accum = torch.zeros((self._xyz.shape[0], 1), device="cuda")
        self.denom = torch.zeros((self._xyz.shape[0], 1), device="cuda")
        self.xyz_error_accum = torch.zeros((self._xyz.shape[0], 1), device="cuda")
        self.xyz_ssim_error_accum = torch.zeros((self._xyz.shape[0], 1), device="cuda")
        self.error_denom = torch.zeros((self._xyz.shape[0], 1), device="cuda")
        self.max_radii2D = torch.zeros((self._xyz.shape[0]), device="cuda")
        self.min_radii2D = torch.ones((self._xyz.shape[0]), device="cuda") * 1000

        self.gs_mask = optimizable_tensors["gs_mask"]
        self.sh_mask = optimizable_tensors["sh_mask"]
        self._check_mask_shape()

    def densify_and_split(self, static_grads, dynamic_grads, dynamic_ssim_errors, grad_threshold, dynamic_grad_threshold, scene_extent, max_screen_size, max_dynamic_screen_size, N=2):
        self._check_mask_shape()
        n_init_points = self._xyz.shape[0]
        # Extract points that satisfy the gradient condition
        padded_static_grad = torch.zeros((n_init_points), device="cuda")
        padded_static_grad[:static_grads.shape[0]] = static_grads.squeeze()
        selected_static_pts_mask = torch.where(padded_static_grad >= grad_threshold, True, False)
        selected_static_pts_mask = torch.logical_and(selected_static_pts_mask,
                                              torch.max(self.get_static_scaling, dim=1).values > self.percent_dense*scene_extent)

        if max_screen_size:
            big_points_vs = self.max_radii2D > max_screen_size
            big_points_ws = self.get_static_scaling.max(dim=1).values > 0.1 * scene_extent
            selected_static_pts_mask = torch.logical_or(torch.logical_or(selected_static_pts_mask, big_points_vs), big_points_ws)

        stds = self.get_static_scaling[selected_static_pts_mask].repeat(N,1)
        means =torch.zeros((stds.size(0), 3),device="cuda")
        samples = torch.normal(mean=means, std=stds)
        rots = build_rotation(self._rotation[selected_static_pts_mask]).repeat(N,1,1)

        new_xyz = torch.bmm(rots, samples.unsqueeze(-1)).squeeze(-1) + self._xyz[selected_static_pts_mask].repeat(N, 1)
        new_scaling = self.scaling_inverse_activation(self.get_static_scaling[selected_static_pts_mask].repeat(N,1) / (0.8*N))
        new_rotation = self._rotation[selected_static_pts_mask].repeat(N,1)
        new_features_dc = self._features_dc[selected_static_pts_mask].repeat(N,1,1)
        new_features_rest = self._features_rest[selected_static_pts_mask].repeat(N,1,1)
        new_opacity = self._opacity[selected_static_pts_mask].repeat(N,1)
        new_xyz_disp = self._xyz_disp[selected_static_pts_mask].repeat(N,1)

        static_prune_filter = torch.cat((selected_static_pts_mask, torch.zeros(N * selected_static_pts_mask.sum(), device="cuda", dtype=bool)))

        new_xyz_error_min = torch.ones((selected_static_pts_mask.sum() * N, 1), device="cuda") * 1000
        new_xyz_error_min_timestamp = torch.ones((selected_static_pts_mask.sum() * N, 1), device="cuda") * -1

        self.xyz_error_min = torch.cat([self.xyz_error_min, new_xyz_error_min])
        self.xyz_error_min_timestamp = torch.cat([self.xyz_error_min_timestamp, new_xyz_error_min_timestamp])

        new_gs_mask = self.gs_mask[selected_static_pts_mask].repeat(N,1)
        new_sh_mask = self.sh_mask[selected_static_pts_mask].repeat(N,1)

        if self._xyz_motion.shape[0] != 0:
            # dynamic points
            n_init_points = self._xyz_motion.shape[0]
            # Extract points that satisfy the gradient condition
            padded_dynamic_grad = torch.zeros((n_init_points), device="cuda")
            padded_dynamic_grad[:dynamic_grads.shape[0]] = dynamic_grads.squeeze()
            selected_dynamic_pts_mask = torch.where(padded_dynamic_grad >= dynamic_grad_threshold, True, False)
            selected_dynamic_pts_mask = torch.logical_and(selected_dynamic_pts_mask,
                                                torch.max(self.get_motion_scaling, dim=1).values > self.percent_dense*scene_extent)

            if max_dynamic_screen_size:
                big_points_vs = self.motion_max_radii2D > max_dynamic_screen_size
                big_points_ws = self.get_motion_scaling.max(dim=1).values > 0.1 * scene_extent
                selected_dynamic_pts_mask = torch.logical_or(torch.logical_or(selected_dynamic_pts_mask, big_points_vs), big_points_ws)

            # we select rot matrix where median point of opacity
            stds = self.get_motion_scaling[selected_dynamic_pts_mask].repeat(N,1)*2
            means = torch.zeros((stds.size(0), 3),device="cuda")
            samples = torch.normal(mean=means, std=stds).unsqueeze(1).repeat(1,self.keyframe_num,1).view(-1,3)
            rots_motion = build_rotation(self._rotation_motion[selected_dynamic_pts_mask].view(-1, 4)).reshape(-1,self.keyframe_num,3,3).repeat(N,1,1,1).view(-1,3,3)
            if self.motion_degree == 1:
                new_xyz_motion = torch.bmm(rots_motion, samples.unsqueeze(-1)).squeeze(-1).view(-1,self.keyframe_num,3) + self._xyz_motion[selected_dynamic_pts_mask].repeat(N,1,1)
            else:
                new_xyz_motion = torch.zeros_like(self._xyz_motion[selected_dynamic_pts_mask]).repeat(N,1,1)
                new_xyz_motion[..., :3] += torch.bmm(rots_motion, samples.unsqueeze(-1)).squeeze(-1).view(-1,self.keyframe_num,3) + self._xyz_motion[selected_dynamic_pts_mask][..., :3].repeat(N,1,1)

            new_scaling_motion = self.scaling_inverse_activation(self.get_motion_scaling[selected_dynamic_pts_mask].repeat(N,1) / (0.8*N))
            new_rotation_motion = self._rotation_motion[selected_dynamic_pts_mask].repeat(N,1,1)
            new_features_dc_motion = self._features_dc_motion[selected_dynamic_pts_mask].repeat(N,1,1)
            new_features_rest_motion = self._features_rest_motion[selected_dynamic_pts_mask].repeat(N,1,1)
            new_opacity_motion = self._opacity_motion[selected_dynamic_pts_mask].repeat(N,1)
            new_opacity_duration_center = self._opacity_duration_center[selected_dynamic_pts_mask].repeat(N,1,1)
            new_opacity_duration_center_len = ((new_opacity_duration_center[:, 1] - new_opacity_duration_center[:, 0]).abs() / 3).clamp_min(2/self.interval)
            new_opacity_duration_center[:, 1] = new_opacity_duration_center[:, 1] + (new_opacity_duration_center_len * (torch.randn_like(new_opacity_duration_center[:, 1])))
            new_opacity_duration_center[:, 0] = new_opacity_duration_center[:, 0] + (new_opacity_duration_center_len * (torch.randn_like(new_opacity_duration_center[:, 0])))
            new_opacity_duration_center = new_opacity_duration_center.clamp((self.time_shift + 1)/self.interval, (self.time_shift+self.duration - 1)/self.interval)
            new_opacity_duration_var = torch.ones_like(self._opacity_duration_var[selected_dynamic_pts_mask].repeat(N,1,1)) * 2
            dynamic_prune_filter = torch.cat((selected_dynamic_pts_mask, torch.zeros(N * selected_dynamic_pts_mask.sum(), device="cuda", dtype=bool)))

            new_motion_xyz_error_min = torch.ones((selected_dynamic_pts_mask.sum() * N, 1), device="cuda") * 1000
            new_motion_xyz_error_min_timestamp = torch.ones((selected_dynamic_pts_mask.sum() * N, 1), device="cuda") * -1

            self.motion_xyz_error_min = torch.cat([self.motion_xyz_error_min, new_motion_xyz_error_min])
            self.motion_xyz_error_min_timestamp = torch.cat([self.motion_xyz_error_min_timestamp, new_motion_xyz_error_min_timestamp])

            new_gs_mask_motion = self.gs_mask_motion[selected_dynamic_pts_mask].repeat(N,1)
            new_sh_mask_motion = self.sh_mask_motion[selected_dynamic_pts_mask].repeat(N,1)
            self.densification_postfix(
                new_xyz, new_features_dc, new_features_rest, new_opacity, new_scaling, new_rotation, new_xyz_disp,
                new_xyz_motion, new_features_dc_motion, new_features_rest_motion, new_scaling_motion, new_opacity_motion, new_opacity_duration_center, new_opacity_duration_var, new_rotation_motion,
                new_gs_mask, new_sh_mask, new_gs_mask_motion, new_sh_mask_motion
                )
        else:
            self.densification_postfix_onlystatic(
                new_xyz, new_features_dc, new_features_rest, new_opacity, new_scaling, new_rotation, new_xyz_disp, new_gs_mask, new_sh_mask
                )
            dynamic_prune_filter = torch.empty(0).cuda()

        self.prune_points(static_prune_filter, dynamic_prune_filter)

    def densify_and_clone(self, static_grads, dynamic_grads, dynamic_ssim_errors,
                          grad_threshold, dynamic_grad_threshold, scene_extent):
        self._check_mask_shape()
        # Extract points that satisfy the gradient condition
        selected_static_pts_mask = torch.where(torch.norm(static_grads, dim=-1) >= grad_threshold, True, False)
        selected_static_pts_mask = torch.logical_and(selected_static_pts_mask,
                                              torch.max(self.get_static_scaling, dim=1).values <= self.percent_dense*scene_extent)

        new_xyz = self._xyz[selected_static_pts_mask]
        new_features_dc = self._features_dc[selected_static_pts_mask]
        new_features_rest = self._features_rest[selected_static_pts_mask]
        new_opacities = self._opacity[selected_static_pts_mask]
        new_scaling = self._scaling[selected_static_pts_mask]
        new_rotation = self._rotation[selected_static_pts_mask]
        new_xyz_disp = self._xyz_disp[selected_static_pts_mask]

        new_xyz_error_min = self.xyz_error_min[selected_static_pts_mask]
        new_xyz_error_min_timestamp = self.xyz_error_min_timestamp[selected_static_pts_mask]
        self.xyz_error_min = torch.cat([self.xyz_error_min, new_xyz_error_min])
        self.xyz_error_min_timestamp = torch.cat([self.xyz_error_min_timestamp, new_xyz_error_min_timestamp])

        new_gs_mask = self.gs_mask[selected_static_pts_mask]
        new_sh_mask = self.sh_mask[selected_static_pts_mask]

        if self._xyz_motion.shape[0] == 0:
            self.densification_postfix_onlystatic(
                new_xyz, new_features_dc, new_features_rest, new_opacities, new_scaling, new_rotation, new_xyz_disp, new_gs_mask, new_sh_mask
                )
            return
        # dynamic points
        selected_dynamic_pts_mask = torch.where(torch.norm(dynamic_grads, dim=-1) >= dynamic_grad_threshold, True, False)
        selected_dynamic_pts_mask = torch.logical_and(selected_dynamic_pts_mask,
                                              torch.max(self.get_motion_scaling, dim=1).values <= self.percent_dense*scene_extent)

        new_xyz_motion = self._xyz_motion[selected_dynamic_pts_mask]
        new_features_dc_motion = self._features_dc_motion[selected_dynamic_pts_mask]
        new_features_rest_motion = self._features_rest_motion[selected_dynamic_pts_mask]
        new_scaling_motion = self._scaling_motion[selected_dynamic_pts_mask]
        new_rotation_motion = self._rotation_motion[selected_dynamic_pts_mask]
        new_opacity_motion = self._opacity_motion[selected_dynamic_pts_mask]
        new_opacity_duration_center = self._opacity_duration_center[selected_dynamic_pts_mask]
        new_opacity_duration_center_len = ((new_opacity_duration_center[:, 1] - new_opacity_duration_center[:, 0]).abs() / 3).clamp_min(2/self.interval)
        new_opacity_duration_center[:, 1] = new_opacity_duration_center[:, 1] + (new_opacity_duration_center_len * (torch.randn_like(new_opacity_duration_center[:, 1])))
        new_opacity_duration_center[:, 0] = new_opacity_duration_center[:, 0] + (new_opacity_duration_center_len * (torch.randn_like(new_opacity_duration_center[:, 0])))
        new_opacity_duration_center = new_opacity_duration_center.clamp((self.time_shift + 1)/self.interval, (self.time_shift+self.duration - 1)/self.interval)
        new_opacity_duration_var = torch.ones_like(self._opacity_duration_var[selected_dynamic_pts_mask]) * 2

        new_motion_xyz_error_min = self.motion_xyz_error_min[selected_dynamic_pts_mask]
        new_motion_xyz_error_min_timestamp = self.motion_xyz_error_min_timestamp[selected_dynamic_pts_mask]
        self.motion_xyz_error_min = torch.cat([self.motion_xyz_error_min, new_motion_xyz_error_min])
        self.motion_xyz_error_min_timestamp = torch.cat([self.motion_xyz_error_min_timestamp, new_motion_xyz_error_min_timestamp])

        new_gs_mask_motion = self.gs_mask_motion[selected_dynamic_pts_mask]
        new_sh_mask_motion = self.sh_mask_motion[selected_dynamic_pts_mask]

        self.densification_postfix(
            new_xyz, new_features_dc, new_features_rest, new_opacities, new_scaling, new_rotation, new_xyz_disp,
            new_xyz_motion, new_features_dc_motion, new_features_rest_motion, new_scaling_motion, new_opacity_motion, new_opacity_duration_center, new_opacity_duration_var, new_rotation_motion,
            new_gs_mask, new_sh_mask, new_gs_mask_motion, new_sh_mask_motion
            )

    def densify_and_prune(self, max_grad, max_dgrad, min_opacity, min_motion_opacity, extent,
                          max_screen_size, max_dynamic_screen_size, duration_thres=-5.0,
                          s_max_ssim=0.5, s_l1_thres=0.1, d_max_ssim=0.5, d_l1_thres=0.1):
        self._check_mask_shape()
        static_grads = self.xyz_gradient_accum / self.denom
        static_grads[static_grads.isnan()] = 0.0

        dynamic_grads = self.motion_xyz_gradient_accum / self.motion_denom
        dynamic_grads[dynamic_grads.isnan()] = 0.0

        dynamic_ssim_errors = self.motion_xyz_ssim_error_accum / (self.motion_error_denom).clamp_min(1e-4)
        dynamic_ssim_errors[dynamic_ssim_errors.isnan()] = 0.0

        self.densify_and_clone(static_grads, dynamic_grads, dynamic_ssim_errors, max_grad, max_dgrad, extent)
        self.densify_and_split(static_grads, dynamic_grads, dynamic_ssim_errors, max_grad, max_dgrad, extent, max_screen_size, max_dynamic_screen_size)

        static_prune_mask = (self.get_opacity < min_opacity).squeeze()
        if max_screen_size:
            big_points_vs = self.max_radii2D > max_screen_size
            big_points_ws = self.get_static_scaling.max(dim=1).values > 0.1 * extent
            static_prune_mask = torch.logical_or(torch.logical_or(static_prune_mask, big_points_vs), big_points_ws)

        static_l1_mask = self.xyz_error_accum / self.error_denom.clamp(1e-4)
        static_l1_mask = static_l1_mask > s_l1_thres
        static_prune_mask = torch.logical_or(static_prune_mask, static_l1_mask.squeeze())

        static_ssim_mask = self.xyz_ssim_error_accum / self.error_denom.clamp(1e-4)
        static_ssim_mask = (static_ssim_mask < s_max_ssim) * (static_ssim_mask > 0)
        static_prune_mask = torch.logical_or(static_prune_mask, static_ssim_mask.squeeze())

        if self._xyz_motion.shape[0] == 0:
            dynamic_prune_mask = torch.empty(0).cuda()
            self.prune_points(static_prune_mask, dynamic_prune_mask)
            torch.cuda.empty_cache()
            return

        # dynamic points
        dynamic_prune_mask = (self.get_motion_opacity < min_motion_opacity).squeeze()

        dynamic_l1_mask = self.motion_xyz_error_mean / self.motion_error_denom.clamp(1e-4)
        dynamic_l1_mask = dynamic_l1_mask > d_l1_thres
        dynamic_prune_mask = torch.logical_or(dynamic_prune_mask, dynamic_l1_mask.squeeze())

        dynamic_ssim_mask = self.motion_xyz_ssim_error_accum / self.motion_error_denom.clamp(1e-4)
        dynamic_ssim_mask = (dynamic_ssim_mask < d_max_ssim) * (dynamic_ssim_mask > 0)
        dynamic_prune_mask = torch.logical_or(dynamic_prune_mask, dynamic_ssim_mask.squeeze())

        if max_dynamic_screen_size:
            big_points_vs = self.motion_max_radii2D > max_dynamic_screen_size
            big_points_ws = self.get_motion_scaling.max(dim=1).values > 0.1 * extent
            dynamic_prune_mask = torch.logical_or(torch.logical_or(dynamic_prune_mask, big_points_vs), big_points_ws)

        self.prune_points(static_prune_mask, dynamic_prune_mask)

        torch.cuda.empty_cache()

    def prune_invisible(self):
        self._check_mask_shape()
        static_prune_mask = (self.xyz_error_min_timestamp < 0).squeeze()
        dynamic_prune_mask = (self.motion_xyz_error_min_timestamp < 0).squeeze()

        torch.logical_or(
            dynamic_prune_mask,
            ((self._opacity_duration_center[:, 1] - self._opacity_duration_center[:, 0]).abs() < 0.5 / self.interval).squeeze()
        )

        self.prune_points(static_prune_mask, dynamic_prune_mask)

        torch.cuda.empty_cache()

        if static_prune_mask.sum() + dynamic_prune_mask.sum() > 0:
            return (static_prune_mask, dynamic_prune_mask)
        else:
            return None

    def prune_small(self):
        self._check_mask_shape()
        static_prune_mask = (self.min_radii2D < 5).squeeze()
        dynamic_prune_mask = (self.motion_min_radii2D < 5).squeeze()

        self.prune_points(static_prune_mask, dynamic_prune_mask)

        torch.cuda.empty_cache()

        if static_prune_mask.sum() + dynamic_prune_mask.sum() > 0:
            return (static_prune_mask, dynamic_prune_mask)
        else:
            return None

    def add_densification_stats(self, viewspace_point_tensor, static_update_filter, dynamic_update_filter, static_num):
        self.xyz_gradient_accum[static_update_filter] += torch.norm(viewspace_point_tensor.grad[:static_num][static_update_filter,:2], dim=-1, keepdim=True)
        self.denom[static_update_filter] += 1

        if self.motion_xyz_gradient_accum.shape[0] == 0:
            return

        self.motion_xyz_gradient_accum[dynamic_update_filter] += torch.norm(viewspace_point_tensor.grad[static_num:][dynamic_update_filter,:2], dim=-1, keepdim=True)
        self.motion_denom[dynamic_update_filter] += 1

    def mark_prune_stats(self, radii, viewspace_point_error_tensor):
        static_num = self._xyz.shape[0]

        static_radii = radii[:static_num]
        static_vis_filter = viewspace_point_error_tensor.grad[:static_num, 0] > 0
        self.min_radii2D[static_vis_filter] = torch.min(self.min_radii2D[static_vis_filter], static_radii[static_vis_filter])

        if self.motion_xyz_gradient_accum.shape[0] == 0:
            return

        dynamic_radii = radii[static_num:]
        dynamic_vis_filter = viewspace_point_error_tensor.grad[static_num:, 0] > 0
        self.motion_min_radii2D[dynamic_vis_filter] = torch.min(self.motion_min_radii2D[dynamic_vis_filter], dynamic_radii[dynamic_vis_filter])

    def add_l1_ssim_stats(self, viewspace_point_error_tensor, static_update_filter, dynamic_update_filter, static_num, timestamp):
        static_errors = viewspace_point_error_tensor.grad[:static_num]
        static_l1_error = static_errors[static_update_filter, 1:2] / static_errors[static_update_filter, 0:1].clamp_min(1e-4)
        self.xyz_error_accum[static_update_filter] += static_l1_error
        self.xyz_error_min_timestamp[static_update_filter] = torch.where(torch.logical_and(self.xyz_error_min[static_update_filter] > static_l1_error, static_errors[static_update_filter, 0:1] > 0.01),
                                                                                 timestamp * torch.ones_like(static_l1_error),
                                                                                 self.xyz_error_min_timestamp[static_update_filter])
        self.xyz_error_min[static_update_filter] = torch.where(torch.logical_and(self.xyz_error_min[static_update_filter] > static_l1_error, static_errors[static_update_filter, 0:1] > 0.01),
                                                                       static_l1_error, self.xyz_error_min[static_update_filter])
        self.xyz_ssim_error_accum[static_update_filter] += static_errors[static_update_filter, 2:3] / static_errors[static_update_filter, 0:1].clamp_min(1e-4)
        self.error_denom[static_update_filter] += (static_errors[static_update_filter, 0:1] > 0).float()

        if self.motion_xyz_error_min.shape[0] == 0:
            return

        dynamic_errors = viewspace_point_error_tensor.grad[static_num:]

        dynamic_l1_error = dynamic_errors[dynamic_update_filter, 1:2] / dynamic_errors[dynamic_update_filter, 0:1].clamp_min(1e-4)
        self.motion_xyz_error_min_timestamp[dynamic_update_filter] = torch.where(torch.logical_and(self.motion_xyz_error_min[dynamic_update_filter] > dynamic_l1_error, dynamic_errors[dynamic_update_filter, 0:1] > 0.01),
                                                                                 timestamp * torch.ones_like(dynamic_l1_error),
                                                                                 self.motion_xyz_error_min_timestamp[dynamic_update_filter])
        self.motion_xyz_error_min[dynamic_update_filter] = torch.where(torch.logical_and(self.motion_xyz_error_min[dynamic_update_filter] > dynamic_l1_error, dynamic_errors[dynamic_update_filter, 0:1] > 0.01),
                                                                       dynamic_l1_error, self.motion_xyz_error_min[dynamic_update_filter])

        self.motion_xyz_error_mean[dynamic_update_filter] += dynamic_l1_error
        self.motion_xyz_ssim_error_accum[dynamic_update_filter] += dynamic_errors[dynamic_update_filter, 2:3] / dynamic_errors[dynamic_update_filter, 0:1].clamp_min(1e-4)
        self.motion_error_denom[dynamic_update_filter] += (dynamic_errors[dynamic_update_filter, 0:1] > 0).float()

    def extract_dynamic_points_from_static(self, viewpoint_loc, timestamp, vis_filter, extent, percentile=0.98, motion_thres=1000.0, min_motion_thres=1e-6, win_size=0, max_dur=None):
        self._check_mask_shape()

        if max_dur is None:
            max_dur = self.duration
        else:
            max_dur = max(float(max_dur), self.interval)

        disp = self._xyz_disp[vis_filter].norm(dim=-1)
        disp_denorm = (self._xyz[vis_filter] - viewpoint_loc.to(self._xyz.device)).norm(dim=-1) ** 2
        disp = disp / (disp_denorm+0.000001)
        disp = disp / (disp.max()+0.000001)
        mv_thresh = torch.quantile(disp, percentile)

        dynamic_mask = (disp > mv_thresh) | (self._xyz_disp.norm(dim=-1)[vis_filter] > motion_thres * extent)
        dynamic_mask = dynamic_mask & (self._xyz_disp.norm(dim=-1)[vis_filter] > min_motion_thres * extent)
        static_prune_mask = vis_filter.clone()
        static_prune_mask[vis_filter] = dynamic_mask
        static_prune_mask = torch.logical_and(static_prune_mask, self.xyz_error_min_timestamp.squeeze() >= 0)

        if self.keyframe_num == 0:
            self.keyframe_num = math.ceil((max_dur + self.time_shift*2 + 1) / self.interval) + 1 + 2
        new_xyz_motion = nn.functional.interpolate(torch.stack([
                self._xyz[static_prune_mask] - self._xyz_disp[static_prune_mask] * self.interval / max_dur,
                self._xyz[static_prune_mask] + self._xyz_disp[static_prune_mask] * (1 + self.interval / max_dur)],dim=2).unsqueeze(-1),
                size=(self.keyframe_num, 1), mode='bilinear').squeeze(-1).transpose(1, 2)
        if self.motion_degree > 1:
            new_xyz_motion = torch.cat([new_xyz_motion, torch.zeros_like(new_xyz_motion).repeat(1, 1, self.motion_degree-1)], dim=-1)

        new_features_dc_motion = self._features_dc[static_prune_mask]
        new_features_rest_motion = self._features_rest[static_prune_mask]
        new_scaling_motion = self._scaling[static_prune_mask]
        new_opacity_motion = self._opacity[static_prune_mask]
        new_gs_mask_motion = self.gs_mask[static_prune_mask]
        new_sh_mask_motion = self.sh_mask[static_prune_mask]
        t = self.xyz_error_min_timestamp[static_prune_mask]
        min_time = 0
        new_opacity_duration_center = torch.stack([
            torch.ones_like(new_opacity_motion) * (t * 1 / 2 + self.time_shift ) / self.interval,
            torch.ones_like(new_opacity_motion) * ( (max_dur + t.clamp_min(min_time) * 1) / 2 + self.time_shift) / self.interval,
            ], dim=1).clamp((min_time+self.time_shift+1)/self.interval, (self.time_shift+max_dur-1)/self.interval)
        new_opacity_duration_var = torch.stack([
            torch.ones_like(new_opacity_motion) * (t + self.time_pad) ,
            torch.ones_like(new_opacity_motion) * (max_dur - t + self.time_pad) ,
            ], dim=1)

        new_rotation_motion = self._rotation[static_prune_mask].unsqueeze(1).repeat(1, self.keyframe_num, 1)

        d = {
        "motion_xyz" : new_xyz_motion,
        "motion_f_dc" : new_features_dc_motion,
        "motion_f_rest" : new_features_rest_motion,
        "motion_scaling" : new_scaling_motion,
        "motion_opacity" : new_opacity_motion,
        "motion_opacity_center" : new_opacity_duration_center,
        "motion_opacity_var" : new_opacity_duration_var,
        "motion_rotation" : new_rotation_motion,
        "gs_mask_motion" : new_gs_mask_motion,
        "sh_mask_motion" : new_sh_mask_motion
        }

        optimizable_tensors = self.cat_tensors_to_optimizer(d)
        # dynamic points
        self._xyz_motion = optimizable_tensors["motion_xyz"]
        self._features_dc_motion = optimizable_tensors["motion_f_dc"]
        self._features_rest_motion = optimizable_tensors["motion_f_rest"]
        self._scaling_motion = optimizable_tensors["motion_scaling"]
        self._opacity_motion = optimizable_tensors["motion_opacity"]
        self._opacity_duration_center = optimizable_tensors["motion_opacity_center"]
        self._opacity_duration_var = optimizable_tensors["motion_opacity_var"]
        self._rotation_motion = optimizable_tensors["motion_rotation"]
        self.gs_mask_motion = optimizable_tensors["gs_mask_motion"]
        self.sh_mask_motion = optimizable_tensors["sh_mask_motion"]

        # reset grad anyway
        self.motion_xyz_gradient_accum = torch.zeros((self._xyz_motion.size(0), 1), device="cuda")
        self.motion_denom = torch.zeros((self._xyz_motion.size(0), 1), device="cuda")
        self.motion_xyz_error_mean = torch.zeros((self._xyz_motion.size(0), 1), device="cuda")
        self.motion_xyz_ssim_error_accum = torch.zeros((self._xyz_motion.size(0), 1), device="cuda")
        self.motion_error_denom = torch.zeros((self._xyz_motion.size(0), 1), device="cuda")
        self.motion_max_radii2D = torch.zeros((self._xyz_motion.size(0)), device="cuda")
        self.motion_min_radii2D = torch.ones((self._xyz_motion.size(0)), device="cuda") * 1000

        new_motion_xyz_error_min = torch.ones((static_prune_mask.sum(), 1), device="cuda") * 1000
        new_motion_xyz_error_min_timestamp = torch.ones((static_prune_mask.sum(), 1), device="cuda") * -1
        self.motion_xyz_error_min = torch.cat([self.motion_xyz_error_min, new_motion_xyz_error_min])
        self.motion_xyz_error_min_timestamp = torch.cat([self.motion_xyz_error_min_timestamp, new_motion_xyz_error_min_timestamp])

        self.prune_points(static_prune_mask, torch.zeros(self._xyz_motion.size(0), device=self._xyz_motion.device, dtype=torch.bool))

        if static_prune_mask.sum() > 0:
            return static_prune_mask
        else:
            return None

    def handle_nan_values(self):
        """Replaces NaNs in non-critical parameters and prunes points if critical parameters have NaNs."""
        self._check_mask_shape()

        num_static_before = self._xyz.shape[0]
        num_dynamic_before = self._xyz_motion.shape[0] if self._xyz_motion is not None else 0

        # 1. Replace NaNs in non-critical parameters with 0.0 (in-place)
        # --- Static parameters ---
        if num_static_before > 0:
            torch.nan_to_num_(self._opacity.data, nan=0.0)
            torch.nan_to_num_(self._features_dc.data, nan=0.0)
            torch.nan_to_num_(self._features_rest.data, nan=0.0)
            torch.nan_to_num_(self.gs_mask.data, nan=0.0)
            torch.nan_to_num_(self.sh_mask.data, nan=0.0)
            torch.nan_to_num_(self._xyz_disp.data, nan=0.0) # Assuming disp NaN is replaceable

        # --- Dynamic parameters ---
        if num_dynamic_before > 0:
            torch.nan_to_num_(self._opacity_motion.data, nan=0.0)
            torch.nan_to_num_(self._opacity_duration_var.data, nan=0.0)
            torch.nan_to_num_(self._opacity_duration_center.data, nan=0.0)
            torch.nan_to_num_(self._features_dc_motion.data, nan=0.0)
            torch.nan_to_num_(self._features_rest_motion.data, nan=0.0)
            torch.nan_to_num_(self.gs_mask_motion.data, nan=0.0)
            torch.nan_to_num_(self.sh_mask_motion.data, nan=0.0)

        # 2. Identify points to prune based on NaNs in *critical* parameters
        # --- Static points ---
        if num_static_before > 0:
            static_xyz_nan = torch.isnan(self._xyz).any(dim=-1)
            static_scale_nan = torch.isnan(self._scaling).any(dim=-1)
            static_rot_nan = torch.isnan(self._rotation).any(dim=-1)
            static_prune_mask = static_xyz_nan | static_scale_nan | static_rot_nan
        else:
            static_prune_mask = torch.zeros(0, device=self._xyz.device, dtype=torch.bool)
        # --- Dynamic points ---
        if num_dynamic_before > 0:
            dynamic_xyz_nan = torch.isnan(self._xyz_motion).flatten(start_dim=1).any(dim=-1)
            dynamic_scale_nan = torch.isnan(self._scaling_motion).any(dim=-1)
            dynamic_rot_nan = torch.isnan(self._rotation_motion).flatten(start_dim=1).any(dim=-1)
            dynamic_prune_mask = dynamic_xyz_nan | dynamic_scale_nan | dynamic_rot_nan
        else:
            dynamic_prune_mask = torch.zeros(0, device=self._xyz_motion.device if self._xyz_motion is not None else "cuda", dtype=torch.bool) # Handle potential empty tensor case device

        num_static_to_prune = static_prune_mask.sum().item()
        num_dynamic_to_prune = dynamic_prune_mask.sum().item()

        # 3. Perform pruning only if critical NaNs were found
        if num_static_to_prune > 0 or num_dynamic_to_prune > 0:
            print(f"Pruning points due to critical NaNs: {num_static_to_prune} static, {num_dynamic_to_prune} dynamic")
            self.prune_points(static_prune_mask, dynamic_prune_mask)
            # Return the masks indicating which points were pruned
            return (static_prune_mask, dynamic_prune_mask)
        else:
            # Return None or (False, False) if no points were pruned
            return None

    def expand_duration(self, duration):
        duration = int(duration)+1
        if duration <= self.duration:
            return False

        # no dynamic points
        if self._xyz_motion.shape[0] == 0:
            # print("No dynamic points to expand. static num: {} dynamic num: {}".format(self._xyz.shape[0], self._xyz_motion.shape[0]))
            self.duration = duration
            return False

        require_dim = math.ceil((duration + self.time_shift + self.time_pad*2 + 1) / self.interval) + 1 + 2
        cur_dim = self._xyz_motion.shape[1]
        num_expand = require_dim - cur_dim

        if num_expand < 1:
            # print("Not expanded: {}".format(duration))
            self.duration = duration
            return False

        # linear linterpolation of lastest frame
        def lin_interp_last(x, n, zero_init=False, average=1):
            diff = (x[:, -average:] - x[:, -average-1:-average]).mean(dim=1, keepdim=True) * 1.0
            new_frames = torch.arange(1, n+1, device="cuda").view(1, -1, * [1] * len(diff.shape[2:])) * diff + x[:, -1:]
            if self.motion_degree > 1 and zero_init:
                new_frames[..., 3:] = 0.
            return torch.cat([x, new_frames], dim=1)

        num_avg = min(self.keyframe_num-2, 4)
        new_xyz_motion = lin_interp_last(self._xyz_motion, num_expand, zero_init=True, average=num_avg)
        new_rotation_motion = lin_interp_last(self._rotation_motion, num_expand, average=num_avg)

        new_opacity_duration_var = self._opacity_duration_var.detach().clone()
        new_opacity_duration_var[:, 1] = torch.where((self._opacity_duration_center + self.time_shift / self.interval > (duration + self.time_shift) / self.interval - 0.5).any(dim=1),
                                                   torch.ones_like(self._opacity_duration_var[:, 1]),
                                                   self._opacity_duration_var[:, 1])
        new_opacity_duration_center = self._opacity_duration_center.clamp_max((self.time_shift+self.duration - 1)/self.interval)

        d = {
        "motion_xyz" : new_xyz_motion,
        "motion_opacity_center" : new_opacity_duration_center,
        "motion_opacity_var" : new_opacity_duration_var,
        "motion_rotation" : new_rotation_motion,
        }

        optimizable_tensors = self.replace_tensor_to_optimizer(d)

        self._xyz_motion = optimizable_tensors["motion_xyz"]
        self._opacity_duration_center = optimizable_tensors["motion_opacity_center"]
        self._opacity_duration_var = optimizable_tensors["motion_opacity_var"]
        self._rotation_motion = optimizable_tensors["motion_rotation"]
        self.keyframe_num = require_dim
        self.duration = duration

        return True

    def mark_error(self, loss, timestamp):
        t_idx = timestamp // self.interval

        if t_idx in self.error_dict.keys():
            self.error_dict[t_idx] = (self.error_dict[t_idx][0] + loss, self.error_dict[t_idx][1] + 1)
        else:
            self.error_dict[t_idx] = (loss, 1)

    def get_errorneous_timestamp(self):
        if len(self.error_dict) == 0:
            return None

        max_loss = 0
        max_idx = 0
        max_count = 0

        for t_idx in self.error_dict.keys():
            loss, count = self.error_dict[t_idx]
            max_count = max(max_count, count)

            if loss / count > max_loss and count > max_count * 0.1:
                max_loss = loss / count
                max_idx = t_idx

        if max_loss == 0:
            return None

        del self.error_dict[max_idx]

        return (max_idx + 0.5) * self.interval

    def adjust_temp_opa(self, max_dur=None):
        if max_dur is None:
            max_dur = self.duration
        else:
            max_dur = float(max_dur)
        if self._xyz_motion.shape[0] == 0:
            return
        new_opacity_duration_var = self._opacity_duration_var.detach().clone()
        new_opacity_duration_var[:, 1] = torch.where((self._opacity_duration_center > (max_dur + self.time_shift) / self.interval - 0.2).any(dim=1),
                                                   self._opacity_duration_var[:, 1].clamp_min(1)*2,
                                                   self._opacity_duration_var[:, 1])
        new_opacity_duration_var[:, 0] = torch.where((self._opacity_duration_center < (self.time_shift) / self.interval + 0.2).any(dim=1),
                                                   self._opacity_duration_var[:, 0].clamp_min(1)*2,
                                                   self._opacity_duration_var[:, 0])
        new_opacity_duration_center = self._opacity_duration_center.clamp((self.time_shift) / self.interval + 0.2, (max_dur + self.time_shift) / self.interval - 0.2)

        new_opacity_duration_var = torch.where(self._opacity_duration_var < 0.5,
                                                   torch.ones_like(self._opacity_duration_var)*0.5,
                                                   new_opacity_duration_var)

        d = {
        "motion_opacity_center" : new_opacity_duration_center,
        "motion_opacity_var" : new_opacity_duration_var,
        }

        optimizable_tensors = self.replace_tensor_to_optimizer(d)

        self._opacity_duration_center = optimizable_tensors["motion_opacity_center"]
        self._opacity_duration_var = optimizable_tensors["motion_opacity_var"]

    # RDO compression and save/load methods
    def apply_and_save_vq(self, vq_path):
        """Apply vector quantization and save compressed representation"""
        mkdir_p(vq_path)

        gs_bitmask = self.apply_gs_mask_at_t(t=0)['gs_bitmask']
        # Split bitmask for static and dynamic points

        if gs_bitmask.shape[0] != self._xyz.shape[0] + self._xyz_motion.shape[0]:
            raise ValueError(f"gs_bitmask shape {gs_bitmask.shape[0]} does not match the total number of points {self._xyz.shape[0] + self._xyz_motion.shape[0]}")

        static_num = self._xyz.shape[0]

        static_bitmask = gs_bitmask[:static_num]
        dynamic_bitmask = gs_bitmask[static_num:]
        self.prune_points(~static_bitmask, ~dynamic_bitmask)
        print(f"Applied Gaussian pruning mask, removed {(~static_bitmask).sum()} static and {(~dynamic_bitmask).sum()} dynamic points")

        static_num = self._xyz.shape[0]
        dynamic_num = self._xyz_motion.shape[0]
        num_gaussians = static_num + dynamic_num

        # 2. Apply SH Pruning Mask
        sh_bitmask = {}
        sh_prune_result_dict = self.apply_sh_mask()
        for sh_key in ['sh1', 'sh2', 'sh3']:
            sh_bitmask[sh_key] = sh_prune_result_dict[f'{sh_key}_bitmask'].detach().contiguous().cpu().numpy()
            if sh_bitmask[sh_key].sum() <= 1:
                # make one of the values 1
                sh_bitmask[sh_key][0] = 1
                sh_bitmask[sh_key][1] = 1

        # 3. Apply Vector Quantization
        vq_out = {}
        vq_indexes = {}
        vq_codebooks = {}
        vq_logits = {}
        remove_negatives(self._quantizer['scale'].codebook)

        # Apply VQ to each attribute
        for vq_key in self.vq_cfg['keys']:
            if vq_key in ['sh1', 'sh2', 'sh3']:
                vq_result = self._quantizer[vq_key](self.vector_dict[vq_key][sh_bitmask[vq_key]])
            elif vq_key == 'rot':
                vq_result_static = self._quantizer[vq_key](self.vector_dict[vq_key][:static_num, 0, :])
                vq_result_dynamic = self._quantizer[vq_key](self.vector_dict[vq_key][static_num:, :, :].reshape(-1, 4))
                vq_result = {
                    "x_hat": torch.cat([vq_result_static['x_hat'], vq_result_dynamic['x_hat']], dim=0),
                    "x_index": torch.cat([vq_result_static['x_index'], vq_result_dynamic['x_index']], dim=0)
                }
            else:
                vq_result = self._quantizer[vq_key](self.vector_dict[vq_key])
            vq_xhat = self.vq_post_process(vq_key, vq_result["x_hat"]) if vq_key != 'rot' else vq_result["x_hat"]
            vq_out[vq_key] = vq_xhat
            vq_indexes[vq_key] = vq_result['x_index'].detach().contiguous().cpu().numpy()
            vq_codebooks[vq_key] = self._quantizer[vq_key].codebook.detach().contiguous().cpu().numpy()
            vq_logits[vq_key] = self._quantizer[vq_key].logits.detach().contiguous().cpu().numpy()

        features_rest = torch.zeros_like(torch.cat([self._features_rest.data, self._features_rest_motion.data], dim=0))
        features_rest[sh_bitmask['sh1'],:3,:] = vq_out['sh1']
        features_rest[sh_bitmask['sh2'],3:8,:] = vq_out['sh2']
        features_rest[sh_bitmask['sh3'],8:,:] = vq_out['sh3']

        self._features_rest.data = features_rest[:static_num]
        self._features_rest_motion.data = features_rest[static_num:]

        self._features_dc.data = vq_out['dc'][:static_num]
        self._features_dc_motion.data = vq_out['dc'][static_num:]

        self._scaling.data = self.scaling_inverse_activation(vq_out['scale'][:static_num])
        self._scaling_motion.data = self.scaling_inverse_activation(vq_out['scale'][static_num:])

        self._rotation.data = vq_out['rot'][:static_num]
        self._rotation_motion.data = vq_out['rot'][static_num:].reshape(dynamic_num, self.keyframe_num, 4)

        # Rearrange Gaussians by SH masks
        sort_idx, boundaries = shmask_sort(sh_bitmask)
        sh = -np.ones((num_gaussians, 3), dtype=int)
        sh[sh_bitmask['sh1'], 0] = vq_indexes['sh1'].squeeze()
        sh[sh_bitmask['sh2'], 1] = vq_indexes['sh2'].squeeze()
        sh[sh_bitmask['sh3'], 2] = vq_indexes['sh3'].squeeze()

        sh = sh[sort_idx]

        vq_indexes['sh1'] = sh[boundaries[3]:,0:1]
        vq_indexes['sh2'] = np.concatenate([sh[boundaries[1]:boundaries[3], 1:2],
                                            sh[boundaries[5]:, 1:2]])
        vq_indexes['sh3'] = np.concatenate([sh[boundaries[0]:boundaries[1], 2:],
                                            sh[boundaries[2]:boundaries[3], 2:],
                                            sh[boundaries[4]:boundaries[5], 2:],
                                            sh[boundaries[6]:, 2:]])
        for key in ['scale', 'rot', 'dc']:
            vq_indexes[key] = vq_indexes[key]
        vq_codebooks, vq_logits, vq_indexes = shrink_codebook(vq_codebooks, vq_logits, vq_indexes)

        # Non-VQ attributes
        xyz = self._xyz.detach().cpu().numpy().astype(np.float16)
        xyz_disp = self._xyz_disp.detach().cpu().numpy().astype(np.float16)

        print(f"sh_bitmask shape: {sh_bitmask['sh1'].shape}, {sh_bitmask['sh2'].shape}, {sh_bitmask['sh3'].shape}")
        print(f'xyz shape: {xyz.shape}, xyz_disp shape: {xyz_disp.shape}')

        # wavelet mask
        is_even = self.keyframe_num % 2 == 0
        half_keyframe_num = self.keyframe_num // 2 if is_even else (self.keyframe_num + 1) // 2
        xyz_motion = self._xyz_motion.detach()
        print(f"xyz_motion shape: {xyz_motion.shape}")
        wavelet_matrix = generate_haar_matrix(self._xyz_motion.shape[1]).to('cuda')
        if not is_even:
            xyz_motion = torch.cat([xyz_motion, xyz_motion[:, -1:, :]], dim=1)
        coeffs = points_to_wavelet_coeffs(xyz_motion, wavelet_matrix)
        coeffs_approx = coeffs[:, :half_keyframe_num, :]
        coeffs_approx_np = coeffs_approx.detach().cpu().numpy().astype(np.float16)
        print(f"coeffs_approx shape: {coeffs_approx.shape}")

        opacity = self._opacity.detach().cpu().numpy()
        opacity_motion = self._opacity_motion.detach().cpu().numpy()
        opacity_center = self._opacity_duration_center.detach().cpu().numpy()
        opacity_var = self._opacity_duration_var.detach().cpu().numpy()

        # np.savez(os.path.join(vq_path, 'opacity_static.npz'), opacity=opacity)
        # np.savez(os.path.join(vq_path, 'opacity_motion.npz'), opacity=opacity_motion)
        # np.savez(os.path.join(vq_path, 'opacity_center.npz'), opacity=opacity_center)
        np.savez_compressed(os.path.join(vq_path, 'opacity_var.npz'), opacity=opacity_var)

        dict_opacity_static = {}
        (dict_opacity_static['quantized_opacities'],
         dict_opacity_static['opacities'],
         dict_opacity_static['opacity_log_prob'],
         dict_opacity_static['step_size'],
         dict_opacity_static['min_opacity']) = opacity_quant(opacity)

        dict_opacity_motion = {}
        (dict_opacity_motion['quantized_opacities'],
         dict_opacity_motion['opacities'],
         dict_opacity_motion['opacity_log_prob'],
         dict_opacity_motion['step_size'],
         dict_opacity_motion['min_opacity']) = opacity_quant(opacity_motion)

        dict_opacity_center = {}
        (dict_opacity_center['quantized_opacities'],
         dict_opacity_center['opacities'],
         dict_opacity_center['opacity_log_prob'],
         dict_opacity_center['step_size'],
         dict_opacity_center['min_opacity']) = opacity_quant(opacity_center)

        # dict_opacity_var = {}
        # (dict_opacity_var['quantized_opacities'],
        #  dict_opacity_var['opacities'],
        #  dict_opacity_var['opacity_log_prob'],
        #  dict_opacity_var['step_size'],
        #  dict_opacity_var['min_opacity']) = opacity_quant(opacity_var)

        vq_logits['opa'] = dict_opacity_static['opacity_log_prob']
        vq_logits['opa_motion'] = dict_opacity_motion['opacity_log_prob']
        vq_logits['opa_center'] = dict_opacity_center['opacity_log_prob']
        # vq_logits['opa_var'] = dict_opacity_var['opacity_log_prob']

        # Arithmetic coding
        index_strings = []
        index_lengths = []

        for vq_key in vq_indexes.keys():
            bits_str = entropy_coding(torch.Tensor(vq_indexes[vq_key]), torch.Tensor(vq_logits[vq_key]))
            index_strings.append(bits_str)
            index_lengths.append(len(bits_str))

        bits_str = entropy_coding(torch.Tensor(dict_opacity_static['quantized_opacities']), torch.Tensor(dict_opacity_static['opacity_log_prob']))
        index_strings.append(bits_str)
        index_lengths.append(len(bits_str))

        bits_str = entropy_coding(torch.Tensor(dict_opacity_motion['quantized_opacities']), torch.Tensor(dict_opacity_motion['opacity_log_prob']))
        index_strings.append(bits_str)
        index_lengths.append(len(bits_str))

        bits_str = entropy_coding(torch.Tensor(dict_opacity_center['quantized_opacities']).view(-1, 1), torch.Tensor(dict_opacity_center['opacity_log_prob']))
        index_strings.append(bits_str)
        index_lengths.append(len(bits_str))

        # bits_str = entropy_coding(torch.Tensor(dict_opacity_var['quantized_opacities']).view(-1, 1), torch.Tensor(dict_opacity_var['opacity_log_prob']))
        # index_strings.append(bits_str)
        # index_lengths.append(len(bits_str))

        index_bitstream = pack_strings(index_strings)
        boundaries_bitstream = pack_uints(boundaries.tolist())

        opacity_header_bitstream_static = pack_floats([dict_opacity_static['step_size'], dict_opacity_static['min_opacity']])
        opacity_header_bitstream_motion = pack_floats([dict_opacity_motion['step_size'], dict_opacity_motion['min_opacity']])
        opacity_header_bitstream_center = pack_floats([dict_opacity_center['step_size'], dict_opacity_center['min_opacity']])
        # opacity_header_bitstream_var = pack_floats([dict_opacity_var['step_size'], dict_opacity_var['min_opacity']])

        with open(os.path.join(vq_path, 'index_bitstream.bin'), "wb") as f:
            f.write(index_bitstream)
        with open(os.path.join(vq_path, 'header.bin'), "wb") as f:
            f.write(boundaries_bitstream)
            f.write(opacity_header_bitstream_static)
            f.write(opacity_header_bitstream_motion)
            f.write(opacity_header_bitstream_center)
            # f.write(opacity_header_bitstream_var)

        np.savez_compressed(os.path.join(vq_path, 'sort_idx.npz'), sort_idx=sort_idx)
        np.savez_compressed(os.path.join(vq_path, 'codebook.npz'), **vq_codebooks)
        np.savez_compressed(os.path.join(vq_path, 'logits.npz'), **vq_logits)
        np.savez_compressed(os.path.join(vq_path, 'position_static.npz'), position=xyz)
        np.savez_compressed(os.path.join(vq_path, 'position_disp.npz'), position=xyz_disp)
        np.savez_compressed(os.path.join(vq_path, 'motion_wavelet_approx.npz'), position=coeffs_approx_np)
        with open(os.path.join(vq_path, 'is_even.bin'), "wb") as f:
            f.write(np.array([is_even], dtype=np.uint8).tobytes())

    def decode(self, bitstream_path):
        # Load static and dynamic position data
        xyz = np.load(os.path.join(bitstream_path, 'position_static.npz'))['position']
        xyz_disp = np.load(os.path.join(bitstream_path, 'position_disp.npz'))['position']
        sort_idx = np.load(os.path.join(bitstream_path, 'sort_idx.npz'))['sort_idx']

        is_even = np.frombuffer(open(os.path.join(bitstream_path, 'is_even.bin'), "rb").read(), dtype=np.uint8)[0]
        coeffs_approx = torch.from_numpy(np.load(os.path.join(bitstream_path, 'motion_wavelet_approx.npz'))['position']).to('cuda')
        coeffs_detail = torch.zeros_like(coeffs_approx)
        coeffs = torch.cat([coeffs_approx, coeffs_detail], dim=1)
        wavelet_matrix = generate_haar_matrix(coeffs.shape[1]).to('cuda')
        xyz_motion = wavelet_coeffs_to_points(coeffs, wavelet_matrix)
        if not is_even:
            xyz_motion = xyz_motion[:, :-1, :]

        static_num = xyz.shape[0]
        dynamic_num = xyz_motion.shape[0]
        self.keyframe_num = xyz_motion.shape[1]

        # Load codebooks and logits
        vq_codebooks = np.load(os.path.join(bitstream_path, 'codebook.npz'))
        logits = np.load(os.path.join(bitstream_path, 'logits.npz'))

        # Load header information (boundaries and opacity parameters dictionary)
        # decode_header now returns a dictionary for opacity parameters
        boundaries, opacity_params = decode_header(os.path.join(bitstream_path, 'header.bin'))

        # Calculate index shapes using static_num and dynamic_num
        # Pass static_num and dynamic_num to get_index_shapes
        shape_dict = get_index_shapes(boundaries, static_num, dynamic_num, self.keyframe_num)
        index_path = os.path.join(bitstream_path, 'index_bitstream.bin')
        # Decode indexes using the calculated shapes
        indexes = decode_indexes(index_path, logits, shape_dict)
        sh_bitmask = get_sh_bitmask(boundaries) # Assuming this remains compatible

        # Decode opacity values using parameters from the dictionary returned by decode_header
        opacities_static = opacity_dequant(indexes.pop('opa'), opacity_params['min_opacity_static'], opacity_params['step_size_static']).to('cuda')
        opacities_motion = opacity_dequant(indexes.pop('opa_motion'), opacity_params['min_opacity_motion'], opacity_params['step_size_motion']).to('cuda')
        opacity_center = opacity_dequant(indexes.pop('opa_center'), opacity_params['min_opacity_center'], opacity_params['step_size_center']).to('cuda').reshape(-1, 2, 1)
        opacity_var = opacity_dequant(indexes.pop('opa_var'), opacity_params['min_opacity_var'], opacity_params['step_size_var']).to('cuda').reshape(-1, 2, 1)

        # Decode VQ attributes
        vq_attributes = {}
        # static_num and dynamic_num already defined above

        for vq_key in vq_codebooks.files:
            codebook = vq_codebooks[vq_key]
            index = indexes[vq_key].squeeze(1)
            attr = codebook[0, index, :]
            attr = torch.tensor(attr, dtype=torch.float, device="cuda")
            if vq_key == 'rot':  # Handle rotation separately for static/dynamic
                # attr has shape [static_num + dynamic_num * keyframe_num, 4]
                static_rot = attr[:static_num] # Get static rotations [static_num, 4]

                # Ensure keyframe_num is set before reshaping dynamic rotations
                if self.keyframe_num == 0:
                     if dynamic_num > 0:
                         # Try to infer keyframe_num (risky, explicit save/load preferred)
                         total_dynamic_rot_elements = attr[static_num:].numel()
                         if total_dynamic_rot_elements == 0: # Should not happen if dynamic_num > 0
                              raise ValueError("Inconsistency: dynamic_num > 0 but no dynamic rotation elements.")
                         if total_dynamic_rot_elements % (dynamic_num * 4) != 0:
                              raise ValueError(f"Cannot infer keyframe_num: total dynamic rotation elements ({total_dynamic_rot_elements}) mismatch with dynamic_num ({dynamic_num}).")
                         inferred_kf_num = total_dynamic_rot_elements // (dynamic_num * 4)
                         print(f"Warning: keyframe_num not set, inferred as {inferred_kf_num}. It's recommended to save/load keyframe_num explicitly.")
                         self.keyframe_num = inferred_kf_num
                     elif dynamic_num == 0:
                          pass # keyframe_num doesn't matter if no dynamic points
                     else:
                          raise ValueError("Invalid dynamic_num.")

                if dynamic_num > 0:
                     # Reshape the remaining part [dynamic_num * keyframe_num, 4] -> [dynamic_num, keyframe_num, 4]
                     dynamic_rot = attr[static_num:].reshape(dynamic_num, self.keyframe_num, 4)
                else:
                     # Create an empty tensor if no dynamic points
                     dynamic_rot = torch.empty((0, self.keyframe_num if self.keyframe_num > 0 else 1, 4), device=attr.device, dtype=attr.dtype)

                vq_attributes[vq_key] = (static_rot, dynamic_rot)
            else:
                vq_attributes[vq_key] = self.vq_post_process(vq_key, attr) if vq_key not in ['scale'] else attr

        # Reconstruct feature tensors
        num_gaussians = static_num + dynamic_num # Use post-pruning numbers
        features_dc = vq_attributes['dc']
        features_extra = torch.zeros((num_gaussians, 15, 3), dtype=torch.float, device="cuda")

        # Apply SH masks - ensure sh_bitmask corresponds to the *sorted* and *pruned* gaussians
        # The sorting happens in apply_and_save_vq, decode needs to respect this order
        # Check if sh_bitmask keys exist before accessing (in case of no SH masking)
        if 'sh1' in vq_attributes and 'sh1' in sh_bitmask:
             features_extra[sh_bitmask['sh1'], :3, :] = vq_attributes['sh1']
        if 'sh2' in vq_attributes and 'sh2' in sh_bitmask:
             features_extra[sh_bitmask['sh2'], 3:8, :] = vq_attributes['sh2']
        if 'sh3' in vq_attributes and 'sh3' in sh_bitmask:
             features_extra[sh_bitmask['sh3'], 8:, :] = vq_attributes['sh3']

        features_extra_unsorted = torch.zeros_like(features_extra)
        features_extra_unsorted[sort_idx] = features_extra

        # Separate static and dynamic features based on post-pruning numbers
        features_dc_static = features_dc[:static_num]
        features_dc_motion = features_dc[static_num:]
        features_extra_static = features_extra_unsorted[:static_num]
        features_extra_motion = features_extra_unsorted[static_num:]

        # Set model parameters
        self._xyz = nn.Parameter(torch.tensor(xyz, dtype=torch.float, device="cuda"))
        self._xyz_disp = nn.Parameter(torch.tensor(xyz_disp, dtype=torch.float, device="cuda"))
        self._xyz_motion = nn.Parameter(xyz_motion)
        self._features_dc = nn.Parameter(features_dc_static)
        self._features_dc_motion = nn.Parameter(features_dc_motion)
        self._features_rest = nn.Parameter(features_extra_static)
        self._features_rest_motion = nn.Parameter(features_extra_motion)
        self._opacity = nn.Parameter(opacities_static)
        self._opacity_motion = nn.Parameter(opacities_motion)
        self._opacity_duration_center = nn.Parameter(opacity_center)
        self._opacity_duration_var = nn.Parameter(opacity_var)
        self._scaling = nn.Parameter(self.scaling_inverse_activation(vq_attributes['scale'][:static_num]))
        self._scaling_motion = nn.Parameter(self.scaling_inverse_activation(vq_attributes['scale'][static_num:]))
        self._rotation = nn.Parameter(vq_attributes['rot'][0]) # Static rotations
        self._rotation_motion = nn.Parameter(vq_attributes['rot'][1]) # Dynamic rotations (already reshaped)

        self.active_sh_degree = self.max_sh_degree

    def load_vq(self, vq_path):
        """Load compressed model data saved by apply_and_save_vq."""
        print(f"Loading compressed model from {vq_path}")

        # 1. Load position data
        xyz = np.load(os.path.join(vq_path, 'position_static.npz'))['position']
        xyz_disp = np.load(os.path.join(vq_path, 'position_disp.npz'))['position']
        sort_idx = np.load(os.path.join(vq_path, 'sort_idx.npz'))['sort_idx']

        is_even = np.frombuffer(open(os.path.join(vq_path, 'is_even.bin'), "rb").read(), dtype=np.uint8)[0]
        coeffs_approx = torch.from_numpy(np.load(os.path.join(vq_path, 'motion_wavelet_approx.npz'))['position']).to('cuda')
        coeffs_detail = torch.zeros_like(coeffs_approx)
        coeffs = torch.cat([coeffs_approx, coeffs_detail], dim=1).to(torch.float32)
        wavelet_matrix = generate_haar_matrix(coeffs.shape[1]).to('cuda')
        xyz_motion = wavelet_coeffs_to_points(coeffs, wavelet_matrix)
        if not is_even:
            xyz_motion = xyz_motion[:, :-1, :]
        static_num = xyz.shape[0]
        dynamic_num = xyz_motion.shape[0]
        self.keyframe_num = xyz_motion.shape[1]
        print(f"Loaded {static_num} static and {dynamic_num} dynamic points.")

        # 2. Load codebooks and logits
        try:
            vq_codebooks = np.load(os.path.join(vq_path, 'codebook.npz'))
            logits = np.load(os.path.join(vq_path, 'logits.npz'))
        except FileNotFoundError:
            print("Error: Codebook or logits file not found.")
            return
        except Exception as e:
            print(f"Error loading codebooks or logits: {e}")
            return

        # 3. Load and decode header
        try:
            boundaries, opacity_params = decode_header(os.path.join(vq_path, 'header.bin'))
        except FileNotFoundError:
            print("Error: Header file (header.bin) not found.")
            return
        except Exception as e:
            print(f"Error decoding header: {e}")
            return

        # 4. Calculate shapes and decode indexes
        try:
            shape_dict = get_index_shapes(boundaries, static_num, dynamic_num, self.keyframe_num)
            index_path = os.path.join(vq_path, 'index_bitstream.bin')
            indexes = decode_indexes(index_path, logits, shape_dict)
        except FileNotFoundError:
            print("Error: Index bitstream file (index_bitstream.bin) not found.")
            return
        except Exception as e:
            print(f"Error decoding indexes: {e}")
            return

        # 5. Get SH bitmask
        sh_bitmask = get_sh_bitmask(boundaries)

        # 6. Dequantize opacity values
        try:
            opacities_static = opacity_dequant(indexes.pop('opa'), opacity_params['min_opacity_static'], opacity_params['step_size_static']).to('cuda')
            opacities_motion = opacity_dequant(indexes.pop('opa_motion'), opacity_params['min_opacity_motion'], opacity_params['step_size_motion']).to('cuda')
            opacity_center = opacity_dequant(indexes.pop('opa_center'), opacity_params['min_opacity_center'], opacity_params['step_size_center']).to('cuda').reshape(-1, 2, 1)
            # opacity_var = opacity_dequant(indexes.pop('opa_var'), opacity_params['min_opacity_var'], opacity_params['step_size_var']).to('cuda').reshape(-1, 2, 1)
            # opacities_static = np.load(os.path.join(vq_path, 'opacity_static.npz'))['opacity']
            # opacities_motion = np.load(os.path.join(vq_path, 'opacity_motion.npz'))['opacity']
            # opacity_center = np.load(os.path.join(vq_path, 'opacity_center.npz'))['opacity']
            opacity_var = np.load(os.path.join(vq_path, 'opacity_var.npz'))['opacity']
        except KeyError as e:
             print(f"Error: Missing opacity key in decoded indexes or opacity_params: {e}")
             return
        except Exception as e:
            print(f"Error dequantizing opacity: {e}")
            return


        # 7. Reconstruct VQ attributes
        vq_attributes = {}
        try:
            for vq_key in vq_codebooks.files:
                codebook = vq_codebooks[vq_key]
                # Check if index exists (might be missing if shape/logits were missing during decode_indexes)
                if vq_key not in indexes:
                     print(f"Warning: Index for key '{vq_key}' not found. Skipping attribute reconstruction.")
                     continue
                index = indexes[vq_key].squeeze(1)
                attr = codebook[0, index, :] # Look up attribute using index
                attr = torch.tensor(attr, dtype=torch.float, device="cuda")

                if vq_key == 'rot':  # Handle rotation separately
                    static_rot = attr[:static_num]

                    # Ensure keyframe_num is set before reshaping
                    if self.keyframe_num == 0:
                        if dynamic_num > 0:
                            total_dynamic_rot_elements = attr[static_num:].numel()
                            if total_dynamic_rot_elements == 0:
                                 raise ValueError("Inconsistency: dynamic_num > 0 but no dynamic rotation elements.")
                            if total_dynamic_rot_elements % (dynamic_num * 4) != 0:
                                raise ValueError(f"Cannot infer keyframe_num: total dynamic rotation elements ({total_dynamic_rot_elements}) mismatch with dynamic_num ({dynamic_num}).")
                            inferred_kf_num = total_dynamic_rot_elements // (dynamic_num * 4)
                            print(f"Warning: keyframe_num not set, inferred as {inferred_kf_num}. Explicitly saving/loading keyframe_num is recommended.")
                            self.keyframe_num = inferred_kf_num
                        elif dynamic_num == 0:
                             pass
                        else:
                             raise ValueError("Invalid dynamic_num.")

                    if dynamic_num > 0:
                        dynamic_rot = attr[static_num:].reshape(dynamic_num, self.keyframe_num, 4)
                    else:
                        dynamic_rot = torch.empty((0, self.keyframe_num if self.keyframe_num > 0 else 1, 4), device=attr.device, dtype=attr.dtype)

                    vq_attributes[vq_key] = (static_rot, dynamic_rot)
                else:
                    vq_attributes[vq_key] = self.vq_post_process(vq_key, attr) if vq_key not in ['scale'] else attr
        except Exception as e:
            print(f"Error reconstructing VQ attributes: {e}")
            return


        # 8. Reconstruct feature tensors
        try:
            num_gaussians = static_num + dynamic_num
            # Ensure 'dc' attribute was loaded
            if 'dc' not in vq_attributes:
                 raise ValueError("Decoded 'dc' attribute missing.")
            features_dc = vq_attributes['dc']
            features_extra = torch.zeros((num_gaussians, 15, 3), dtype=torch.float, device="cuda")

            # Apply SH masks safely
            if 'sh1' in vq_attributes and 'sh1' in sh_bitmask:
                features_extra[sh_bitmask['sh1'], :3, :] = vq_attributes['sh1']
            if 'sh2' in vq_attributes and 'sh2' in sh_bitmask:
                features_extra[sh_bitmask['sh2'], 3:8, :] = vq_attributes['sh2']
            if 'sh3' in vq_attributes and 'sh3' in sh_bitmask:
                features_extra[sh_bitmask['sh3'], 8:, :] = vq_attributes['sh3']

            features_extra_unsorted = torch.zeros_like(features_extra)
            features_extra_unsorted[sort_idx] = features_extra

            # Separate static and dynamic features based on post-pruning numbers
            features_dc_static = features_dc[:static_num]
            features_dc_motion = features_dc[static_num:]
            features_extra_static = features_extra_unsorted[:static_num]
            features_extra_motion = features_extra_unsorted[static_num:]

        except Exception as e:
            print(f"Error reconstructing feature tensors: {e}")
            return

        # 9. Set model parameters
        try:
             # Ensure 'scale' attribute was loaded
             if 'scale' not in vq_attributes:
                  raise ValueError("Decoded 'scale' attribute missing.")
             # Ensure 'rot' attribute was loaded
             if 'rot' not in vq_attributes:
                  raise ValueError("Decoded 'rot' attribute missing.")

             self._xyz = nn.Parameter(torch.tensor(xyz, dtype=torch.float, device="cuda"))
             self._xyz_motion = nn.Parameter(xyz_motion)
             self._xyz_disp = nn.Parameter(torch.tensor(xyz_disp, dtype=torch.float, device="cuda"))
             self._features_dc = nn.Parameter(features_dc_static)
             self._features_dc_motion = nn.Parameter(features_dc_motion)
             self._features_rest = nn.Parameter(features_extra_static)
             self._features_rest_motion = nn.Parameter(features_extra_motion)
             self._opacity = nn.Parameter(torch.tensor(opacities_static, dtype=torch.float, device="cuda"))
             self._opacity_motion = nn.Parameter(torch.tensor(opacities_motion, dtype=torch.float, device="cuda"))
             self._opacity_duration_center = nn.Parameter(torch.tensor(opacity_center, dtype=torch.float, device="cuda"))
             self._opacity_duration_var = nn.Parameter(torch.tensor(opacity_var, dtype=torch.float, device="cuda"))
             self._scaling = nn.Parameter(self.scaling_inverse_activation(vq_attributes['scale'][:static_num]))
             self._scaling_motion = nn.Parameter(self.scaling_inverse_activation(vq_attributes['scale'][static_num:]))
             self._rotation = nn.Parameter(vq_attributes['rot'][0]) # Static rotations
             self._rotation_motion = nn.Parameter(vq_attributes['rot'][1]) # Dynamic rotations

             # Initialize masks as zeros, they are not saved in compressed format
             self.gs_mask = nn.Parameter(torch.zeros(static_num, 1, device="cuda"))
             self.sh_mask = nn.Parameter(torch.zeros(static_num, 3, device="cuda"))
             self.gs_mask_motion = nn.Parameter(torch.zeros(dynamic_num, 1, device="cuda"))
             self.sh_mask_motion = nn.Parameter(torch.zeros(dynamic_num, 3, device="cuda"))

        except Exception as e:
            print(f"Error setting model parameters: {e}")
            return

        # 10. Set active SH degree
        self.active_sh_degree = self.max_sh_degree
        print(f"Successfully loaded compressed model.")
