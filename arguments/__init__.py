from argparse import ArgumentParser, Namespace
import sys
import os

class GroupParams:
    pass

class ParamGroup:
    def __init__(self, parser: ArgumentParser, name : str, fill_none = False):
        group = parser.add_argument_group(name)
        for key, value in vars(self).items():
            shorthand = False
            if key.startswith("_"):
                shorthand = True
                key = key[1:]
            t = type(value)
            value = value if not fill_none else None
            if shorthand:
                if t == bool:
                    group.add_argument("--" + key, ("-" + key[0:1]), default=value, action="store_true")
                else:
                    group.add_argument("--" + key, ("-" + key[0:1]), default=value, type=t)
            else:
                if t == bool:
                    group.add_argument("--" + key, default=value, action="store_true")
                else:
                    group.add_argument("--" + key, default=value, type=t)

    def extract(self, args):
        group = GroupParams()
        for arg in vars(args).items():
            if arg[0] in vars(self) or ("_" + arg[0]) in vars(self):
                setattr(group, arg[0], arg[1])
        return group

class ModelParams(ParamGroup):
    def __init__(self, parser, sentinel=False):
        self.sh_degree = 3
        self._source_path = ""
        self._model_path = ""
        self._images = "images"
        self._resolution = -1
        self._white_background = False
        self.data_device = "cuda"
        self.eval = True
        self.model = "cubic"
        self.loader = "neural3dvideo"
        self.interp_type = "cube"
        self.rot_interp_type = "slerp"
        self.lazy_loader = True
        self.llffhold = 8
        self.time_interval = 5
        self.time_pad = 3
        self.var_pad = 3
        self.time_pad_type = 0 # 0: none, 1: reflect 2: repeat
        self.kernel_size = 0.1
        self.start_duration = 5
        self.duration = -1
        self.sample_every = 1
        self.progressive_step = 1
        self.start_timestamp = 0
        self.end_timestamp = -1
        self.near = 0.2
        self.far = 300.0
        super().__init__(parser, "Loading Parameters", sentinel)

    def extract(self, args):
        g = super().extract(args)
        g.source_path = os.path.abspath(g.source_path)
        return g

class PipelineParams(ParamGroup):
    def __init__(self, parser):
        self.convert_SHs_python = False
        self.compute_cov3D_python = False
        self.debug = False
        super().__init__(parser, "Pipeline Parameters")

class OptimizationParams(ParamGroup):
    def __init__(self, parser):
        self.mini_batch = 1
        self.iterations = 40_000
        self.position_lr_init = 0.00016
        self.position_lr_final = 0.0000016
        self.position_lr_delay_mult = 0.01
        self.position_lr_max_steps = 30_000
        self.dynamic_position_lr_init = 0.00016
        self.dynamic_position_lr_final = 0.000016
        self.dynamic_position_lr_delay_mult = 0.01
        self.dynamic_position_lr_max_steps = 30_000
        self.feature_lr = 0.0025
        self.opacity_lr = 0.05
        self.scaling_lr = 0.005
        self.rotation_lr = 0.00001
        self.disp_lr = 0.0001
        self.feature_motion_lr = 0.0025
        self.rotation_motion_lr = 0.001
        self.opacity_motion_lr = 0.05
        self.opacity_motion_center_lr = 0.001
        self.opacity_motion_var_lr = 0.0005
        self.percent_dense = 0.01
        self.lambda_dssim = 0.2
        self.lambda_flow = 0.00
        self.l1_accum = True
        self.densification_interval = 200
        self.opacity_reset_interval = 3000
        self.densify_from_iter = 500
        self.extract_from_iter = 500
        self.densify_until_iter = 30_000
        self.progressive_growing_steps = 300
        self.error_base_prune_steps = 20000
        self.ssim_prune_every = 5
        self.l1_prune_every = 5
        self.make_dynamic_interval = 200 # != densification_interval
        self.extracton_interval = 3000
        self.extract_every = 1
        self.extract_percentile = 0.98
        self.prune_invisible_interval = 6000
        self.densify_grad_threshold = 0.0002
        self.densify_dgrad_threshold = 0.0001
        self.s_max_ssim = 0.6
        self.s_l1_thres = 0.08
        self.d_max_ssim = 0.6
        self.d_l1_thres = 0.08
        self.static_reg = 0.0001
        self.motion_reg = 0.0001
        self.rot_reg = 0.00
        self.coord_reg = 0.00
        self.random_background = True

        # Gaussain prune cfgs
        self.gs_prune_start_iter = 30_001
        self.gs_mask_lr = 0.01
        self.gs_mask_lambda = 0.0005
        # Adaptive SHs prune cfgs
        self.sh_prune_start_iter = 30_001
        self.sh_mask_lr = 0.05
        self.sh_mask_lambda = 0.005
        # Wavelet mask cfgs
        self.wv_prune_start_iter = 30_001
        # VQ cfgs
        self.vq_start_iter = 30_001
        self.rate_constrain_iter = 33_001
        self.reactivate_codeword_period = 1000
        self.vq_cb_lr = 0.0002
        self.vq_logits_lr = 0.002
        self.vq_scale_lmbda = 32768
        self.vq_rot_lmbda = 256
        self.vq_dc_lmbda = 256
        self.vq_sh1_lmbda = 256
        self.vq_sh2_lmbda = 256
        self.vq_sh3_lmbda = 256
        self.vq_scale_cbsize = 8192
        self.vq_rot_cbsize = 8192
        self.vq_dc_cbsize = 8192
        self.vq_sh1_cbsize = 4096
        self.vq_sh2_cbsize = 4096
        self.vq_sh3_cbsize = 4096
        self.vq_patch_size = 16384
        super().__init__(parser, "Optimization Parameters")

def get_combined_args(parser : ArgumentParser):
    cmdlne_string = sys.argv[1:]
    cfgfile_string = "Namespace()"
    args_cmdline = parser.parse_args(cmdlne_string)

    try:
        cfgfilepath = os.path.join(args_cmdline.model_path, "cfg_args")
        print("Looking for config file in", cfgfilepath)
        with open(cfgfilepath) as cfg_file:
            print("Config file found: {}".format(cfgfilepath))
            cfgfile_string = cfg_file.read()
    except TypeError:
        print("Config file not found at")
        pass
    args_cfgfile = eval(cfgfile_string)

    merged_dict = vars(args_cfgfile).copy()
    for k,v in vars(args_cmdline).items():
        if v != None:
            merged_dict[k] = v
    return Namespace(**merged_dict)
