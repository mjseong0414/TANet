import numpy as np

from second.protos import box_coder_pb2
from second.pytorch.core.box_coders import (BevBoxCoderTorch,
                                              GroundBox3dCoderTorch)


def build(box_coder_config, bev_target, l_fixed=None, w_fixed=None):
    """Create optimizer based on config.

    Args:
        optimizer_config: A Optimizer proto message.

    Returns:
        An optimizer and a list of variables for summary.

    Raises:
        ValueError: when using an unsupported input data type.
    """
    box_coder_type = box_coder_config.WhichOneof('box_coder')
    if box_coder_type == 'ground_box3d_coder':
        cfg = box_coder_config.ground_box3d_coder
        return GroundBox3dCoderTorch(cfg.linear_dim, cfg.encode_angle_vector)
    elif box_coder_type == 'bev_box_coder':
        cfg = box_coder_config.bev_box_coder
        if bev_target =='rectangle':
            return BevBoxCoderTorch(cfg.linear_dim, cfg.encode_angle_vector, cfg.z_fixed, cfg.h_fixed)
        elif bev_target == "center":
            return BevBoxCoderTorch(cfg.linear_dim, cfg.encode_angle_vector, cfg.z_fixed, cfg.h_fixed, l_fixed=l_fixed, w_fixed=w_fixed, center_coder=True)
        else:
            raise Exception("unknown bev target")
    else:
        raise ValueError("unknown box_coder type")
