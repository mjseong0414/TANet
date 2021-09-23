from abc import ABCMeta
from abc import abstractmethod
from abc import abstractproperty
from second.core.box_coders import GroundBox3dCoder, BevBoxCoder
from second.pytorch.core import box_torch_ops, box_torch_ops_JRDB
import torch

class GroundBox3dCoderTorch(GroundBox3dCoder):
    def encode_torch(self, boxes, anchors):
        return box_torch_ops.second_box_encode(boxes, anchors, self.vec_encode, self.linear_dim)

    def decode_torch(self, boxes, anchors):
        return box_torch_ops.second_box_decode(boxes, anchors, self.vec_encode, self.linear_dim)



class BevBoxCoderTorch(BevBoxCoder):
    def encode_torch(self, boxes, anchors):
        if self.center_coder == False:
            anchors = anchors[..., [0, 1, 3, 4, 6]]
            boxes = boxes[..., [0, 1, 3, 4, 6]]
            return box_torch_ops.bev_box_encode(boxes, anchors, self.vec_encode, self.linear_dim)
        elif self.center_coder == True:
            anchors = anchors[..., [0, 1, 3, 4, 6]]
            boxes = boxes[..., [0, 1, 3, 4, 6]]
            return box_torch_ops_JRDB.center_box_encode(boxes, anchors, self.vec_encode, self.linear_dim)

    def decode_torch(self, encodings, anchors):
        if self.center_coder == False:
            anchors = anchors[..., [0, 1, 3, 4, 6]]
            ret = box_torch_ops.bev_box_decode(encodings, anchors, self.vec_encode, self.linear_dim)
            z_fixed = torch.full([*ret.shape[:-1], 1], self.z_fixed, dtype=ret.dtype, device=ret.device)
            h_fixed = torch.full([*ret.shape[:-1], 1], self.h_fixed, dtype=ret.dtype, device=ret.device)
            return torch.cat([ret[..., :2], z_fixed, ret[..., 2:4], h_fixed, ret[..., 4:]], dim=-1)
        elif self.center_coder == True:
            anchors = anchors[..., [0, 1, 6]]
            ret = box_torch_ops_JRDB.center_box_decode(encodings, anchors, self.vec_encode, self.linear_dim)
            z_fixed = torch.full([*ret.shape[:-1], 1], self.z_fixed, dtype=ret.dtype, device=ret.device)
            h_fixed = torch.full([*ret.shape[:-1], 1], self.h_fixed, dtype=ret.dtype, device=ret.device)
            l_fixed = torch.full([*ret.shape[:-1], 1], self.l_fixed, dtype=ret.dtype, device=ret.device)
            w_fixed = torch.full([*ret.shape[:-1], 1], self.w_fixed, dtype=ret.dtype, device=ret.device)
            return torch.cat([ret[..., :2], z_fixed, l_fixed, w_fixed, h_fixed, ret[..., 4:]], dim=-1) # xyzwlh
            # return torch.cat([ret[..., :2], z_fixed, w_fixed, l_fixed, h_fixed, ret[..., 4:]], dim=-1) # xyzlwh


