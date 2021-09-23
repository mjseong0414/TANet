from abc import ABCMeta
from abc import abstractmethod
from abc import abstractproperty
from second.core import box_np_ops
from second.core import box_np_ops_JRDB
import numpy as np

class BoxCoder(object):
    """Abstract base class for box coder."""
    __metaclass__ = ABCMeta

    @abstractproperty
    def code_size(self):
        pass

    def encode(self, boxes, anchors):
        return self._encode(boxes, anchors)

    def decode(self, rel_codes, anchors):
        return self._decode(rel_codes, anchors)

    @abstractmethod
    def _encode(self, boxes, anchors):
        pass

    @abstractmethod
    def _decode(self, rel_codes, anchors):
        pass


class GroundBox3dCoder(BoxCoder):
    def __init__(self, linear_dim=False, vec_encode=False):
        super().__init__()
        self.linear_dim = linear_dim
        self.vec_encode = vec_encode

    @property
    def code_size(self):
        return 8 if self.vec_encode else 7

    def _encode(self, boxes, anchors):
        return box_np_ops.second_box_encode(boxes, anchors, self.vec_encode, self.linear_dim)

    def _decode(self, encodings, anchors):
        return box_np_ops.second_box_decode(encodings, anchors, self.vec_encode, self.linear_dim)


class BevBoxCoder(BoxCoder):
    """WARNING: this coder will return encoding with size=5, but 
    takes size=7 boxes, anchors
    """
    # def __init__(self, linear_dim=False, vec_encode=False, z_fixed=-1.0, h_fixed=2.0):
    def __init__(self, linear_dim=False, vec_encode=False, z_fixed=-1.0, h_fixed=2.0, l_fixed=None, w_fixed=None, center_coder=False):
        super().__init__()
        self.linear_dim = linear_dim
        self.z_fixed = z_fixed
        self.h_fixed = h_fixed
        self.l_fixed = l_fixed
        self.w_fixed = w_fixed
        self.center_coder = center_coder
        self.vec_encode = vec_encode
        
        if self.center_coder == True and (self.l_fixed == None or self.w_fixed == None):
            raise Exception("Set l_fixed and w_fixed for center regression.")

    @property
    def code_size(self):
        if self.center_coder == False:
            return 6 if self.vec_encode else 5
        elif self.center_coder == True:
            return 4 if self.vec_encode else 3

    def _encode(self, boxes, anchors):
        if self.center_coder == False:
            anchors = anchors[..., [0, 1, 3, 4, 6]]
            boxes = boxes[..., [0, 1, 3, 4, 6]]
            return box_np_ops.bev_box_encode(boxes, anchors, self.vec_encode, self.linear_dim)
        elif self.center_coder == True:
            anchors = anchors[..., [0, 1, 6]]
            boxes = boxes[..., [0, 1, 6]]
            return box_np_ops_JRDB.center_box_encode(boxes, anchors, self.vec_encode, self.linear_dim) ##########3 center_box_encode

    def _decode(self, encodings, anchors):
        if self.center_coder == False:
            anchors = anchors[..., [0, 1, 3, 4, 6]]
            ret = box_np_ops.bev_box_decode(encodings, anchors, self.vec_encode, self.linear_dim)
            z_fixed = np.full([*ret.shape[:-1], 1], self.z_fixed, dtype=ret.dtype)
            h_fixed = np.full([*ret.shape[:-1], 1], self.h_fixed, dtype=ret.dtype)
            return np.concatenate([ret[..., :2], z_fixed, ret[..., 2:4], h_fixed, ret[..., 4:]], axis=-1)
        elif self.center_coder == True:
            anchors = anchors[..., [0, 1, 6]]
            ret = box_np_ops_JRDB.center_box_decode(encodings, anchors, self.vec_encode, self.linear_dim) ################## center_box_decode
            z_fixed = np.full([*ret.shape[:-1], 1], self.z_fixed, dtype=ret.dtype)
            h_fixed = np.full([*ret.shape[:-1], 1], self.h_fixed, dtype=ret.dtype)
            l_fixed = np.full([*ret.shape[:-1], 1], self.l_fixed, dtype=ret.dtype)
            w_fixed = np.full([*ret.shape[:-1], 1], self.w_fixed, dtype=ret.dtype)
            return np.concatenate([ret[..., :2], z_fixed, l_fixed, w_fixed, h_fixed, ret[..., 4:]], axis=-1) # xyzwlh
            # return np.concatenate([ret[..., :2], z_fixed, w_fixed, l_fixed, h_fixed, ret[..., 4:]], axis=-1) # xyzlwh