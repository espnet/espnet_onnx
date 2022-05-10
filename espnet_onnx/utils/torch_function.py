
import torch
import torch.nn as nn
import numpy as np

MAX_SEQ_LEN = 512
mask_pad = torch.Tensor(1 - np.tri(MAX_SEQ_LEN))

def make_pad_mask(lengths, xs=None, length_dim=-1, maxlen=None):
    """Make mask tensor containing indices of padded part.
    This implementation creates the same mask tensor with original make_pad_mask,
    which can be converted into onnx format.
    Dimension of xs should be 2 or 3.
    """
    if length_dim == 0:
        raise ValueError("length_dim cannot be 0: {}".format(length_dim))
    
    if xs is not None and len(xs.shape) == 3:
        if length_dim == 1:
            lengths = lengths.unsqueeze(1).expand(*xs.transpose(1, 2).shape[:2])
        else:
            lengths = lengths.unsqueeze(1).expand(*xs.shape[:2])
            
    if maxlen is not None:
        m = maxlen
    elif xs is not None:
        m = xs.shape[-1]
    else:
        m = torch.max(lengths)
        
    mask = mask_pad[lengths - 1][..., :m]
    
    if length_dim == 1:
        return mask.transpose(1, 2)
    else:
        return mask
