import torch
import torch.nn as nn
import torch.nn.functional as F
from espnet.nets.pytorch_backend.rnn.attentions import (
    AttAdd, AttCov, AttCovLoc, AttDot, AttLoc, AttLoc2D, AttLocRec,
    AttMultiHeadAdd, AttMultiHeadDot, AttMultiHeadLoc, AttMultiHeadMultiResLoc,
    NoAtt)

from espnet_onnx.utils.torch_function import normalize


def get_attention(model):
    if isinstance(model, NoAtt):
        return OnnxNoAtt(model)
    elif isinstance(model, AttDot):
        return OnnxAttDot(model)
    elif isinstance(model, AttAdd):
        return OnnxAttAdd(model)
    elif isinstance(model, AttLoc):
        return OnnxAttLoc(model)
    elif isinstance(model, AttLoc2D):
        raise ValueError("Currently AttLoc2D is not supported.")
    elif isinstance(model, AttLocRec):
        raise ValueError("not supported.")
    elif isinstance(model, AttCov):
        return OnnxAttCov(model)
    elif isinstance(model, AttCovLoc):
        return OnnxAttCovLoc(model)
    elif isinstance(model, AttMultiHeadDot):
        raise ValueError("not supported.")
    elif isinstance(model, AttMultiHeadAdd):
        raise ValueError("not supported.")
    elif isinstance(model, AttMultiHeadLoc):
        raise ValueError("not supported.")
    elif isinstance(model, AttMultiHeadMultiResLoc):
        raise ValueError("not supported.")


def require_tanh(model):
    if isinstance(model, AttDot):
        return True
    else:
        return False


class OnnxNoAtt(torch.nn.Module):
    """No attention"""

    def __init__(self, model):
        super().__init__()
        self.model = model
        self.att_type = "noatt"

    def get_dynamic_axes(self):
        return 1

    def forward(
        self,
        dec_z,
        att_prev,
        pre_compute_enc_h,
        enc_h,
        mask,
    ):
        batch = 1
        h_length = enc_h.size(1)
        # initialize attention weight with uniform dist.
        # if no bias, 0 0-pad goes 0
        c = torch.sum(enc_h * att_prev.view(batch, h_length, 1), dim=1)
        return c, att_prev


class OnnxAttDot(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

        self.dunits = model.dunits
        self.att_dim = model.att_dim
        self.att_type = "dot"

    def get_dynamic_axes(self):
        return 1

    def forward(self, dec_z, att_prev, pre_compute_enc_h, enc_h, mask, scaling=2.0):
        batch = 1
        dec_z = dec_z.view(batch, self.dunits)
        h_length = enc_h.size(1)

        e = torch.sum(
            pre_compute_enc_h
            * torch.tanh(self.model.mlp_dec(dec_z)).view(batch, 1, self.att_dim),
            dim=2,
        )  # utt x frame

        # NOTE consider zero padding when compute w.
        e = e + mask
        w = F.softmax(scaling * e, dim=1)

        # weighted sum over flames
        # utt x hdim
        # NOTE use bmm instead of sum(*)
        c = torch.sum(enc_h * w.view(batch, h_length, 1), dim=1)
        return c, w


class OnnxAttAdd(torch.nn.Module):
    """Additive attention"""

    def __init__(self, model):
        super().__init__()
        self.model = model
        self.dunits = model.dunits
        self.att_dim = model.att_dim
        self.att_type = "add"

    def get_dynamic_axes(self):
        return 1

    def forward(self, dec_z, att_prev, pre_compute_enc_h, enc_h, mask, scaling=2.0):
        """AttAdd forward"""
        batch = 1
        h_length = enc_h.size(1)
        dec_z = dec_z.view(batch, self.dunits)

        # dec_z_tiled: utt x frame x att_dim
        dec_z_tiled = self.model.mlp_dec(dec_z).view(batch, 1, self.att_dim)

        # dot with gvec
        # utt x frame x att_dim -> utt x frame
        e = self.model.gvec(torch.tanh(pre_compute_enc_h + dec_z_tiled)).squeeze(2)

        # NOTE consider zero padding when compute w.
        e = e + mask
        w = F.softmax(scaling * e, dim=1)

        # weighted sum over flames
        # utt x hdim
        # NOTE use bmm instead of sum(*)
        c = torch.sum(enc_h * w.view(batch, h_length, 1), dim=1)
        return c, w


class OnnxAttLoc(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.dunits = model.dunits
        self.att_dim = model.att_dim
        self.att_type = "location"

    def get_dynamic_axes(self):
        return 1

    def forward(
        self,
        dec_z,
        att_prev,
        pre_compute_enc_h,
        enc_h,
        mask,
        scaling=2.0,
        last_att_mask=None,
    ):
        batch = 1
        dec_z = dec_z.view(batch, self.dunits)
        # att_prev: utt x frame -> utt x 1 x 1 x frame
        # -> utt x att_conv_chans x 1 x frame
        att_conv = self.model.loc_conv(att_prev.view(batch, 1, 1, enc_h.size(1)))
        # att_conv: utt x att_conv_chans x 1 x frame -> utt x frame x att_conv_chans
        att_conv = att_conv.squeeze(2).transpose(1, 2)
        # att_conv: utt x frame x att_conv_chans -> utt x frame x att_dim
        att_conv = self.model.mlp_att(att_conv)

        # dec_z_tiled: utt x frame x att_dim
        dec_z_tiled = self.model.mlp_dec(dec_z).view(batch, 1, self.att_dim)

        # dot with gvec
        # utt x frame x att_dim -> utt x frame
        e = self.model.gvec(
            torch.tanh(att_conv + pre_compute_enc_h + dec_z_tiled)
        ).squeeze(2)

        # mask is an array with -inf
        e = e + mask

        # apply monotonic attention constraint (mainly for TTS)
        if last_att_mask is not None:
            e = e + last_att_mask

        w = F.softmax(scaling * e, dim=1)

        # weighted sum over flames
        # utt x hdim
        c = torch.sum(enc_h * w.view(batch, enc_h.size(1), 1), dim=1)

        return c, w


# class OnnxAttLoc2D(torch.nn.Module):
#     """2D location-aware attention
#     """
#     def __init__(self, model):
#         super().__init__()
#         self.model = model

#         self.dunits = model.dunits
#         self.att_dim = model.att_dim
#         self.att_win = model.att_win
#         self.att_type = "location2d"

#     def forward(
#         self,
#         dec_z,
#         att_prev,
#         pre_compute_enc_h,
#         enc_h,
#         mask,
#         scaling=2.0
#     ):
#         """AttLoc2D forward
#         """
#         batch = 1
#         h_length = enc_h.size(1)
#         dec_z = dec_z.view(batch, self.dunits)

#         # att_prev: B x att_win x Tmax -> B x 1 x att_win x Tmax -> B x C x 1 x Tmax
#         att_conv = self.model.loc_conv(att_prev.unsqueeze(1))
#         # att_conv: B x C x 1 x Tmax -> B x Tmax x C
#         att_conv = att_conv.squeeze(2).transpose(1, 2)
#         # att_conv: utt x frame x att_conv_chans -> utt x frame x att_dim
#         att_conv = self.model.mlp_att(att_conv)

#         # dec_z_tiled: utt x frame x att_dim
#         dec_z_tiled = self.model.mlp_dec(dec_z).view(batch, 1, self.att_dim)

#         # dot with gvec
#         # utt x frame x att_dim -> utt x frame
#         e = self.model.gvec(
#             torch.tanh(att_conv + pre_compute_enc_h + dec_z_tiled)
#         ).squeeze(2)

#         # NOTE consider zero padding when compute w.
#         e = e + mask
#         w = F.softmax(scaling * e, dim=1)

#         # weighted sum over flames
#         # utt x hdim
#         # NOTE use bmm instead of sum(*)
#         c = torch.sum(enc_h * w.view(batch, h_length, 1), dim=1)

#         # update att_prev: B x att_win x Tmax -> B x att_win+1 x Tmax
#         # -> B x att_win x Tmax
#         att_prev = torch.cat([att_prev, w.unsqueeze(1)], dim=1)
#         att_prev = att_prev[:, 1:]

#         return c, att_prev


class OnnxAttCov(torch.nn.Module):
    """Coverage mechanism attention
    Reference: Get To The Point: Summarization with Pointer-Generator Network
       (https://arxiv.org/abs/1704.04368)
    """

    def __init__(self, model):
        super().__init__()
        self.model = model

        self.dunits = model.dunits
        self.att_dim = model.att_dim
        self.att_type = "coverage"

    def get_dynamic_axes(self):
        return 2

    def forward(self, dec_z, att_prev, pre_compute_enc_h, enc_h, mask, scaling=2.0):
        """AttCov forward"""
        batch = 1
        h_length = enc_h.size(1)
        dec_z = dec_z.view(batch, self.dunits)

        # att_prev_list: L' * [B x T] => cov_vec B x T
        cov_vec = torch.sum(att_prev, dim=0)
        # cov_vec: B x T => B x T x 1 => B x T x att_dim
        cov_vec = self.model.wvec(cov_vec.unsqueeze(-1))

        # dec_z_tiled: utt x frame x att_dim
        dec_z_tiled = self.model.mlp_dec(dec_z).view(batch, 1, self.att_dim)

        # dot with gvec
        # utt x frame x att_dim -> utt x frame
        e = self.model.gvec(
            torch.tanh(cov_vec + pre_compute_enc_h + dec_z_tiled)
        ).squeeze(2)

        # NOTE consider zero padding when compute w.
        e = e + mask
        w = F.softmax(scaling * e, dim=1)
        att_prev = torch.cat([att_prev, w.unsqueeze(0)], dim=0)

        # weighted sum over flames
        # utt x hdim
        # NOTE use bmm instead of sum(*)
        c = torch.sum(enc_h * w.view(batch, h_length, 1), dim=1)

        return c, att_prev


class OnnxAttCovLoc(torch.nn.Module):
    """Coverage mechanism location aware attention

    This attention is a combination of coverage and location-aware attentions.
    """

    def __init__(self, model):
        super().__init__()
        self.model = model

        self.dunits = model.dunits
        self.att_dim = model.att_dim
        self.att_type = "coverage_location"

    def get_dynamic_axes(self):
        return 2

    def forward(self, dec_z, att_prev, pre_compute_enc_h, enc_h, mask, scaling=2.0):
        """AttCovLoc forward"""

        batch = 1
        h_length = enc_h.size(1)
        dec_z = dec_z.view(batch, self.dunits)

        # att_prev_list: L' * [B x T] => cov_vec B x T
        cov_vec = torch.sum(att_prev, dim=0)

        # cov_vec: B x T -> B x 1 x 1 x T -> B x C x 1 x T
        att_conv = self.model.loc_conv(cov_vec.view(batch, 1, 1, h_length))
        # att_conv: utt x att_conv_chans x 1 x frame -> utt x frame x att_conv_chans
        att_conv = att_conv.squeeze(2).transpose(1, 2)
        # att_conv: utt x frame x att_conv_chans -> utt x frame x att_dim
        att_conv = self.model.mlp_att(att_conv)

        # dec_z_tiled: utt x frame x att_dim
        dec_z_tiled = self.model.mlp_dec(dec_z).view(batch, 1, self.att_dim)

        # dot with gvec
        # utt x frame x att_dim -> utt x frame
        e = self.model.gvec(
            torch.tanh(att_conv + pre_compute_enc_h + dec_z_tiled)
        ).squeeze(2)

        # NOTE consider zero padding when compute w.
        e = e + mask
        w = F.softmax(scaling * e, dim=1)
        att_prev = torch.cat([att_prev, w.unsqueeze(0)], dim=0)

        # weighted sum over flames
        # utt x hdim
        # NOTE use bmm instead of sum(*)
        c = torch.sum(enc_h * w.view(batch, h_length, 1), dim=1)

        return c, att_prev


class OnnxAttForward(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

        self.dunits = model.dunits
        self.att_dim = model.att_dim
        self.att_type = "att_for"

    def get_dynamic_axes(self):
        return 1

    def forward(
        self,
        dec_z,
        att_prev,
        pre_compute_enc_h,
        enc_h,
        mask,
        scaling=2.0,
        last_att_mask=None,
    ):
        """AttForward forward"""
        batch = 1

        # att_prev: utt x frame -> utt x 1 x 1 x frame
        # -> utt x att_conv_chans x 1 x frame
        att_conv = self.model.loc_conv(att_prev.view(batch, 1, 1, enc_h.size(1)))
        # att_conv: utt x att_conv_chans x 1 x frame -> utt x frame x att_conv_chans
        att_conv = att_conv.squeeze(2).transpose(1, 2)
        # att_conv: utt x frame x att_conv_chans -> utt x frame x att_dim
        att_conv = self.model.mlp_att(att_conv)

        # dec_z_tiled: utt x frame x att_dim
        dec_z_tiled = self.model.mlp_dec(dec_z).unsqueeze(1)

        # dot with gvec
        # utt x frame x att_dim -> utt x frame
        e = self.model.gvec(
            torch.tanh(pre_compute_enc_h + dec_z_tiled + att_conv)
        ).squeeze(2)

        # NOTE: consider zero padding when compute w.
        e = e + mask

        # apply monotonic attention constraint (mainly for TTS)
        if last_att_mask is not None:
            e = e + last_att_mask

        w = F.softmax(scaling * e, dim=1)

        # forward attention
        att_prev_shift = F.pad(att_prev, (1, 0))[:, :-1]
        w = (att_prev + att_prev_shift) * w
        # NOTE: clamp is needed to avoid nan gradient
        w = normalize(torch.clamp(w, 1e-6), p=1, dim=1)

        # weighted sum over flames
        # utt x hdim
        # NOTE use bmm instead of sum(*)
        c = torch.sum(enc_h * w.unsqueeze(-1), dim=1)

        return c, w
