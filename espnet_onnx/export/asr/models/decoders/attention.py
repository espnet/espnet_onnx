
import torch
import torch.nn as nn
import torch.nn.functional as F

from espnet.nets.pytorch_backend.rnn.attentions import (
    NoAtt,
    AttDot,
    AttAdd,
    AttLoc,
    AttLoc2D,
    AttLocRec,
    AttCov,
    AttCovLoc,
    AttMultiHeadDot,
    AttMultiHeadAdd,
    AttMultiHeadLoc,
    AttMultiHeadMultiResLoc
)


def get_attention(model):
    if isinstance(model, NoAtt):
        return OnnxNoAtt(model)
    elif isinstance(model, AttDot):
        return OnnxAttDot(model)
    elif isinstance(model, AttAdd):
        raise ValueError('not supported.')
    elif isinstance(model, AttLoc):
        return OnnxAttLoc(model)
    elif isinstance(model, AttLoc2D):
        raise ValueError('not supported.')
    elif isinstance(model, AttLocRec):
        raise ValueError('not supported.')
    elif isinstance(model, AttCov):
        raise ValueError('not supported.')
    elif isinstance(model, AttCovLoc):
        raise ValueError('not supported.')
    elif isinstance(model, AttMultiHeadDot):
        raise ValueError('not supported.')
    elif isinstance(model, AttMultiHeadAdd):
        raise ValueError('not supported.')
    elif isinstance(model, AttMultiHeadLoc):
        raise ValueError('not supported.')
    elif isinstance(model, AttMultiHeadMultiResLoc):
        raise ValueError('not supported.')


class OnnxNoAtt(torch.nn.Module):
    """No attention"""

    def __init__(self, model):
        super().__init__()
        self.model = model

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
        c = torch.zeros(1, enc_h.size(2))
        # initialize attention weight with uniform dist.
        if att_prev is None:
            # if no bias, 0 0-pad goes 0
            c = torch.sum(
                enc_h * mask.view(batch, h_length, 1), dim=1
            )

        return c, att_prev


class OnnxAttDot(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

        self.dunits = model.dunits
        self.att_dim = model.att_dim

    def forward(
        self,
        dec_z,
        att_prev,
        pre_compute_enc_h,
        enc_h,
        mask,
        scaling=2.0
    ):
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


class OnnxAttLoc(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.dunits = model.dunits
        self.att_dim = model.att_dim

    def forward(
        self,
        dec_z,
        att_prev,
        pre_compute_enc_h,
        enc_h,
        mask,
        scaling=2.0,
        last_attended_idx=None,
        backward_window=1,
        forward_window=3,
    ):
        batch = 1
        dec_z = dec_z.view(batch, self.dunits)
        # att_prev: utt x frame -> utt x 1 x 1 x frame
        # -> utt x att_conv_chans x 1 x frame
        frame_length = att_prev.size(1)
        att_conv = self.model.loc_conv(
            att_prev.view(batch, 1, 1, frame_length))
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
        if last_attended_idx is not None:
            e = _apply_attention_constraint(
                e, last_attended_idx, backward_window, forward_window
            )

        w = F.softmax(scaling * e, dim=1)

        # weighted sum over flames
        # utt x hdim
        c = torch.sum(enc_h * w.view(batch, frame_length, 1), dim=1)

        return c, w