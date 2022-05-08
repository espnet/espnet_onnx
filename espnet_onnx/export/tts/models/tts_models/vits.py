
from typing import (
    Union,
    Tuple,
    Optional
)

import math
import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from espnet2.gan_tts.vits.monotonic_align.core import maximum_path_c

    is_cython_avalable = True
except ImportError:
    is_cython_avalable = False
    warnings.warn(
        "Cython version is not available. Fallback to 'EXPERIMETAL' numba version. "
        "If you want to use the cython version, please build it as follows: "
        "`cd espnet2/gan_tts/vits/monotonic_align; python setup.py build_ext --inplace`"
    )
    
from espnet2.gan_tts.vits.monotonic_align import maximum_path_numba
from espnet2.gan_tts.vits.flow import (
    FlipFlow,
    ConvFlow
)
from espnet.nets.pytorch_backend.nets_utils import make_non_pad_mask
from espnet_onnx.export.asr.models.language_models.lm import get_pos_emb

from ..abs_model import AbsModel, AbsSubModel

def maximum_path(neg_x_ent: torch.Tensor, attn_mask: torch.Tensor, path: torch.Tensor) -> torch.Tensor:
    """Calculate maximum path.
    Args:
        neg_x_ent (Tensor): Negative X entropy tensor (B, T_feats, T_text).
        attn_mask (Tensor): Attention mask (B, T_feats, T_text).
        path (Tensor): Path. np.zeros(neg_x_ent.shape)
    Returns:
        Tensor: Maximum path tensor (B, T_feats, T_text).
    """
    device, dtype = neg_x_ent.device, neg_x_ent.dtype
    neg_x_ent = neg_x_ent.cpu().numpy().astype(np.float32)
    t_t_max = attn_mask.sum(1)[:, 0].cpu().numpy().astype(np.int32)
    t_s_max = attn_mask.sum(2)[:, 0].cpu().numpy().astype(np.int32)
    if is_cython_avalable:
        maximum_path_c(path, neg_x_ent, t_t_max, t_s_max)
    else:
        maximum_path_numba(path, neg_x_ent, t_t_max, t_s_max)

    return torch.from_numpy(path).to(device=device, dtype=dtype)


class TextEncoder(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.attention_dim = model.attention_dim
        self.emb = model.emb
        self.encoder = model.encoder
        self.encoder.embed[0] = get_pos_emb(self.encoder.embed[0])
        self.proj = model.proj
    
    def forward(self, text, x_mask):
        x = self.emb(text) * math.sqrt(self.attention_dim)
        # encoder assume the channel last (B, T_text, attention_dim)
        # but mask shape shoud be (B, 1, T_text)
        x, _ = self.encoder(x, x_mask)

        # convert the channel first (B, attention_dim, T_text)
        x = x.transpose(1, 2)
        stats = self.proj(x) * x_mask
        m, logs = stats.split(stats.size(1) // 2, dim=1)

        return x, m, logs


class PosteriorEncoder(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.input_conv = model.input_conv
        self.encoder = model.encoder
        self.proj = model.proj
    
    def forward(self, x, x_mask, g=None):
        x = self.input_conv(x) * x_mask
        x = self.encoder(x, x_mask, g=g)
        stats = self.proj(x) * x_mask
        m, logs = stats.split(stats.size(1) // 2, dim=1)
        z = (m + torch.randn_like(m) * torch.exp(logs)) * x_mask
        return z, m, logs


class OnnxVITSGenerator(nn.Module):
    def __init__(self, model, use_teacher_forcing):
        # HP
        super().__init__()
        self.use_teacher_forcing = use_teacher_forcing

        # models
        self.posterior_encoder = PosteriorEncoder(model.posterior_encoder)
        self.flow = model.flow
        self._fix_flows() # remove torch.flip() from onnx model.
        self.decoder = model.decoder
        self.duration_predictor = model.duration_predictor
        
    def _fix_flows(self):
        # since it seems utorch.flip contains a bug in onnx conversion.
        # So I will flip weight/bias of conv to avoid torch.flip.
        flip = False
        new_flows = nn.ModuleList()
        for f in self.flow.flows:
            if isinstance(f, FlipFlow):
                if not flip:
                    flip = True
            elif isinstance(f, ConvFlow) and flip:
                f.input_conv.weight = \
                    nn.Parameter(torch.flip(f.input_conv.weight, [1]))
                new_flows.append(f)
            else:
                new_flows.append(f)
        self.flow.flows = new_flows

    def forward(
        self,
        m_p: Optional[torch.Tensor],
        logs_p: Optional[torch.Tensor],
        g: torch.Tensor,
        x_mask: torch.Tensor,
        y_mask: torch.Tensor,
        path: torch.Tensor,
        feats: Optional[torch.Tensor] = None,
        dur: Optional[torch.Tensor] = None,
        noise_scale: float = 0.667,
        noise_scale_dur: float = 0.8,
        alpha: float = 1.0,
        max_len: int = None
    ):
        if self.use_teacher_forcing:
            # forward posterior encoder
            z, m_q, logs_q = self.posterior_encoder(
                feats, y_mask, g=g)

            # forward flow
            z_p = self.flow(z, y_mask, g=g)  # (B, H, T_feats)

            # monotonic alignment search
            s_p_sq_r = torch.exp(-2 * logs_p)  # (B, H, T_text)
            # (B, 1, T_text)
            neg_x_ent_1 = torch.sum(
                -0.5 * math.log(2 * math.pi) - logs_p,
                [1],
                keepdim=True,
            )
            # (B, T_feats, H) x (B, H, T_text) = (B, T_feats, T_text)
            neg_x_ent_2 = torch.matmul(
                -0.5 * (z_p**2).transpose(1, 2),
                s_p_sq_r,
            )
            # (B, T_feats, H) x (B, H, T_text) = (B, T_feats, T_text)
            neg_x_ent_3 = torch.matmul(
                z_p.transpose(1, 2),
                (m_p * s_p_sq_r),
            )
            # (B, 1, T_text)
            neg_x_ent_4 = torch.sum(
                -0.5 * (m_p**2) * s_p_sq_r,
                [1],
                keepdim=True,
            )
            # (B, T_feats, T_text)
            neg_x_ent = neg_x_ent_1 + neg_x_ent_2 + neg_x_ent_3 + neg_x_ent_4
            # (B, 1, T_feats, T_text)
            attn_mask = torch.unsqueeze(
                x_mask, 2) * torch.unsqueeze(y_mask, -1)
            # monotonic attention weight: (B, 1, T_feats, T_text)
            attn = maximum_path(
                neg_x_ent,
                attn_mask.squeeze(1),
                path,
            ).unsqueeze(1)
            dur = attn.sum(2).squeeze(1)  # (B, 1, T_text) -> (B, T_text)

            # forward decoder with random segments
            wav = self.decoder(z * y_mask, g=g)
        else:
            # duration, y_mask is computed outside this onnx model.
            attn_mask = torch.unsqueeze(
                x_mask, 2) * torch.unsqueeze(y_mask, -1)
            
            attn = self._generate_path(dur, attn_mask, path)

            # expand the length to match with the feature sequence
            # (B, T_feats, T_text) x (B, T_text, H) -> (B, H, T_feats)
            m_p = torch.matmul(
                attn.squeeze(1),
                m_p.transpose(1, 2),
            ).transpose(1, 2)
            # (B, T_feats, T_text) x (B, T_text, H) -> (B, H, T_feats)
            logs_p = torch.matmul(
                attn.squeeze(1),
                logs_p.transpose(1, 2),
            ).transpose(1, 2)

            # decoder
            z_p = m_p + torch.randn_like(m_p) * torch.exp(logs_p) * noise_scale
            z = self.flow(z_p, y_mask, g=g, inverse=True)
            wav = self.decoder((z * y_mask)[:, :, :max_len], g=g)

        return wav.squeeze(1), attn.squeeze(1), dur

    def _generate_path(self, dur, mask, path):
        """
        t_y = mask.shape[2]
        path = torch.arange(t_y, dtype=dur.dtype)
        """
        b, _, t_y, t_x = mask.shape
        cum_dur = torch.cumsum(dur.unsqueeze(1), -1)
        cum_dur_flat = cum_dur.view(b * t_x)
        path = path.unsqueeze(0) < cum_dur_flat.unsqueeze(1)
        path = path.view(b, t_x, t_y).to(dtype=mask.dtype)
        # path will be like (t_x = 3, t_y = 5):
        # [[[1., 1., 0., 0., 0.],      [[[1., 1., 0., 0., 0.],
        #   [1., 1., 1., 1., 0.],  -->   [0., 0., 1., 1., 0.],
        #   [1., 1., 1., 1., 1.]]]       [0., 0., 0., 0., 1.]]]
        path = (path * F.pad(path, [0, 0, 1, 0, 0, 0])[:, :-1]).type(torch.float32)
        return path.unsqueeze(1).transpose(2, 3) * mask


class OnnxVITSModel (nn.Module, AbsModel):
    def __init__(self, model, use_teacher_forcing: bool = False):
        super().__init__()
        self.model = model
        self.generator = OnnxVITSGenerator(model.generator, use_teacher_forcing)
        self.use_teacher_forcing = use_teacher_forcing
        self.require_sids = model.generator.spks is not None
        self.require_lids = model.generator.langs is not None
        self.noise_scale = 0.667
        self.noise_scale_dur = 0.8
        self.alpha = 1.0
        self.y_mask_length = 0
        self.max_len = None

    def forward(
        self,
        m_p: Optional[torch.Tensor],
        logs_p: Optional[torch.Tensor],
        g: torch.Tensor,
        x_mask: torch.Tensor,
        y_mask: torch.Tensor,
        path: torch.Tensor,
        feats: Optional[torch.Tensor] = None,
        durations: Optional[torch.Tensor] = None,
    ):
        if not self.use_teacher_forcing:
            assert durations is not None

        if self.use_teacher_forcing:
            # feats
            assert feats is not None
            feats = feats[None].transpose(1, 2)

        wav, att_w, dur = self.generator(
            m_p=m_p,
            logs_p=logs_p,
            g=g,
            path=path,
            x_mask=x_mask,
            y_mask=y_mask,
            feats=feats,
            dur=durations,
            noise_scale=self.noise_scale,
            noise_scale_dur=self.noise_scale_dur,
            alpha=self.alpha,
            max_len=self.max_len,
        )
        return wav.view(-1), att_w[0], dur[0]

    def get_dummy_inputs(self):
        ret = []
        # _text = torch.LongTensor([0, 1]).unsqueeze(0)
        _text_lengths = torch.tensor(
            [2],
            dtype=torch.long,
        )
        m_p = torch.randn(1, self.model.generator.text_encoder.attention_dim, _text_lengths)
        logs_p = torch.randn(1, self.model.generator.text_encoder.attention_dim, _text_lengths)
        x_mask = make_non_pad_mask(_text_lengths).unsqueeze(1)
        if self.require_sids and self.require_lids:
            g = torch.randn(1, self.model.generator.global_emb.embedding_dim, 1)
        else: g = None
        
        ret += [m_p, logs_p, g, x_mask]
        
        if self.use_teacher_forcing:
            feats = torch.randn(1,
                self.generator.posterior_encoder.input_conv.in_channels, text.size(1))
            feats_lengths = text_lengths
            y_mask = make_non_pad_mask(feats_lengths).unsqueeze(1)
            self.y_mask_length = int(feats_lengths)
            path = torch.zeros(1, feats_lengths, text_lengths)
            ret += [y_mask, path, feats, None]
        else:
            x = torch.randn(1, self.generator.duration_predictor.pre.in_channels, _text_lengths)
            _logw = self.generator.duration_predictor(x, x_mask, inverse=True, noise_scale=self.noise_scale_dur)
            _w = torch.exp(_logw) * x_mask * self.alpha
            dur = torch.ceil(_w)
            _y_length = torch.clamp_min(
                torch.sum(dur, [1,2]), 1
            ).long()
            y_mask = make_non_pad_mask(_y_length).unsqueeze(1)
            self.y_mask_length = int(_y_length)
            _attn_mask = torch.unsqueeze(x_mask, 2) * torch.unsqueeze(y_mask, -1)
            path = torch.arange(_attn_mask.shape[2])
            ret += [y_mask, path, None, dur.squeeze(1)]

        return tuple(ret)

    def get_input_names(self):
        return ['m_p', 'logs_p', 'x_mask', 'y_mask', 'path',
            'feats', 'durations']

    def get_output_names(self):
        return ['wav', 'att_w', 'out_duration']

    def get_dynamic_axes(self):
        ret = {}
        ret.update({
            'm_p': { 2: 'm_length' },
            'logs_p': { 2: 'logs_length' },
            'x_mask': { 2: 'x_mask_length'},
            'wav': {0: 'wav_length'},
            'att_w': {1: 'att_w_length'}
        })
        if self.use_teacher_forcing:
            ret.update({
                'feats': { 2: 'feats_length' },
                'path': { 
                    1: 'path_height',
                    2: 'path_width'
                }
            })
        else:
            ret.update({
                'path': { 0: 'path_width'},
                'durations': {1: 'duration_length'}
            })
        
        return ret
    
    def get_submodel(self):
        # VITS requires to separate DurationEstimator into another onnx model.
        return [DurationPredictor(
            self.model.generator,
            self.model.generator.text_encoder,
            noise_scale=self.noise_scale_dur
        )]

    def get_model_config(self, path):
        return {
            'model_path': str(path / 'tts_model.onnx'),
            'model_type': 'VITS',
            'use_teacher_forcing': self.use_teacher_forcing,
            'submodel': {
                'duration_predictor': {
                    'model_path': str(path / 'duration_predictor.onnx')
                },
                "noise_scale": self.noise_scale,
                "alpha": self.alpha
            }
        }


class DurationPredictor(nn.Module, AbsSubModel):
    def __init__(self, model, text_encoder, noise_scale=1.0):
        # sub model for 
        super().__init__()
        self.duration_predictor = model.duration_predictor
        self.text_encoder = TextEncoder(text_encoder)
        self.noise_scale = noise_scale
        self.spks = model.spks
        self.spk_embed_dim = model.spk_embed_dim
        self.langs = model.langs
        flows = list(reversed(self.duration_predictor.flows))
        self.flows = flows[:-2] + [flows[-1]]
        self._fix_flows()
        
        if hasattr(model, 'global_conv'):
            self.global_emb = model.global_emb
        if hasattr(model, 'spemb_proj'):
            self.spemb_proj = model.spemb_proj
        if hasattr(model, 'lang_emb'):
            self.lang_emb = model.lang_emb

    def _fix_flows(self):
        # since it seems utorch.flip contains a bug in onnx conversion.
        # So I will flip weight/bias of conv to avoid torch.flip.
        flip = False
        new_flows = nn.ModuleList()
        for f in self.flows:
            if isinstance(f, FlipFlow):
                if not flip:
                    flip = True
            elif isinstance(f, ConvFlow) and flip:
                f.input_conv.weight = \
                    nn.Parameter(torch.flip(f.input_conv.weight, [1]))
                new_flows.append(f)
            else:
                new_flows.append(f)
        self.flows = new_flows

    def forward(self, text, x_mask, z, sids, spembs, lids):
        x, m_p, logs_p = self.text_encoder(text, x_mask)
        g = None
        if self.spks is not None:
            # (B, global_channels, 1)
            g = self.global_emb(sids.view(-1)).unsqueeze(-1)
        if self.spk_embed_dim is not None:
            # (B, global_channels, 1)
            g_ = self.spemb_proj(F.normalize(
                spembs.unsqueeze(0))).unsqueeze(-1)
            if g is None:
                g = g_
            else:
                g = g + g_
        if self.langs is not None:
            # (B, global_channels, 1)
            g_ = self.lang_emb(lids.view(-1)).unsqueeze(-1)
            if g is None:
                g = g_
            else:
                g = g + g_

        x = self.duration_predictor.pre(x)
        if g is not None:
            x = x + self.duration_predictor.global_conv(g)
        x = self.duration_predictor.dds(x, x_mask)
        x = self.duration_predictor.proj(x) * x_mask
        
        for flow in self.flows:
            z = flow(z, x_mask, g=x, inverse=True)
            
        z0, z1 = z.split(1, 1)
        logw = z0
        return logw, m_p, logs_p, g

    def get_dummy_inputs(self):
        ret = []
        # text = [0, 1]
        x = torch.LongTensor([0, 1]).unsqueeze(0)
        _text_lengths = torch.tensor(
            [2],
            dtype=torch.long,
        )
        x_mask = make_non_pad_mask(_text_lengths).unsqueeze(1)
        z = torch.randn(1, 2, _text_lengths)* self.noise_scale
        ret += [x, x_mask, z]
        
        if self.spks is not None:
            sids = torch.LongTensor([0]).unsqueeze(0)
            spemb = torch.randn(1, self.generator.spemb_proj.in_features)
            ret += [sids, spemb]
        else:
            ret += [None, None]
        
        if self.langs is not None:
            lids = torch.LongTensor([0]).unsqueeze(0)
            ret += [lids]
        else:
            ret += [None]
        
        return tuple(ret)
    
    def get_input_names(self):
        return ['text', 'x_mask', 'z', 'sids', 'spemb', 'lids']

    def get_output_names(self):
        if (self.spks is not None) and (self.langs is not None):
            return ['logw', 'm_p', 'logs_p', 'g']
        else:
            return ['logw', 'm_p', 'logs_p']

    def get_dynamic_axes(self):
        return {
            'text': {1: 'text_length'},
            'x_mask': { 2: 'x_mask_length'},
            'z': {2: 'z_length'},
            'm_p': {2: 'm_length'},
            'logs_p': {2: 'logs_length'},
        }

    def get_model_name(self):
        return 'duration_predictor'