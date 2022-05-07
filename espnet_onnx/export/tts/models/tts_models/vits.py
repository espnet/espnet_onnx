
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
from espnet_onnx.export.asr.models.language_models.lm import Embedding

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
        self.emb = Embedding(model.emb) # RelPositionalEncoding
        self.encoder = model.encoder
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
        return z, m, logs, x_mask


class OnnxVITSGenerator(nn.Module):
    def __init__(self, model, use_teacher_forcing):
        # HP
        super().__init__()
        self.spks = model.spks
        self.spk_embed_dim = model.spk_embed_dim
        self.langs = model.langs
        self.use_teacher_forcing = use_teacher_forcing

        # models
        self.text_encoder = TextEncoder(model.text_encoder)
        if hasattr(model, 'global_conv'):
            self.global_emb = model.global_emb
        if hasattr(model, 'spemb_proj'):
            self.spemb_proj = model.spemb_proj
        if hasattr(model, 'lang_emb'):
            self.lang_emb = model.lang_emb
        self.posterior_encoder = PosteriorEncoder(model.posterior_encoder)
        self.flow = model.flow
        self._fix_flows() # remove torch.flip() from onnx model.
        self.decoder = model.decoder
        self.duration_predictor = model.duration_predictor
        
    def _fix_flows(self):
        # since it seems utorch.flip contains a bug in onnx conversion.
        # So I will flip weight/bias of conv to avoid torch.flip.
        flip = False
        for i in range(len(self.flows)):
            if isinstance(self.flows[i], FlipFlow):
                self.flows[i] = OnnxFlipFlow()
                if not flip:
                    flip = True
            elif isinstance(self.flows[i], ConvFlow) and flip:
                self.flows[i].input_conv.weight = \
                    nn.Parameter(torch.flip(self.flows[i].input_conv.weight, [1]))

    def forward(
        self,
        text: torch.Tensor,
        text_lengths: torch.Tensor,
        x_mask: torch.Tensor,
        y_mask: torch.Tensor,
        path: torch.Tensor,
        feats: Optional[torch.Tensor] = None,
        feats_lengths: Optional[torch.Tensor] = None,
        dur: Optional[torch.Tensor] = None,
        sids: Optional[torch.Tensor] = None,
        spembs: Optional[torch.Tensor] = None,
        lids: Optional[torch.Tensor] = None,
        noise_scale: float = 0.667,
        noise_scale_dur: float = 0.8,
        alpha: float = 1.0,
        max_len: int = None
    ):
        # encoder
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
        self.generator = OnnxVITSGenerator(model.generator, use_teacher_forcing)
        self.use_teacher_forcing = use_teacher_forcing
        self.require_sids = model.generator.spks is not None
        self.require_lids = model.generator.langs is not None
        self.noise_scale = 0.667
        self.noise_scale_dur = 0.8
        self.alpha = 1.0
        self.y_mask_length = 0
        self.max_len = None

    def forward(self,
                text: torch.Tensor,
                text_lengths: torch.Tensor,
                x_mask: torch.Tensor,
                y_mask: torch.Tensor,
                path: torch.Tensor,
                feats: Optional[torch.Tensor] = None,
                feats_lengths: Optional[torch.Tensor] = None,
                durations: Optional[torch.Tensor] = None,
                sids: Optional[torch.Tensor] = None,
                spembs: Optional[torch.Tensor] = None,
                lids: Optional[torch.Tensor] = None,
                ):
        if not self.use_teacher_forcing:
            assert durations is not None

        if self.require_sids:
            assert sids is not None
            assert spembs is not None
            sids = sids.view(1)

        if self.require_lids:
            assert lids is not None
            lids = lids.view(1)

        if self.use_teacher_forcing:
            # feats
            assert feats is not None
            feats = feats[None].transpose(1, 2)

        wav, att_w, dur = self.generator(
            text=text,
            text_lengths=text_lengths,
            path=path,
            x_mask=x_mask,
            y_mask=y_mask,
            feats=feats,
            feats_lengths=feats_lengths,
            dur=durations,
            sids=sids,
            spembs=spembs,
            lids=lids,
            noise_scale=self.noise_scale,
            noise_scale_dur=self.noise_scale_dur,
            alpha=self.alpha,
            max_len=self.max_len,
        )
        return wav.view(-1), att_w[0], dur[0]

    def get_dummy_inputs(self):
        ret = []
        text = torch.LongTensor([0, 1]).unsqueeze(0)
        text_lengths = torch.tensor(
            [text.size(1)],
            dtype=torch.long,
        )
        x_mask = make_non_pad_mask(text_lengths).unsqueeze(1)
        ret += [text, text_lengths, x_mask]
        
        if self.use_teacher_forcing:
            feats = torch.randn(1,
                self.generator.posterior_encoder.input_conv.in_channels, text.size(1))
            feats_lengths = text_lengths
            y_mask = make_non_pad_mask(feats_lengths).unsqueeze(1)
            self.y_mask_length = int(feats_lengths)
            path = torch.zeros(1, feats_lengths, text_lengths)
            ret += [y_mask, path, feats, feats_lengths, None]
        else:
            x = torch.randn(1, self.generator.duration_predictor.pre.in_channels, text_lengths)
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
            ret += [y_mask, path, None, None, dur.squeeze(1)]
        
        if self.require_sids:
            sids = torch.LongTensor([0]).unsqueeze(0)
            spemb = torch.randn(1, self.generator.spemb_proj.in_features)
            ret += [sids, spemb]
        else:
            ret += [None, None]
        
        if self.require_lids:
            lids = torch.LongTensor([0]).unsqueeze(0)
            ret += [lids]
        else:
            ret += [None]
        
        return tuple(ret)

    def get_input_names(self):
        return ['text', 'text_lengths', 'x_mask', 'y_mask', 'path',
            'feats', 'feat_lengths', 'durations', 'sids', 'spembs', 'lids']

    def get_output_names(self):
        return ['wav', 'att_w', 'out_duration']

    def get_dynamic_axes(self):
        ret = {}
        ret.update({
            'text': { 1: 'text_length' },
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
        return [DurationPredictor(self.generator.duration_predictor, inverse=True, noise_scale=self.noise_scale_dur)]

    def get_model_config(self, path):
        return {
            'model_path': path / 'tts_model.onnx',
            'submodel': {
                'duration_predictor': {
                    'model_path': path / 'duration_predictor.onnx'
                }
            }
        }


class OnnxFlipFlow(torch.nn.Module):
    """Flip flow module."""

    def forward(
        self, x: torch.Tensor, *args, inverse: bool = False, **kwargs
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """Calculate forward propagation.
        Args:
            x (Tensor): Input tensor (B, channels, T).
            inverse (bool): Whether to inverse the flow.
        Returns:
            Tensor: Flipped tensor (B, channels, T).
            Tensor: Log-determinant tensor for NLL (B,) if not inverse.
        """
        if not inverse:
            logdet = x.new_zeros(x.size(0))
            return x, logdet
        else:
            return x


class DurationPredictor(nn.Module, AbsSubModel):
    def __init__(self, model, inverse=False, noise_scale=1.0):
        # sub model for 
        super().__init__()
        self.model = model
        self.inverse = inverse
        self.noise_scale = noise_scale
        if inverse:
            flows = list(reversed(model.flows))
            self.flows = flows[:-2] + [flows[-1]]
        else:
            self.flows = model.flows
        self._fix_flows()
        
    def _fix_flows(self):
        # since it seems utorch.flip contains a bug in onnx conversion.
        # So I will flip weight/bias of conv to avoid torch.flip.
        flip = False
        for i in range(len(self.flows)):
            if isinstance(self.flows[i], FlipFlow):
                self.flows[i] = OnnxFlipFlow()
                if not flip:
                    flip = True
            elif isinstance(self.flows[i], ConvFlow) and flip:
                self.flows[i].input_conv.weight = \
                    nn.Parameter(torch.flip(self.flows[i].input_conv.weight, [1]))

    def forward(self, x, x_mask, z, w, g):
        x = self.model.pre(x)
        if g is not None:
            x = x + self.model.global_conv(g)
        x = self.model.dds(x, x_mask)
        x = self.model.proj(x) * x_mask

        if not self.inverse:
            assert w is not None, "w must be provided."
            h_w = self.model.post_pre(w)
            h_w = self.model.post_dds(h_w, x_mask)
            h_w = self.model.post_proj(h_w) * x_mask
            z_q = z
            logdet_tot_q = 0.0
            for flow in self.model.post_flows:
                z_q, logdet_q = flow(z_q, x_mask, g=(x + h_w))
                logdet_tot_q += logdet_q
            z_u, z1 = torch.split(z_q, [1, 1], 1)
            u = torch.sigmoid(z_u) * x_mask
            z0 = (w - u) * x_mask
            logdet_tot_q += torch.sum(
                (F.logsigmoid(z_u) + F.logsigmoid(-z_u)) * x_mask, [1, 2]
            )
            logq = (
                torch.sum(-0.5 * (math.log(2 * math.pi) + (z**2)) * x_mask, [1, 2])
                - logdet_tot_q
            )

            logdet_tot = 0
            z0, logdet = self.model.log_flow(z0, x_mask)
            logdet_tot += logdet
            z = torch.cat([z0, z1], 1)
            for flow in self.flows:
                z, logdet = flow(z, x_mask, g=x, inverse=self.inverse)
                logdet_tot = logdet_tot + logdet
            nll = (
                torch.sum(0.5 * (math.log(2 * math.pi) + (z**2)) * x_mask, [1, 2])
                - logdet_tot
            )
            return nll + logq  # (B,)
        else:
            for flow in self.flows:
                z = flow(z, x_mask, g=x, inverse=self.inverse)
                
            z0, z1 = z.split(1, 1)
            logw = z0
            return logw

    def get_dummy_inputs(self):
        ret = []
        # text = [0, 1]
        x = torch.randn(1, self.model.pre.in_channels, 2)
        _text_lengths = torch.tensor(
            [2],
            dtype=torch.long,
        )
        x_mask = make_non_pad_mask(_text_lengths).unsqueeze(1)
        ret += [x, x_mask]
        
        # feed random value
        if not self.inverse:
            e_q = torch.randn(0, 2, _text_lengths) * x_mask
            w = torch.randn(1, 1, _text_lengths)
            ret += [e_q, w]
        else:
            z = torch.randn(1, 2, _text_lengths)* self.noise_scale
            ret += [z, None]
        
        if hasattr(self.model, 'global_conv'):
            g = torch.randn(1, self.model.global_conv.in_features, 1)
            ret.append(g)
        else:
            ret.append(None)
        
        return tuple(ret)
    
    def get_input_names(self):
        return ['x', 'x_mask', 'z', 'w', 'g']

    def get_output_names(self):
        return ['logw']
    
    def get_dynamic_axes(self):
        ret = {
            'x': {2: 'x_length'},
            'x_mask': { 2: 'x_mask_length'},
            'z': {2: 'z_length'},
        }
        if not self.inverse:
            ret.update({
                'w': {2: 'w_length'}
            })
        return ret    

    def get_model_name(self):
        return 'duration_predictor'