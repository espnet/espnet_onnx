"""Exportable inverse spline flow used by the VITS duration predictor."""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class OnnxConvFlow(nn.Module):
    """Use a tensor-only inverse of ESPnet's linear-tail ConvFlow."""

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x, x_mask, g=None, inverse=False):
        if not inverse:
            return self.model(x, x_mask, g=g, inverse=False)

        model = self.model
        xa, xb = x.split(x.size(1) // 2, 1)
        h = model.proj(model.dds_conv(model.input_conv(xa), x_mask, g=g)) * x_mask
        batch, channels, length = xa.shape
        h = h.reshape(batch, channels, -1, length).permute(0, 1, 3, 2)
        scale = math.sqrt(model.hidden_channels)
        widths = F.softmax(h[..., : model.bins] / scale, dim=-1)
        heights = F.softmax(h[..., model.bins : 2 * model.bins] / scale, dim=-1)
        widths = 0.001 + (1 - 0.001 * model.bins) * widths
        heights = 0.001 + (1 - 0.001 * model.bins) * heights
        bound = model.tail_bound
        cumwidths = 2 * bound * F.pad(torch.cumsum(widths, dim=-1), (1, 0)) - bound
        cumheights = 2 * bound * F.pad(torch.cumsum(heights, dim=-1), (1, 0)) - bound
        widths = cumwidths[..., 1:] - cumwidths[..., :-1]
        heights = cumheights[..., 1:] - cumheights[..., :-1]

        constant = math.log(math.exp(1 - 0.001) - 1)
        derivatives = 0.001 + F.softplus(
            F.pad(h[..., 2 * model.bins :], (1, 1), value=constant)
        )
        inside = (xb >= -bound) & (xb <= bound)
        inputs = xb.clamp(-bound, bound)
        # Search in all bins without boolean indexing, which has a dynamic shape.
        index = torch.sum(inputs.unsqueeze(-1) >= cumheights, dim=-1) - 1
        index = index.clamp(0, model.bins - 1).unsqueeze(-1)

        def gather(values):
            return values.gather(-1, index).squeeze(-1)

        input_cumwidths = gather(cumwidths)
        input_widths = gather(widths)
        input_cumheights = gather(cumheights)
        input_heights = gather(heights)
        delta = input_heights / input_widths
        derivative = gather(derivatives)
        next_derivative = gather(derivatives[..., 1:])
        offset = inputs - input_cumheights
        a = offset * (derivative + next_derivative - 2 * delta) + input_heights * (
            delta - derivative
        )
        b = input_heights * derivative - offset * (derivative + next_derivative - 2 * delta)
        c = -delta * offset
        root = (2 * c) / (-b - torch.sqrt(b.square() - 4 * a * c))
        transformed = root * input_widths + input_cumwidths
        xb = torch.where(inside, transformed, xb)
        return torch.cat([xa, xb], dim=1) * x_mask
