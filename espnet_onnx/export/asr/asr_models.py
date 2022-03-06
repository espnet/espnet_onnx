
import torch
import torch.nn as nn


class Encoder(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
        
    def forward(self, speech, mask):
        xs_pad, mask = self.model.embed(speech, mask)
        xs_pad, masks = self.model.encoders(xs_pad, mask)
        xs_pad = self.model.after_norm(xs_pad)
        olens = masks.squeeze(1).sum(1)
        return xs_pad, olens, None


class Decoder(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
        
    def forward(self, tgt, tgt_mask, memory, cache):
        x = self.model.embed(tgt)
        new_cache = []
        for c, decoder in zip(cache, self.model.decoders):
            x, tgt_mask, memory, memory_mask = decoder(
                x, tgt_mask, memory, None, cache=c
            )
            new_cache.append(x) # (1, L, 512) * n_layer
        y = self.model.after_norm(x[:, -1])
        y = torch.log_softmax(self.model.output_layer(y), dim=-1)
        return y, new_cache


class CTC(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
        
    def forward(self, x):
        return torch.log_softmax(self.model(x), dim=2)


# class LM(nn.Module):
#     def __init__(self, model):
#         super().__init__()
#         self.model = model
    
#     def forward(self, )


# class JointNetwork(nn.Module):
#     def __init__(self, model):
#         self.model = model

#     def forward(self, enc_out, dec_out):
#         x = self.model(enc_out, dec_out)
#         return torch.log_softmax(x, dim=-1)


# class TransducerDecoder(nn.Module):
#     def __init__(self, emb, decoder):
#         self.emb = emb
#         self.decoder = decoder

#     def forward(
#         self,
#         label,
#         h_prev,
#         c_prev,
#     ):
#         sequence = self.embed(label)
#         h_nexts = []
#         c_nexts = []

#         for layer in range(len(self.decoder)):
#             if self.dtype == "lstm":
#                 sequence, (
#                     h_next,
#                     c_next,
#                 ) = self.decoder[layer](
#                     sequence, hx=(h_prev[layer], c_prev[layer])
#                 )
#                 h_nexts.append(h_next)
#                 c_nexts.append(c_next)
#             else:
#                 sequence, h_next = self.decoder[layer](
#                     sequence, hx=h_prev[layer]
#                 )
#                 h_nexts.append(h_next)

#         return sequence, h_nexts, c_nexts
