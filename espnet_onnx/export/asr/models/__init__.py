from .ctc import CTC
from .lm import LanguageModel

# encoder
from espnet2.asr.encoder.rnn_encoder import RNNEncoder as espnetRNNEncoder
from espnet2.asr.encoder.vgg_rnn_encoder import VGGRNNEncoder as espnetVGGRNNEncoder
from espnet2.asr.encoder.contextual_block_transformer_encoder import ContextualBlockTransformerEncoder as espnetContextualTransformer
from espnet2.asr.encoder.contextual_block_conformer_encoder import ContextualBlockConformerEncoder as espnetContextualConformer
from .encoders.rnn import RNNEncoder
from .encoders.xformer import XformerEncoder
from .encoders.contextual_block_xformer import ContextualBlockXformerEncoder

# decoder
from espnet2.asr.decoder.rnn_decoder import RNNDecoder as espnetRNNDecoder
from .decoders.rnn import (
    RNNDecoder,
    PreDecoder
)
from .decoders.xformer import XformerDecoder


def get_encoder(model):
    if isinstance(model, espnetRNNEncoder) or isinstance(model, espnetVGGRNNEncoder):
        return RNNEncoder(model)
    elif isinstance(model, espnetContextualTransformer) or isinstance(model, espnetContextualConformer):
        return ContextualBlockXformerEncoder(model)
    else:
        return XformerEncoder(model)


def get_decoder(model):
    if isinstance(model, espnetRNNDecoder):
        return RNNDecoder(model)
    else:
        return XformerDecoder(model)
