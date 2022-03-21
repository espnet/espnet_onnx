from .ctc import CTC
from .lm import LanguageModel

# encoder
from espnet2.asr.encoder.rnn_encoder import RNNEncoder as espnetRNNEncoder
from .encoders.rnn import RNNEncoder
from .encoders.xformer import XformerEncoder

# decoder
from espnet2.asr.decoder.rnn_decoder import RNNDecoder as espnetRNNDecoder
from .decoders.rnn import (
    RNNDecoder,
    PreDecoder
)
from .decoders.xformer import XformerDecoder


def get_encoder(model):
    if isinstance(model, espnetRNNEncoder):
        return RNNEncoder(model)
    else:
        return XformerEncoder(model)


def get_decoder(model):
    if isinstance(model, espnetRNNDecoder):
        return RNNDecoder(model)
    else:
        return XformerDecoder(model)
