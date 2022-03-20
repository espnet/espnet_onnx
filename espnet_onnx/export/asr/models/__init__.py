from .decoder import Decoder
from .ctc import CTC
from .lm import LanguageModel

# encoder
from espnet2.asr.encoder.rnn_encoder import RNNEncoder as espnetRNNEncoder
from .encoders.rnn import RNNEncoder
from .encoders.xformer import XformerEncoder


def get_encoder(model):
    if isinstance(model, espnetRNNEncoder):
        return RNNEncoder(model)
    else:
        return XformerEncoder(model)
