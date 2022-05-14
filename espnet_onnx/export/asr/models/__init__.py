from .ctc import CTC
from .joint_network import JointNetwork

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
from espnet2.asr.decoder.transformer_decoder import TransformerDecoder as espnetTransformerDecoder
from espnet2.asr.transducer.transducer_decoder import TransducerDecoder as espnetTransducerDecoder
from .decoders.rnn import (
    RNNDecoder,
    PreDecoder
)
from .decoders.xformer import XformerDecoder
from .decoders.transducer import TransducerDecoder

# lm
from espnet2.lm.seq_rnn_lm import SequentialRNNLM as espnetSequentialRNNLM
from espnet2.lm.transformer_lm import TransformerLM as espnetTransformerLM
from .language_models.seq_rnn import SequentialRNNLM
from .language_models.transformer import TransformerLM


def get_encoder(model, export_config):
    if isinstance(model, espnetRNNEncoder) or isinstance(model, espnetVGGRNNEncoder):
        return RNNEncoder(model, **export_config)
    elif isinstance(model, espnetContextualTransformer) or isinstance(model, espnetContextualConformer):
        return ContextualBlockXformerEncoder(model, **export_config)
    else:
        return XformerEncoder(model, **export_config)


def get_decoder(model, export_config):
    if isinstance(model, espnetRNNDecoder):
        return RNNDecoder(model, **export_config)
    elif isinstance(model, espnetTransducerDecoder):
        return TransducerDecoder(model, **export_config)
    else:
        return XformerDecoder(model, **export_config)


def get_lm(model, export_config):
    if isinstance(model, espnetSequentialRNNLM):
        return SequentialRNNLM(model, **export_config)
    elif isinstance(model, espnetTransformerLM):
        return TransformerLM(model, **export_config)
