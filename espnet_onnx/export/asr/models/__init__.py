from .ctc import CTC
from .joint_network import JointNetwork

# encoder
from espnet2.asr.encoder.rnn_encoder import RNNEncoder as espnetRNNEncoder
from espnet2.asr.encoder.vgg_rnn_encoder import VGGRNNEncoder as espnetVGGRNNEncoder
from espnet2.asr.encoder.contextual_block_transformer_encoder import ContextualBlockTransformerEncoder as espnetContextualTransformer
from espnet2.asr.encoder.contextual_block_conformer_encoder import ContextualBlockConformerEncoder as espnetContextualConformer
from espnet2.asr.encoder.transformer_encoder import TransformerEncoder as espnetTransformerEncoder
from espnet2.asr.encoder.conformer_encoder import ConformerEncoder as espnetConformerEncoder
from .encoders.rnn import RNNEncoder
from .encoders.transformer import TransformerEncoder
from .encoders.conformer import ConformerEncoder
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

# frontend
from espnet2.asr.frontend.s3prl import S3prlFrontend as espnetS3PRLModel
from .frontends.s3prl import S3PRLModel


def get_encoder(model, frontend, preencoder, export_config):
    if isinstance(model, espnetRNNEncoder) or isinstance(model, espnetVGGRNNEncoder):
        return RNNEncoder(model, preencoder, **export_config)
    elif isinstance(model, espnetContextualTransformer) or isinstance(model, espnetContextualConformer):
        return ContextualBlockXformerEncoder(model, preencoder, **export_config)
    elif isinstance(model, espnetTransformerEncoder):
        return TransformerEncoder(model, frontend, preencoder, **export_config)
    elif isinstance(model, espnetConformerEncoder):
        return ConformerEncoder(model, frontend, preencoder, **export_config)


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


def get_frontend_models(model, export_config):
    if isinstance(model, espnetS3PRLModel):
        return S3PRLModel(model, **export_config)
    else:
        return None
    
    