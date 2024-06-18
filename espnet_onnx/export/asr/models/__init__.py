# decoder
from espnet2.asr.decoder.rnn_decoder import RNNDecoder as espnetRNNDecoder
from espnet2.asr.encoder.contextual_block_conformer_encoder import (
    ContextualBlockConformerEncoder as espnetContextualConformer,
)
from espnet2.asr.encoder.contextual_block_transformer_encoder import (
    ContextualBlockTransformerEncoder as espnetContextualTransformer,
)


from espnet_onnx.export.asr.models.layers.ctc import CTC
from espnet_onnx.export.asr.models.decoders.rnn import RNNDecoder
from espnet_onnx.export.asr.models.decoders.transducer import TransducerDecoder
from espnet_onnx.export.asr.models.decoders.xformer import XformerDecoder
from espnet_onnx.export.asr.models.encoders.contextual_block_xformer import (
    ContextualBlockXformerEncoder,
)
from espnet_onnx.export.asr.models.layers.joint_network import JointNetwork

try:
    from espnet2.asr.transducer.transducer_decoder import (
        TransducerDecoder as espnetTransducerDecoder,
    )
except:
    from espnet2.asr.decoder.transducer_decoder import (
        TransducerDecoder as espnetTransducerDecoder,
    )


# lm
# frontend
from espnet2.asr.frontend.s3prl import S3prlFrontend as espnetS3PRLModel
from espnet2.lm.seq_rnn_lm import SequentialRNNLM as espnetSequentialRNNLM
from espnet2.lm.transformer_lm import TransformerLM as espnetTransformerLM

from espnet_onnx.export.asr.models.frontends.s3prl import S3PRLModel
from espnet_onnx.export.asr.models.language_models.seq_rnn import SequentialRNNLM
from espnet_onnx.export.asr.models.language_models.transformer import TransformerLM


# conversion
from espnet_onnx.utils.export_function import replace_modules, get_replace_modules
from espnet_onnx.export.asr.models.encoder_wrapper import DefaultEncoder


def get_encoder(model, frontend, preencoder, export_config, convert_map):
    if isinstance(model, espnetContextualTransformer) or isinstance(
        model, espnetContextualConformer
    ):
        return ContextualBlockXformerEncoder(model, **export_config)
    else:
        _model = replace_modules(
            get_replace_modules(
                convert_map,
                "asr_optimization" if export_config.get("optimize", False) else "asr",
            ),
            model,
            preencoder=preencoder,
            **export_config,
        )
        return DefaultEncoder(_model, frontend, **export_config)


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
