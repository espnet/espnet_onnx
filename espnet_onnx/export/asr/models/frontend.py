

from espnet2.asr.frontend.default import espnetDefaultFrontend

from espnet_onnx.export.asr.frontends import (
    MaskEstimator
)

def get_frontend_models(model):
    if isinstance(model, espnetDefaultFrontend):
        ret = {}
        if model.frontend.use_wpe:
            ret['wpe'] = MaskEstimator(model.frontend.wpe.mask_est)
            
        if model.frontend.use_beamformer:
            ret['beamformer'] = {
                'AttentionReference': AttentionReference(model.frontend.beamformer.ref),
                'MaskEstimator': MaskEstimator(model.frontend.beamformer.mask)
            }
        return ret
    else:
        raise ValueError('not supported.')
        

def get_front_model_configs(encoder, models, path):
    model_config = dict()
    if 'wpe' in models.keys():
        model_config.update(
            wpe=dict(
                mask_estimator=models['wpe'].get_model_config(path),
                iterations=encoder.frontend.wpe.iterations,
                taps=encoder.frontend.wpe.taps,
                delay=encoder.frontend.wpe.delay,
                normalization=encoder.frontend.wpe.normalization,
                use_dnn_mask=encoder.frontend.wpe.use_dnn_mask,
            )
        )
    if 'beamformer' in models.keys():
            model_config.update(
                beamformer=dict(
                    mask_estimator=models['beamformer']['MaskEstimator'].get_model_config(path),
                    ref=models['beamformer']['AttentionReference'].get_model_config(path),
                    ref_channel=encoder.frontend.beamformer.ref_channel,
                    nmask=encoder.frontend.beamformer.nmask,
                    beamformer_type=encoder.frontend.beamformer.beamformer_type,
                )
            )
        return model_config
    
    else:
        raise ValueError('not supported.')
















