



def get_tts_model(model):
    if isinstance(model, espnetVITSModel):
        return onnxVITSModel(model)
    else:
        raise RuntimeError('Currently only VITS model is supported.')