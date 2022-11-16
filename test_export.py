from espnet_onnx.export import ASRModelExport
from espnet2.bin.asr_inference import Speech2Text as EspnetSpeech2Text
from espnet_onnx import Speech2Text
import librosa
import time
import torch

tag = 'pyf98/librispeech_conformer'
wav_file = '029f6450-447a-11e9-a9a5-5dbec3b8816a.wav'
audio, sr = librosa.load(wav_file)
espnet_model = EspnetSpeech2Text.from_pretrained(tag)


# test export encoder

print(audio.shape)
audio_len = 250000
feats_dim = 512
dummy_input = torch.randn(1, audio_len, feats_dim), [audio_len]

print(espnet_model.asr_model.encoder)
torch.onnx.export(
    espnet_model.asr_model.encoder,
    dummy_input,
    'example.onnx',
    verbose=True,
    opset_version=15,
    input_names=["input"],
    output_names=["output"],
    dynamic_axes={
        "input": {1: "audio_len"},
        "output": {1: "audio_len"},
    }
)

print('done exporting')

# espnet inference
print('running regular inference')
start_time = time.time()
nbest = espnet_model(audio)
transcript = nbest[0][0]
latency = time.time() - start_time
print(f'transcript={transcript}, latency={latency}')

# onnx export
# m = ASRModelExport()
# m.export(espnet_model, tag)
# onnx_model = Speech2Text(tag_name=tag)

# onnx inference
print('running ONNX inference')
start_time = time.time()
nbest = onnx_model(audio)
transcript = nbest[0][0]
latency = time.time() - start_time
print(f'transcript={transcript}, latency={latency}')
