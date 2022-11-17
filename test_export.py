from espnet_onnx.export import ASRModelExport
from espnet2.bin.asr_inference import Speech2Text as EspnetSpeech2Text
from espnet_onnx import Speech2Text
import librosa
import time
import torch
import onnxruntime as rt

# read audio
tag = 'pyf98/librispeech_conformer'
wav_file = '029f6450-447a-11e9-a9a5-5dbec3b8816a.wav'
audio, sr = librosa.load(wav_file)
espnet_model = EspnetSpeech2Text.from_pretrained(tag)
print('audio shape', audio.shape)

# get features using espnet frontend
print(espnet_model.asr_model.frontend)
audio_tensor = torch.Tensor(audio).unsqueeze(0)
audio_lengths = torch.Tensor([len(audio)])
print('audio tensor shape', audio_tensor.shape)
print('audio_lengths', audio_lengths)
feats, feats_lens = espnet_model.asr_model.frontend(audio_tensor, audio_lengths)
print('feats shape', feats.shape)

# export encoder
print('start export encoder')
audio_len = 1000
feats_dim = 80
dummy_input = (torch.randn(1, audio_len, feats_dim), [audio_len])

# Masao's
# dummy_feats = torch.randn(1, 100, feats_dim)
# dummy_input = ((dummy_feats),torch.ones(dummy_feats[:, :, 0].shape).sum(dim=-1).type(torch.long))

# Actual input
# dummy_input = (feats, feats_lens)

torch.onnx.export(
    espnet_model.asr_model.encoder,
    dummy_input,
    'encoder.onnx',
    verbose=False,
    opset_version=15,
    input_names=["feats", "feats_lens"],
    output_names=["enc_out, enc_out_lens"],
    dynamic_axes={
        "feats": {1: "feats_length"},
        "enc_out": {1: "enc_out_length"},
    }
)

# test espnet encoder
print('test encoder')
# encoder_out, encoder_out_lens, _ = espnet_model.asr_model.encoder(feats, feats_lens)
# print('encoder out shape', encoder_out.shape)

# test onnx encoder
print('test ONNX encoder')
options = rt.SessionOptions()
options.graph_optimization_level = rt.GraphOptimizationLevel.ORT_ENABLE_ALL
onnx_model = rt.InferenceSession('encoder.onnx', options)

# input_name = onnx_model.get_inputs()[0].name
# output_name = onnx_model.get_outputs()[0].name
onnx_enc_out, onnx_enc_out_lens = onnx_model.run(None, {'feats': feats.numpy()})

print('onnx enc out shape', onnx_enc_out.shape)

## espnet inference
#print('running regular inference')
#start_time = time.time()
#nbest = espnet_model(audio)
#transcript = nbest[0][0]
#latency = time.time() - start_time
#print(f'transcript={transcript}, latency={latency}')
#
## onnx export
## m = ASRModelExport()
## m.export(espnet_model, tag)
## onnx_model = Speech2Text(tag_name=tag)
#
## onnx inference
#print('running ONNX inference')
#start_time = time.time()
#nbest = onnx_model(audio)
#transcript = nbest[0][0]
#latency = time.time() - start_time
#print(f'transcript={transcript}, latency={latency}')
