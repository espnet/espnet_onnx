# espnet_onnx

![](https://circleci.com/gh/espnet/espnet_onnx.svg?style=shield)
![](https://img.shields.io/badge/licence-MIT-blue)
[![](https://img.shields.io/badge/pypi-0.1.10-brightgreen)](https://pypi.org/project/espnet-onnx/)

**ESPnet without PyTorch!**

Utility library to easily export, quantize, and optimize espnet models to onnx format. 
There is no need to install PyTorch or ESPnet on your machine if you already have exported files!



## Install

1. `espnet_onnx` can be installed with pip

```shell
pip install espnet_onnx
```

2. If you want to export pretrained model, you need to install `torch>=1.11.0`, `espnet`, `espnet_model_zoo`, `onnx` additionally.
`onnx==1.12.0` might cause some errors. If you got an error while inference or exporting, please consider downgrading the onnx version.



## Usage

#### Export models

1. `espnet_onnx` can export pretrained model published on `espnet_model_zoo`. By default, exported files will be stored in `${HOME}/.cache/espnet_onnx/<tag_name>`.

```python
from espnet2.bin.asr_inference import Speech2Text
from espnet_onnx.export import ASRModelExport

m = ASRModelExport()

# download with espnet_model_zoo and export from pretrained model
m.export_from_pretrained('<tag name>', quantize=True)

# export from trained model
speech2text = Speech2Text(args)
m.export(speech2text, '<tag name>', quantize=True)
```

2. You can export pretrained model from zipped file. The zipped file should contain `meta.yaml`.

```python
from espnet_onnx.export import ASRModelExport

m = ASRModelExport()
m.export_from_zip(
  'path/to/the/zipfile',
  tag_name='tag_name_for_zipped_model',
  quantize=True
)
```

3. You can set some configuration for export. The available configurations are shown in the details for each models.

- [Details for ASR models and configurations](./docs/ASRModelDetail.md)
- [Details for TTS models and configurations](./docs/TTSModelDetail.md)
- [Details for optimization configurations](./doc/optimize)

```python
from espnet_onnx.export import ASRModelExport

m = ASRModelExport()
# Set maximum sequence length to 3000
m.set_export_config(max_seq_len=3000)
m.export_from_zip(
  'path/to/the/zipfile',
  tag_name='tag_name_for_zipped_model',
)
```

4. You can easily optimize your model by using the `optimize` option. If you want to fully optimize your model, you need to install the custom version of onnxruntime from [here](https://github.com/Masao-Someki/espnet_onnx/releases/download/custom_ort_v1.11.1-espnet_onnx/onnxruntime-1.11.1_espnet_onnx-cp38-cp38-linux_x86_64.whl). Please read [this document](./docs/Optimization.md) for more detail.

```python
from espnet_onnx.export import ASRModelExport

m = ASRModelExport()
m.export_from_zip(
  'path/to/the/zipfile',
  tag_name='tag_name_for_zipped_model',
  optimize=True,
  quantize=True
)
```

5. You can export model from the command line.

```shell
python -m espnet_onnx.export \
  --model_type asr \
  --input ${path_to_zip} \
  --tag transformer_lm \
  --apply_optimize \
  --apply_quantize
```



#### Inference

1. For inference, `tag_name` or `model_dir` is used to load onnx file. `tag_name` has to be defined in `tag_config.yaml`

```python
import librosa
from espnet_onnx import Speech2Text

speech2text = Speech2Text(tag_name='<tag name>')
# speech2text = Speech2Text(model_dir='path to the onnx directory')

y, sr = librosa.load('sample.wav', sr=16000)
nbest = speech2text(y)
```

3. For streaming asr, you can use `StreamingSpeech2Text` class. The speech length should be the same as `StreamingSpeech2Text.hop_size`

```python
from espnet_onnx import StreamingSpeech2Text

stream_asr = StreamingSpeech2Text(tag_name)

# start streaming asr
stream_asr.start()
while streaming:
  wav = <some code to get wav>
  assert len(wav) == stream_asr.hop_size
  stream_text = stream_asr(wav)[0][0]

# You can get non-streaming asr result with end function
nbest = stream_asr.end()
```

You can also simulate streaming model with your wav file with `simulate` function. Passing `True` as the second argument will show the streaming text as the following code.

```python
import librosa
from espnet_onnx import StreamingSpeech2Text

stream_asr = StreamingSpeech2Text(tag_name)
y, sr = librosa.load('path/to/wav', sr=16000)
nbest = stream_asr.simulate(y, True)
# Processing audio with 6 processes.
# Result at position 0 :
# Result at position 1 :
# Result at position 2 : this
# Result at position 3 : this is
# Result at position 4 : this is a
# Result at position 5 : this is a
print(nbest[0][0])
# 'this is a pen'
```

4. If you installed the custom version of onnxruntime, you can run optimized model for inference. You don't have to change any code from the above. If the model was optimized, then espnet_onnx would automatically load the optimized version. 

5. You can use only hubert model for your frontend.

```python
from espnet_onnx.export import ASRModelExport

# export your model
tag_name = 'ESPnet pretrained model with hubert'
m = ASRModelExport()
m.export_from_pretrained(tag_name, optimize=True)

# load only the frontend model
from espnet_onnx.asr.frontend import Frontend
frontend = Frontend.get_frontend(tag_name)

# use the model in your application
import librosa
y, sr = librosa.load('wav file')
# y: (B, T)
# y_len: (B,)
feats = frontend(y[None,:], np.array([len(y)]))
```

6. If you installed `torch` in your environment, you can use frontend in your training.

```python
from espnet_onnx.asr.frontend import TorchFrontend
frontend = TorchFrontend.get_frontend(tag_name) # load pretrained frontend model

# use the model while training
import librosa
y, sr = librosa.load('wav file')

# You need to place your data on GPU,
# and specify the output shape in tuple
y = torch.Tensor(y).unsqueeze(0).to('cuda') # (1, wav_length)
output_shape = (batch_size, feat_length, feats_dims)
feats = frontend(y, y.size(1), output_shape)
```

#### Text2Speech inference

1. You can export TTS models as ASR models.

```python
from espnet2.bin.tts_inference import Text2Speech
from espnet_onnx.export import TTSModelExport

m = TTSModelExport()

# download with espnet_model_zoo and export from pretrained model
m.export_from_pretrained('<tag name>', quantize=True)

# export from trained model
text2speech = Text2Speech(args)
m.export(text2speech, '<tag name>', quantize=True)
```

2. You can generate wav files with just simply using the Text2Speech class.

```python
from espnet_onnx import Text2Speech

tag_name = 'kan-bayashi/ljspeech_vits'
text2speech = Text2Speech(tag_name, use_quantized=True)

text = 'Hello world!'
output_dict = text2speech(text) # inference with onnx model.
wav = output_dict['wav']
```

## How to use GPU on espnet_onnx

**Install dependency.**

First, we need `onnxruntime-gpu` library, instead of `onnxruntime`. Please follow [this article](https://onnxruntime.ai/docs/execution-providers/CUDA-ExecutionProvider.html) to select and install the correct version of `onnxruntime-gpu`, depending on your CUDA version.

**Inference on GPU**

Now you can speedup the inference speed with GPU. All you need is to select the correct providers, and give it to the `Speech2Text` or `StreamingSpeech2Text` instance. See [this article](https://onnxruntime.ai/docs/execution-providers/) for more information about providers.

```python
import librosa
from espnet_onnx import Speech2Text

PROVIDERS = ['CUDAExecutionProvider']
tag_name = 'some_tag_name'

speech2text = Speech2Text(
  tag_name,
  providers=PROVIDERS
)
y, sr = librosa.load('path/to/wav', sr=16000)
nbest = speech2text(y) # runs on GPU.
```

Note that some quantized models are not supported for GPU computation. If you got an error with quantized model, please try not-quantized model.

## Changes from ESPNet

To avoid the cache problem, I modified some scripts from the original espnet implementation.

1. Add `<blank>` before `<sos>`
2. Give some `torch.zeros()` arrays to the model.
3. Remove the first token in post process. (remove `blank`)
4. Replace `make_pad_mask` into new implementation, which can be converted into onnx format.

5. Removed `extend_pe()` from positional encoding module. The length of `pe` is 512 by default.

## Supported Archs

ASR: [Supported architecture for ASR](./docs/ASRSupported.md)

TTS: [Supported architecture for TTS](./docs/TTSSupported.md)

## References

- [ESPNet: end-to-end speech processing toolkit](https://github.com/espnet/espnet)
- [ESPNet Model Zoo](https://github.com/espnet/espnet_model_zoo)

## COPYRIGHT

Copyright (c) 2022 Maso Someki

Released under [MIT licence](https://opensource.org/licenses/mit-license.php)

## Author

Masao Someki

contact: `masao.someki@gmail.com`
