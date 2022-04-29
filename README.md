# espnet_onnx
![](https://circleci.com/gh/Masao-Someki/espnet_onnx.svg?style=shield)
![](https://img.shields.io/badge/licence-MIT-blue)
[![](https://img.shields.io/badge/pypi-0.1.3-brightgreen)](https://pypi.org/project/espnet-onnx/)

**ESPNet without PyTorch!**  

Utility library to easily export espnet models to onnx format. 
There is no need to install PyTorch or ESPNet on your machine if you already have exported files!

**Note**

Currently TTS is not supported.


## Install

1. `espnet_onnx` can be installed with pip

```shell
pip install espnet_onnx
```

2. If you want to export pretrained model, you need to install `torch>=1.11.0`, `espnet`, `espnet_model_zoo`, `onnx` additionally.

## Usage

1. `espnet_onnx` can export pretrained model published on `espnet_model_zoo`.

   By default, exported files will be stored in `${HOME}/.cache/espnet_onnx/<tag_name>`. 

```python
from espnet2.bin.asr_inference import Speech2Text
from espnet_onnx.export import ModelExport

m = ModelExport()

# download with espnet_model_zoo and export from pretrained model
m.export_from_pretrained('<tag name>', quantize=True)

# export from trained model
speech2text = Speech2Text(args)
m.export(speech2text, '<tag name>', quantize=True)
```

2. For inference, `tag_name` or `model_dir` is used to load onnx file. `tag_name` has to be defined in `tag_config.yaml` 

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

4. You can export pretrained model from zipped file. The zipped file should contain `meta.yaml`.

```python
from espnet_onnx.export import ModelExport

m = ModelExport()
m.export_from_zip(
  'path/to/the/zipfile',
  tag_name='tag_name_for_zipped_model',
  quantize=True
)
```

5. You can use GPU for inference. Please see `How to use GPU on espnet_onnx` in detail.


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


## API Reference

`espnet_onnx.Speech2Text`

**args**

- `tag_name` : `tag_name` defined in `table.csv` in `espnet_model_zoo`.
If a user set a custom `model_name` when export model with `export()`, then `tag_name` should be `model_name`. The `tag_name` should be defined in `tag_config.yaml`, which will be created when exporting model.
  
- `model_dir`: Path to the model directory. Configuration file should be located in `<model_dir>/config.yaml`

- `use_quantized`: Flag to use quantized model.



`espnet_onnx.export.ModelExport`

**function**

- `export`
  - `model`: Instance of `espnet2.bin.asr_inference.Speech2Text`.
  - `tag_name`: Tag name to identify onnx model.
  - `quantize`: Flag to create quantized model.
- `export_from_pretrained`
  - `tag_name`: Tag name to identify onnx model.
  - `quantize`: Flag to create quantized model.



## Changes from ESPNet

To avoid the cache problem, I modified some scripts from the original espnet implementation.

1. Add `<blank>` before `<sos>`
2. Give some `torch.zeros()` arrays to the model.
3. Remove the first token in post process. (remove `blank`)

And I removed `extend_pe()` from positional encoding module. The length of `pe` is 512 by default. 



## Supported Archs

ASR: [Supported architecture for ASR](./docs/ASRSupported.md)



## References

- [ESPNet: end-to-end speech processing toolkit](https://github.com/espnet/espnet)
- [ESPNet Model Zoo](https://github.com/espnet/espnet_model_zoo)



## COPYRIGHT

Copyright (c) 2022 Maso Someki

Released under [MIT licence](https://opensource.org/licenses/mit-license.php)



## Author

Masao Someki

contact: `masao.someki@gmail.com`
