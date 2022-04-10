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

2. If you want to export pretrained model, you need to install `torch>=1.11.0`, `espnet`, `espnet_model_zoo` additionally.

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

3. For inference, `tag_name` or `model_dir` is used to load onnx file. `tag_name` has to be defined in `tag_config.yaml` 

```python
import librosa
from espnet_onnx import Speech2Text

speech2text = Speech2Text(tag_name='<tag name>')
# speech2text = Speech2Text(model_dir='path to the onnx directory')

y, sr = librosa.load('sample.wav', sr=16000)
nbest = speech2text(y)
```



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

ASR: [Supported architecture for ASR](./doc/ASRSupported.md)



## References

- [ESPNet: end-to-end speech processing toolkit](https://github.com/espnet/espnet)
- [ESPNet Model Zoo](https://github.com/espnet/espnet_model_zoo)



## COPYRIGHT

Copyright (c) 2022 Maso Someki

Released under [MIT licence](https://opensource.org/licenses/mit-license.php)



## Author

Masao Someki

contact: `masao.someki@gmail.com`
