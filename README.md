# espnet_onnx
**ESPNet without PyTorch!**

Users can easily export espnet models to onnx format.

There is no need to install PyTorch or ESPNet on your machine if you already have exported files!

**Note**: This repository is not registered to pypi yet, so please clone and place scripts in some directory.



## Table of Contents

- Usage
- API Reference
- Modification from ESPNet
- Supported archs



## Usage

1. Run the following command to setup environment

```shell
cd tools

# setup environment for only inference
make

# setup environment for model exportation
make venv onnx_export
```

2. For model export, you can export pretrained model from  `espnet_model_zoo`.

   By default, exported onnx files will be stored in `${HOME}/.cache/espnet_onnx/<tag_name>`. 

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



## Modification from ESPNet

To avoid the cache problem, I modified some scripts from the original espnet implementation.

1. Add `<blank>` before `<sos>`
2. Give some `torch.zeros()` arrays to the model.
3. Remove the first token in post process. (remove `blank`)



And I removed `extend_pe()` from positional encoding module. The length of `pe` is 512 by default. 



## Supported Archs

**Encoder**

| arch name                         | supported  |
| --------------------------------- | ---------- |
| ConformerEncoder                  | ◯          |
| ContextualBlockConformerEncoder   | not tested |
| ContextualBlockTransformerEncoder | not tested |
| FairseqHubertEncoder              | not tested |
| RNNEncoder                        | ×          |
| TransformerEncoder                | ◯          |
| VGGRNNEncoder                     | not tested |
| FairSeqWav2Vec2Encoder            | not tested |



**Decoder**

| arch name                                  | supported  |
| ------------------------------------------ | ---------- |
| RNNDecoder                                 | ×          |
| TransformerDecoder                         | ◯          |
| LightweightConvolutionTransformerDecoder   | not tested |
| LightweightConvolution2DTransformerDecoder | not tested |
| DynamicConvolutionTransformerDecoder       | not tested |
| DynamicConvolution2DTransformerDecoder     | not tested |



**Language Model**

| arch name       | supported |
| --------------- | --------- |
| SequentialRNNLM | ◯         |
| TransformerLM   | ◯         |



**pre encoder**

not supported.



**post encoder**

not supported.



**transducer**

not supported.



## References

- [ESPNet: end-to-end speech processing toolkit](https://github.com/espnet/espnet)
- [ESPNet Model Zoo](https://github.com/espnet/espnet_model_zoo)



## Author

Masao Someki

contact: `masao.someki@gmail.com`

