# Model optimization

`espnet_onnx` supports the node fusion logic for ESPnet model. You can improve inference performance with this node fusion.



## How to optimize

Simply add `optimize=True` while exporting is enough.

```python
from espnet_onnx.export import ASRModelExport

m = ASRModelExport()
m.export_from_zip(
  'path/to/the/zipfile',
  tag_name='tag_name_for_zipped_model',
  optimize=True
)
```

If you are using the command line tool to export your model, then add `apply_optimize` flag.

```shell
python -m espnet_onnx.export \
  --model_type asr \
  --input ${path_to_zip} \
  --tag transformer_lm \
  --apply_optimize 
```



## Supported layers

`espnet_onnx` is currently supporting the following layers for node fusion logic. Note that the customized version of onnxruntime can enable espnet_onnx to fully optimize your model.

| supported layer         | CPU_op | Quant_op | GPU_op | ROCM_op |
| ----------------------- | ------ | -------- | ------ | ------- |
| TransformerEncoderLayer | ◯      | ◯        | ◯      | ◯       |
| TransformerDecoderLayer | ◯      | ◯        | x      | x       |
| ConformerEncoderLayer   | x      | x        | x      | x       |

**NOTE**: The original onnxruntime supports only `TransformerEncoderLayer`. If you want to optimize decoder and conformer layer, please follow the instruction to install the custom version of onnxruntime.



## Performance comparison

Here is the comparison of the inference time for each model. All test was executed on Ryzen3800X and RTX2080TI.

**CPU**

- test data : `np.random.random((1, 100, 80), dtype=np.float32)`

| layer                              | before optimize (msec) | after optimize (msec) | torch (msec) |
| ---------------------------------- | ---------------------- | --------------------- | ------------ |
| TransformerEncoderLayer (18 layer) | 45.02                  | **37.52**             | 89.85        |
| TransformerDecoderLayer (6 layer)  | 9.37                   | **6.88**              | 20.32        |
| ConformerEncoderLayer              | x                      | x                     | x            |



**CPU (quantized)**

- test data : `np.random.random((1, 100, 80), dtype=np.float32)`

| layer                              | before optimize (msec) | after optimize (msec) | torch (msec) |
| ---------------------------------- | ---------------------- | --------------------- | ------------ |
| TransformerEncoderLayer (18 layer) | 39.55                  | **34.07**             | 53.65        |
| TransformerDecoderLayer (6 layer)  | 5.98                   | **4.32**              | 8.71         |
| ConformerEncoderLayer              | x                      | x                     | x            |



## Install custom onnxruntime

The custom version of onnxruntime is released [here](https://github.com/Masao-Someki/onnxruntime/releases). Run the following command to download and install it.

```shell
wget install https://github.com/Masao-Someki/onnxruntime/releases/v1.11.1-espnet_onnx.dist
pip install *.dist
```



