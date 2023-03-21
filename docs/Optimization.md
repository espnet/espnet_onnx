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
| ConformerEncoderLayer   | ◯      | ◯        | ◯      | x       |

**NOTE**
- The original onnxruntime supports only `TransformerEncoderLayer`. If you want to optimize decoder and conformer layer, please follow the instruction to install the custom version of onnxruntime.
- As for Conformer optimization, CPU supports `RelPosAttention` node to improve performance.
  However, this op cannot efficiently calculate when sequence length is long or hidden dimension is large. (e.g., sequence length > 256 or hidden dimensios > 512)
  `RelativeShift` op can be effectively improve inference speed for both CPU and GPU op.
- Currently optimization with `RelPosAttention` node is not supported with torch==1.12

## Performance comparison

Here is the comparison of the inference time for each model. All test was executed on Ryzen3800X and RTX2080TI.

**CPU**

- test data : `np.random.random((1, 100, 80), dtype=np.float32)`

| layer                                   | number of layers | before optimize (msec) | after optimize (msec) | torch (msec) |
| --------------------------------------- | ---------------- | ---------------------- | --------------------- | ------------ |
| TransformerEncoderLayer                 | 18               | 45.02                  | **37.52**             | 89.85        |
| TransformerDecoderLayer                 | 6                | 9.37                   | **6.88**              | 20.32        |
| ConformerEncoderLayer (RelativeShift)   | 12               | 26.82                  | **23.76**             | 48.59        |
| ConformerEncoderLayer (RelPosAttention) | 12               | -                      | **22.68**             | -            |



**CPU (quantized)**

- test data : `np.random.random((1, 100, 80), dtype=np.float32)`

| layer                                   | number of layers | before optimize (msec) | after optimize (msec) | torch (msec) |
| --------------------------------------- | ---------------- | ---------------------- | --------------------- | ------------ |
| TransformerEncoderLayer                 | 18               | 39.55                  | **34.07**             | 53.65        |
| TransformerDecoderLayer                 | 6                | 5.98                   | **4.32**              | 8.71         |
| ConformerEncoderLayer (RelativeShift)   | 12               | 20.29                  | **19.04**             | 32.21        |
| ConformerEncoderLayer (RelPosAttention) | 12               | -                      | **17.69**             | -            |



## Install custom onnxruntime

The custom version of onnxruntime is released [here](https://github.com/Masao-Someki/onnxruntime/releases). Run the following command to download and install it.

```shell
wget install https://github.com/Masao-Someki/espnet_onnx/releases/download/custom_ort_v1.11.1-espnet_onnx/onnxruntime-1.11.1_espnet_onnx-cp38-cp38-linux_x86_64.whl
pip install *.dist
```
