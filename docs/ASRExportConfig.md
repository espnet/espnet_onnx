# Details for models and configurations

This document explains the details of each models, such as export configuration, and input/output argument for onnx model.



## Encoder

### RNNEncoder

**Export configuration**

- `feats_dim` : Dimension of the input feature.
  - type: int
  - default: 80

**model input**

- `feats` : Input feature of the speech. 
  - shape: `(1, feats_length, feats_dim)`
  - dynamic dimension: 1
  - dtype: `float32`
- `feats_length`: Length of input feature.
  - shape: `(1,)`
  - dynamic_axes: None
  - dtype: `int64`

**model output**

- `encoder_out`: Output feature of encoder.
  - shape: `(1, feats_length, encoder_feats_dim)`
  - dynamic dimension: 1
  - dtype: `float32`
- `encoder_out_lens`: Length of output feature.
  - shape: `(1,)`
  - dynamic_axes: None
  - dtype: `int64`



### XformerEncoder

Xformer encoder supports the following models

| models             |
| ------------------ |
| ConformerEncoder   |
| TransformerEncoder |

**Export configuration**

- `feats_dim` : Dimension of the input feature.
  - type: int
  - default: 80
- `max_seq_len`: Maximum sequence length. 
  - type: int
  - default: 512

**model input**

- `feats` : Input feature of the speech. 
  - shape: `(1, feats_length, feats_dim)`
  - dynamic dimension: 1
  - dtype: `float32`
- `feats_length`: Length of input feature.
  - shape: `(1,)`
  - dynamic_axes: None
  - dtype: `int64`

**model output**

- `encoder_out`: Output feature of encoder.
  - shape: `(1, feats_length, encoder_feats_dim)`
  - dynamic dimension: 1
  - dtype: `float32`
- `encoder_out_lens`: Length of output feature.
  - shape: `(1,)`
  - dynamic_axes: None
  - dtype: `int64`



### Contextual block xformer encoder

ContextualBlockXformer supports the following models.

| models                       |
| ---------------------------- |
| contextual_block_conformer   |
| contextual_block_transformer |

**Export configuration**

- `feats_dim` : Dimension of the input feature.
  - type: int
  - default: 80

**model input**

- `xs_pad` : Input feature of the speech. 

  - shape: `(1, hop_size * subsample + 1, feats_dim)`
  - dynamic dimension: 1
  - dtype: `float32`

- `mask`: Mask for every encoders.

  - shape: `(1, 1, block_size + 2, block_size + 2)`

  - dynamic dimension: 2, 3

  - dtype: `float32`

  - note: This mask should be created with the following process.

    ```python
    mask = np.zeros((1, 1, block_size+2, block_size+2), dtype=np.float32)
    mask[..., 1:, :-1] = 1
    ```

- `buffer_before_downsampling`: Model cache. This will be concatenated before the subsampling.

  - shape: `(1, subsample, feats_dim)`
  - dynamic dimension: 1
  - dtype: `float32`
  - note: subsample is the `subsample` attribute of the model. This argument should be an output of the previous inference, and zeros for the first inference.

- `buffer_after_downsampling`: Model cache. This will be concatenated after the subsampling.

  - shape: `(1, overlap_size, embed_dim)`
  - dynamic dimension: 1
  - dtype: `float32`
  - note: overlap_size is computed as `block_size - hop_size`. `embed_dim` is output dimension of positional encoding. This argument should be an output of the previous inference, and zeros for the first inference.

- `prev_addin`: Addin to append before computation of the encoders. 
  - shape: `(1, 1, embed_dim)`
  - dynamic dimension: None
  - dtype: `float32`
  - note: `embed_dim` is output dimension of positional encoding. This argument should be an output of the previous inference, and zeros for the first inference.

- `pos_enc_xs`: Positional encoding for input feature. (Temporary)
  - shape: `(1, block_size, embed_dim)`
  - dynamic dimension: 1
  - dtype: `float32`
  - note: `embed_dim` is output dimension of positional encoding. I separated positional encoding weights from onnx model because there was an exportation problem. I will fix this in the future, so this argument will be removed in the future.
- `pos_enc_addin`: Positional encoding for input addin. (Temporary)
  - shape: `(1, 1, embed_dim)`
  - dynamic dimension: None
  - dtype: `float32`
  - note: `embed_dim` is output dimension of positional encoding. I separated positional encoding weights from onnx model because there was an exportation problem. I will fix this in the future, so this argument will be removed in the future.

- `past_encoder_ctx`: Previous contexutal vector

  - shape: `(1, n_encoders, h_enc)`
  - dynamic dimension: None
  - dtype: `float32`
  - note: `n_encoders` is the number of `ContextualBlockEncoderLayer` of the model. `h_enc` is the output dimension size of each `ContextualBlockEncoderLayer`.  This argument should be an output of the previous inference, and zeros for the first inference.

- `indicies`: Indicies for some dynamic slice.

  - shape: `(1,)`

  - dynamic dimension: None

  - dtype: `int64`

  - note: This input should be the following array.

    ```python
    indicies = np.array([
        offset,
        offset + hop_size,
        overlap_size
    ])
    ```

    `offset = 0`  for the first inference, and `offset = block_size - look_ahead - hop_size + 1` for the other inference. `overlap_size` is calculated as  `block_size - hop_size`

**model output**

- `ys_pad`: Output of the streaming encoder.
  - shape: `(1, hop_size, encoder_feats_dim)`
  - dynamic dimension: 1
  - dtype: `float32`
- `next_buffer_before_downsampling`: This output will be an input for the next inference as `buffer_before_downsampling`
  - shape: `(1, subsample, feats_dim)`
  - dynamic dimension: 1
  - dtype: `float32`

- `next_buffer_after_downsampling`:  This output will be an input for the next inference as `buffer_after_downsampling`
  - shape: `(1, overlap_size, embed_dim)`
  - dynamic dimension: 1
  - dtype: `float32`

- `next_addin`: This output will be an input for the next inference as `prev_addin`
  - shape: `(1, 1, embed_dim)`
  - dynamic dimension: None
  - dtype: `float32`

- `next_encoder_ctx`: This output will be an input for the next inference as `past_encoder_ctx`
  - shape: `(1, n_encoders, h_enc)`
  - dynamic dimension: None
  - dtype: `float32`



