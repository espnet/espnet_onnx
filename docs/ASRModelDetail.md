# Details for ASR models and configurations

This document explains the details of each models, such as export configuration, and input/output argument for onnx model.

## Encoder

### RNNEncoder

**Export configuration**

| config name | type | default | detail                         |
| ----------- | ---- | ------- | ------------------------------ |
| feats_dim   | int  | 80      | Dimension of the input feature |

**model input**

| input name | detail                       | shape                          | dtype   | dynamic dim |
| ---------- | ---------------------------- | ------------------------------ | ------- | ----------- |
| feats      | Input feature of the speech. | `(1, feats_length, feats_dim)` | float32 | 1           |

**model output**

| output name      | detail                     | shape                                  | dtype   | dynamic dim |
| ---------------- | -------------------------- | -------------------------------------- | ------- | ----------- |
| encoder_out      | Output feature of encoder. | `(1, feats_length, encoder_feats_dim)` | float32 | 1           |
| encoder_out_lens | Length of output feature.  | `(1,)`                                 | int64   | -           |

### XformerEncoder

Xformer encoder supports the following models

| models             |
| ------------------ |
| ConformerEncoder   |
| TransformerEncoder |

**Export configuration**

| config name | type | default | detail                         |
| ----------- | ---- | ------- | ------------------------------ |
| feats_dim   | int  | 80      | Dimension of the input feature |
| max_seq_len | int  | 512     | Maximum sequence length.       |

**model input**

| input name | detail                       | shape                          | dtype   | dynamic dim |
| ---------- | ---------------------------- | ------------------------------ | ------- | ----------- |
| feats      | Input feature of the speech. | `(1, feats_length, feats_dim)` | float32 | 1           |

**model output**

| output name      | detail                     | shape                                  | dtype   | dynamic dim |
| ---------------- | -------------------------- | -------------------------------------- | ------- | ----------- |
| encoder_out      | Output feature of encoder. | `(1, feats_length, encoder_feats_dim)` | float32 | 1           |
| encoder_out_lens | Length of output feature.  | `(1,)`                                 | int64   | -           |

### Contextual block xformer encoder

ContextualBlockXformer supports the following models.

| models                       |
| ---------------------------- |
| contextual_block_conformer   |
| contextual_block_transformer |

**Export configuration**

| config name | type | default | detail                         |
| ----------- | ---- | ------- | ------------------------------ |
| feats_dim   | int  | 80      | Dimension of the input feature |

**model input**

| input name                 | detail                                                         | shape                                    | dtype   | dynamic dim | previous |
| -------------------------- | -------------------------------------------------------------- | ---------------------------------------- | ------- | ----------- | -------- |
| xs_pad                     | Input feature of the speech.                                   | * Check Note for detail                  | float32 | 1           | -        |
| mask                       | Mask for every encoders.                                       | `(1, 1, block_size + 2, block_size + 2)` | float32 | 2, 3        | -        |
| buffer_before_downsampling | Model cache. This will be concatenated before the subsampling. | `(1, subsample * 2, feats_dim)`          | float32 | 1           | ◯        |
| buffer_after_downsampling  | Model cache. This will be concatenated after the subsampling.  | `(1, overlap_size, embed_dim)`           | float32 | 1           | ◯        |
| prev_addin                 | Addin to append before computation of the encoders.            | `(1, 1, embed_dim)`                      | float32 | -           | ◯        |
| pos_enc_xs                 | Positional encoding for input feature.                         | `(1, block_size, embed_dim)`             | float32 | 1           | -        |
| pos_enc_addin              | Positional encoding for input addin.                           | `(1, 1, embed_dim)`                      | float32 | -           | -        |
| past_encoder_ctx           | Previous contexutal vector                                     | `(1, n_encoders, h_enc)`                 | float32 | -           | ◯        |
| is_first                   | Flag to check if the first iteration                           | `(1,)`                                   | int64   | -           | -        |

Note:

- `mask` should be created with the following process.

```python
mask = np.zeros((1, 1, block_size+2, block_size+2), dtype=np.float32)
mask[..., 1:, :-1] = 1
```

- Arguments with `previous == ◯ ` should be an output of the previous inference, and zeros for the first inference.

- `overlap_size` is computed as `block_size - hop_size`.

- `embed_dim` is output dimension of positional encoding.

- `is_first` is 1 for the first iteration, and 0 for the second and later iterations

- The size of `xs_pad` should should be the following:
  - First iteration: `(1, (block_size + 2) * subsample, feats_dim)`
  - Second or later iteration: `(1, hop_size * subsample, feats_dim)`

**model output**

| input name                      | detail                                                                              | shape                              | dtype   | dynamic dim | next |
| ------------------------------- | ----------------------------------------------------------------------------------- | ---------------------------------- | ------- | ----------- | ---- |
| ys_pad                          | Output of the streaming encoder.                                                    | `(1, hop_size, encoder_feats_dim)` | float32 | 1           | -    |
| next_buffer_before_downsampling | This output will be an input for the next inference as `buffer_before_downsampling` | `(1, subsample * 2, feats_dim)`    | float32 | 1           | ◯    |
| next_buffer_after_downsampling  | This output will be an input for the next inference as `buffer_after_downsampling`  | `(1, overlap_size, embed_dim)`     | float32 | 1           | ◯    |
| next_addin                      | This output will be an input for the next inference as `prev_addin`                 | `(1, 1, embed_dim)`                | float32 | -           | ◯    |
| next_encoder_ctx                | This output will be an input for the next inference as `past_encoder_ctx`           | `(1, n_encoders, h_enc)`           | float32 | -           | ◯    |

## Decoder

### RNNDecoder

**Export configuration**

There is no configuration available.

**model input**

| input name   | detail                                                       | shape                              | dtype   | dynamic dim |
| ------------ | ------------------------------------------------------------ | ---------------------------------- | ------- | ----------- |
| vy           | Input sequence                                               | `(1, seq_len)`                     | int64   | 1           |
| z\_prev\_{i} | List of caches. The length equals to number of decoders.     | List[`(1, hidden_size)`]           | float32 | -           |
| c\_prev\_{i} | List of caches. The length equals to number of decoders.     | List[`(1, hidden_size)`]           | float32 | -           |
| a\_prev\_{i} | List of caches. The length equals to number of attentions.   | \*1                                | float32 | \*1         |
| pceh\_{i}    | List of caches. pceh stands for `pre_computed_enc_h`         | \*2                                | float32 | 1           |
| enc\_h\_{i}  | List of caches. The length equals to number of attentions.   | `(1, feat_length, enc_size)`       | float32 | 1           |
| mask\_{i}    | List of mask. The length equals to number of attentions. \*3 | List[`(feat_length, feat_length)`] | float32 | 0, 1        |

- \*1: The shape and dynamic axes of `a_prev_{i}` depends on the attention type.

  | Attention type                     | shape                 | dynamic axes |
  | ---------------------------------- | --------------------- | ------------ |
  | `coverage`, or `coverage_location` | `(1, 1, feat_length)` | 2            |
  | others                             | `(1,feat_length)`     | 1            |

- \*2: The shape pf `pceh_{i}` depends on the attention type.

  | Attention type | shape                           |
  | -------------- | ------------------------------- |
  | NoAtt          | `(1, 1, 1)`                     |
  | others         | `(1, feat_length, out_feature)` |

  where `out_features` equals to `mlp_enc.out_features` of each attention in `att_list`.

- \*3: Each `mask` should be created as follows:

  ```python
  from espnet_onnx.utils.function import make_pad_mask

  mask = make_pad_mask([feat_length]) * -float('inf')
  ```

**model output**

| output name  | detail                                                    | shape                                  | dtype   | dynamic dim |
| ------------ | --------------------------------------------------------- | -------------------------------------- | ------- | ----------- |
| logp         | Output feature of decoder.                                | `(1, feats_length, decoder_feats_dim)` | float32 | 1           |
| c\_list\_{i} | This argument should be an input of the next `c_prev_{i}` | List[`(1, hidden_size)`]               | float32 | -           |
| z\_list\_{i} | This argument should be an input of the next `z_prev_{i}` | List[`(1, hidden_size)`]               | float32 | -           |
| att_w \*1    | This argument should be an input of the next `a_prev_{i}` | \*1                                    | float32 | -           |
| att\_w\_{i}  | This argument should be an input of the next `a_prev_{i}` | \*1                                    | float32 | -           |

- \*1: When `num_enc == 1`, then output name is `att_w`, otherwise `att_w_{i}`. The shape is as the same with model input.

### XformerDecoder

Xformer decoder supports the following models

| models             |
| ------------------ |
| TransformerEncoder |

**Export configuration**

| config name | type | default | detail                   |
| ----------- | ---- | ------- | ------------------------ |
| max_seq_len | int  | 512     | Maximum sequence length. |

**model input**

| input name | detail                                                                        | shape                             | dtype   | dynamic dim |
| ---------- | ----------------------------------------------------------------------------- | --------------------------------- | ------- | ----------- |
| tgt        | Input token ids                                                               | `(batch, maxlen_out)`             | int64   | 0, 1        |
| memory     | encoded memory                                                                | `(batch, maxlen_in, feat)`        | float32 | 0, 1        |
| cache      | List of cached outputs. The length of list is the same as number of decoders. | List[`(1, max_time_out-1, size)`] | float32 | 0, 1        |

**model output**

| output name  | detail                                                                                                                | shape                                      | dtype   | dynamic dim |
| ------------ | --------------------------------------------------------------------------------------------------------------------- | ------------------------------------------ | ------- | ----------- |
| y            | Output feature of decoder.                                                                                            | `(batch, feats_length, decoder_feats_dim)` | float32 | 0, 1        |
| out_cache{i} | List of caches. The length of list is the same as number of decoders. This argument should be inputs for next `cache` | List[`(1, max_time_out-1, size)`]          | float32 | 0, 1        |

### TransducerDecoder

**Export configuration**

| config name | type | default | detail                   |
| ----------- | ---- | ------- | ------------------------ |
| max_seq_len | int  | 512     | Maximum sequence length. |

**model input**

| input name | detail             | shape                  | dtype   | dynamic dim |
| ---------- | ------------------ | ---------------------- | ------- | ----------- |
| labels     | Label ID sequences | `(batch, seq_len)`     | int64   | 0, 1        |
| h_cache    | Cache for RNN      | `(dlayers, 1, dunits)` | float32 | 1           |
| c_cache    | Cache for RNN      | `(dlayers, 1, dunits)` | float32 | 1           |

**model output**

| output name | detail                                                               | shape                                 | dtype   | dynamic dim |
| ----------- | -------------------------------------------------------------------- | ------------------------------------- | ------- | ----------- |
| sequence    | Output sequence of decoder.                                          | `(batch, seq_len, decoder_feats_dim)` | float32 | -           |
| out_h_cache | List of rnn cache. This argument should be inputs for next `h_cache` | `(dlayers, 1, dunits)`                | float32 | 1           |
| out_c_cache | List of rnn cache. This argument should be inputs for next `c_cache` | `(dlayers, 1, dunits)`                | float32 | 1           |

## Language models

### SeqentialRNNLM

**Export configuration**

There is no available configuration.

**model input**

| input name | detail                                           | shape                  | dtype   | dynamic dim |
| ---------- | ------------------------------------------------ | ---------------------- | ------- | ----------- |
| x          | Label ID sequences                               | `(batch, seq_len)`     | int64   | 0, 1        |
| in_hidden1 | Cache for RNN                                    | `(dlayers, 1, dunits)` | float32 | 1           |
| in_hidden2 | Cache for RNN. Not required if rnn_type is lstm. | `(dlayers, 1, dunits)` | float32 | 1           |

**model output**

| output name | detail                                                                  | shape                                 | dtype   | dynamic dim |
| ----------- | ----------------------------------------------------------------------- | ------------------------------------- | ------- | ----------- |
| y           | Output sequence of decoder.                                             | `(batch, seq_len, decoder_feats_dim)` | float32 | -           |
| out_hidden1 | List of rnn cache. This argument should be inputs for next `in_hidden1` | `(dlayers, 1, dunits)`                | float32 | 1           |
| out_hidden2 | List of rnn cache. This argument should be inputs for next `in_hidden2` | `(dlayers, 1, dunits)`                | float32 | 1           |

### TransformerLM

**Export configuration**

| config name | type | default | detail                   |
| ----------- | ---- | ------- | ------------------------ |
| max_seq_len | int  | 512     | Maximum sequence length. |

**model input**

| input name | detail                                                                  | shape                   | dtype   | dynamic dim |
| ---------- | ----------------------------------------------------------------------- | ----------------------- | ------- | ----------- |
| tgt        | Label ID sequences                                                      | `(batch, seq_len)`      | int64   | 0, 1        |
| cache\_{i} | Cache for encoder. The length of list is same as the number of encoders | `(batch, 1, enc_feats)` | float32 | 0, 1        |

**model output**

| output name | detail                                                                  | shape                                 | dtype   | dynamic dim |
| ----------- | ----------------------------------------------------------------------- | ------------------------------------- | ------- | ----------- |
| y           | Output sequence of decoder.                                             | `(batch, seq_len, decoder_feats_dim)` | float32 | -           |
| cache\_{i}  | Cache for encoder. The length of list is same as the number of encoders | `(batch, 1, enc_feats)`               | float32 | 0, 1        |
