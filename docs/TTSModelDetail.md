# Details for TTS models and configurations

This document explains the details of each models, such as export configuration, and input/output argument for onnx model.

## TTS models

### VITS

**Export configuration**

| config key          | type  | note                                                      | default |
| ------------------- | ----- | --------------------------------------------------------- | ------- |
| max_seq_len         | int   | Maximum sequence length.                                  | 512     |
| noise_scale         | float | Noise scale parameter for flow.                           | 0.667   |
| noise_scale_dur     | float | Noise scale parameter for duration predictor.             | 0.8     |
| alpha               | float | Alpha parameter to control the speed of generated speech. | 1.0     |
| use_teacher_forcing | bool  | Whether to use teacher forcing.                           | False   |
| predict_duration    | bool  | Whether to predict duration while inference.              | True    |

**model input**

| input name   | detail                                                                                          | shape                      | dtype   | dynamic dim |
| ------------ | ----------------------------------------------------------------------------------------------- | -------------------------- | ------- | ----------- |
| text         | Input text token ids.                                                                           | `(1,)`                     | int64   | 0           |
| feats        | Feature vector. Required if `use_teacher_forcing` is True.                                      | `(feats_length, feat_dim)` | float32 | 0           |
| sids         | Speaker id. Required if exported model requires speaker id.                                     | `(1,)`                     | int64   | -           |
| spembs       | Speaker vector. Required if exported model requires speaker embedding.                          | `(spk_embed_dim,)`         | float32 | -           |
| lids         | Language id. Required if exported model required language id.                                   | `(1,)`                     | int64   | -           |
| duration     | Ground-truth duration tensor. Required if `predict_duration` is False when exporting the model. | `(len_text,)`              | float32 | 0           |

**model output**

| output name | detail                            | shape                   | dtype   | dynamic dim |
| ----------- | --------------------------------- | ----------------------- | ------- | ----------- |
| wav         | Generated waveform tensor         | `(len_wav,)`            | float32 | 0           |
| att_w       | Monotonic attention weight tensor | `(feats_len, len_text)` | float32 | 0, 1        |
| dur         | Predicted duration tensor         | `(len_text,)`           | float32 | 0           |
