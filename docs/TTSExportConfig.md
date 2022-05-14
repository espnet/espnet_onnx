# Available export configurations

**VITS**

`VITS` model supports the following export configurations.

| config key          | type | note                                                      | default |
| ------------------- | ---- | --------------------------------------------------------- | ------- |
| max_seq_len         | ◯    | Maximum sequence length.                                  | 512     |
| noise_scale         | ◯    | Noise scale parameter for flow.                           | 0.667   |
| noise_scale_dur     | ◯    | Noise scale parameter for duration predictor.             | 0.8     |
| alpha               | ◯    | Alpha parameter to control the speed of generated speech. | 1.0     |
| use_teacher_forcing | ◯    | Whether to use teacher forcing.                           | False   |
| predict_duration    | ◯    | Whether to predict duration while inference.              | False   |
