# Available export configurations

**Encoder**

The following models can set `max_seq_len`.
Default is 512.

| arch name          | supported |
| ------------------ | --------- |
| ConformerEncoder   | ◯         |
| TransformerEncoder | ◯         |

**Decoder**

The following models can set `max_seq_len`.
Default is 512.

| arch name                                  | supported |
| ------------------------------------------ | --------- |
| TransformerDecoder                         | ◯         |
| LightweightConvolutionTransformerDecoder   | ◯         |
| LightweightConvolution2DTransformerDecoder | ◯         |
| TransducerDecoder                          | ◯         |

**Language Model**

The following models can set `max_seq_len`.
Default is 512.

| arch name     | supported |
| ------------- | --------- |
| TransformerLM | ◯         |
