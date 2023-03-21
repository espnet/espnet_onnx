# Supported Model architecture

**Encoder**

| arch name                          | supported |
| ---------------------------------- | --------- |
| ConformerEncoder                   | ◯         |
| ContextualBlockConformerEncoder    | ◯         |
| ContextualBlockTransformerEncoderx | ◯         |
| FairseqHubertEncoder               | x         |
| RNNEncoder                         | ◯         |
| TransformerEncoder                 | ◯         |
| VGGRNNEncoderx                     | ◯         |
| FairSeqWav2Vec2Encoder             | x         |
| FairseqHubertEncoder               | x         |
| FairseqHubertPretrainEncoder       | x         |
| LongformerEncoder                  | x         |

**Decoder**

| arch name                                  | supported |
| ------------------------------------------ | --------- |
| RNNDecoder                                 | ◯         |
| TransformerDecoder                         | ◯         |
| LightweightConvolutionTransformerDecoder   | ◯         |
| LightweightConvolution2DTransformerDecoder | ◯         |
| DynamicConvolutionTransformerDecoder       | x         |
| DynamicConvolution2DTransformerDecoder     | x         |
| TransducerDecoder                          | ◯         |
| MLMDecoder                                 | x         |

**Language Model**

| arch name       | supported |
| --------------- | --------- |
| SequentialRNNLM | ◯         |
| TransformerLM   | ◯         |

**frontend**

| arch name       | supported |
| --------------- | --------- |
| DefaultFrontend | ◯         |
| SlidingWindow   | x         |
| HuBERT          | ◯         |
| Wav2Vec         | x         |
| FusedFrontends  | x         |

**normalize**

| arch name    | supported |
| ------------ | --------- |
| GlobalMVN    | ◯         |
| UtteranceMVN | ◯         |

**PositionalEmbedding**

| arch name                   | supported |
| --------------------------- | --------- |
| PositionalEncoding          | ◯         |
| ScaledPositionalEncoding    | ◯         |
| LegacyRelPositionalEncoding | ◯         |
| RelPositionalEncoding       | ◯         |
| StreamPositionalEncoding    | ◯         |

**Attention**

| arch name               | supported |
| ----------------------- | --------- |
| NoAtt                   | ◯         |
| AttDot                  | ◯         |
| AttAdd                  | ◯         |
| AttLoc                  | ◯         |
| AttCov                  | ◯         |
| AttLoc2D                | x         |
| AttLocRec               | x         |
| AttCovLoc               | ◯         |
| AttMultiHeadDot         | x         |
| AttMultiHeadAdd         | x         |
| AttMultiHeadLoc         | x         |
| AttMultiHeadMultiResLoc | x         |
| AttForward              | x         |
| AttForwardTA            | x         |

**pre encoder**
The following shows the encoder module that support preencoder export and inference.
| arch name       | supported |
| --------------- | --------- |
| RNN             | ◯         |
| Transformer     | ◯         |
| Conformer       | ◯         |
| ContextualBlock | x         |

**post encoder**
not supported.
