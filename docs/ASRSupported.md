# Supported Model architecture

**Encoder**

| arch name                          | supported |
| ---------------------------------- | --------- |
| ConformerEncoder                   | Yes       |
| ContextualBlockConformerEncoder    | Yes       |
| ContextualBlockTransformerEncoderx | Yes       |
| FairseqHubertEncoder               | No        |
| RNNEncoder                         | Yes       |
| TransformerEncoder                 | Yes       |
| VGGRNNEncoderx                     | Yes       |
| FairSeqWav2Vec2Encoder             | No        |
| FairseqHubertEncoder               | No        |
| FairseqHubertPretrainEncoder       | No        |
| LongformerEncoder                  | No        |
| BranchformerEncoder                | Yes       |
| E-BranchformerEncoder              | Yes       |


**Decoder**

| arch name                                  | supported |
| ------------------------------------------ | --------- |
| RNNDecoder                                 | Yes       |
| TransformerDecoder                         | Yes       |
| LightweightConvolutionTransformerDecoder   | Yes       |
| LightweightConvolution2DTransformerDecoder | Yes       |
| DynamicConvolutionTransformerDecoder       | No        |
| DynamicConvolution2DTransformerDecoder     | No        |
| TransducerDecoder                          | Yes       |
| MLMDecoder                                 | No        |

**Language Model**

| arch name       | supported |
| --------------- | --------- |
| SequentialRNNLM | Yes       |
| TransformerLM   | Yes       |

**frontend**

| arch name       | supported |
| --------------- | --------- |
| DefaultFrontend | Yes       |
| SlidingWindow   | No        |
| HuBERT          | Yes       |
| Wav2Vec         | No        |
| FusedFrontends  | No        |
| No Frontend     | Yes       |

**normalize**

| arch name    | supported |
| ------------ | --------- |
| GlobalMVN    | Yes       |
| UtteranceMVN | Yes       |

**PositionalEmbedding**

| arch name                   | supported |
| --------------------------- | --------- |
| PositionalEncoding          | Yes       |
| ScaledPositionalEncoding    | Yes       |
| LegacyRelPositionalEncoding | Yes       |
| RelPositionalEncoding       | Yes       |
| StreamPositionalEncoding    | Yes       |

**Attention**

| arch name               | supported |
| ----------------------- | --------- |
| NoAtt                   | Yes       |
| AttDot                  | Yes       |
| AttAdd                  | Yes       |
| AttLoc                  | Yes       |
| AttCov                  | Yes       |
| AttLoc2D                | No        |
| AttLocRec               | No        |
| AttCovLoc               | Yes       |
| AttMultiHeadDot         | No        |
| AttMultiHeadAdd         | No        |
| AttMultiHeadLoc         | No        |
| AttMultiHeadMultiResLoc | No        |
| AttForward              | No        |
| AttForwardTA            | No        |

**pre encoder**
The following shows the encoder module that support preencoder export and inference.
| arch name       | supported |
| --------------- | --------- |
| RNN             | Yes       |
| Transformer     | Yes       |
| Conformer       | Yes       |
| ContextualBlock | No        |

**post encoder**
not supported.
