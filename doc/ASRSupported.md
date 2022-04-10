# Supported Model architecture

**Encoder**

| arch name                          | supported  |
| ---------------------------------- | ---------- |
| ConformerEncoder                   | ◯          |
| ContextualBlockConformerEncoder    | x          |
| ContextualBlockTransformerEncoderx | x          |
| FairseqHubertEncoder               | not tested |
| RNNEncoder                         | ◯          |
| TransformerEncoder                 | ◯          |
| VGGRNNEncoderx                     | ◯          |
| FairSeqWav2Vec2Encoder             | not tested |



**Decoder**

| arch name                                  | supported  |
| ------------------------------------------ | ---------- |
| RNNDecoder                                 | ◯          |
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



**PositionalEmbedding**

| arch name                   | supported  |
| --------------------------- | ---------- |
| PositionalEncoding          | ◯          |
| ScaledPositionalEncoding    | ◯          |
| LegacyRelPositionalEncoding | ◯          |
| RelPositionalEncoding       | ◯         |
| StreamPositionalEncoding    | x          |



**Attention**

| arch name               | supported |
| ----------------------- | --------- |
| NoAtt                   | x         |
| AttDot                  | x         |
| AttAdd                  | x         |
| AttLoc                  | ◯         |
| AttCov                  | x         |
| AttLoc2D                | x         |
| AttLocRec               | x         |
| AttCovLoc               | x         |
| AttMultiHeadDot         | x         |
| AttMultiHeadAdd         | x         |
| AttMultiHeadLoc         | x         |
| AttMultiHeadMultiResLoc | x         |
| AttForward              | x         |
| AttForwardTA            | x         |



**pre encoder**  
not supported.

**post encoder**  
not supported.

**transducer**  
not supported.