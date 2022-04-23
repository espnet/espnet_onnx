# Supported Model architecture

**Encoder**

| arch name                          | supported  |
| ---------------------------------- | ---------- |
| ConformerEncoder                   | ◯          |
| ContextualBlockConformerEncoder    | ◯          |
| ContextualBlockTransformerEncoderx | ◯          |
| FairseqHubertEncoder               | x           |
| RNNEncoder                         | ◯          |
| TransformerEncoder                 | ◯          |
| VGGRNNEncoderx                     | ◯          |
| FairSeqWav2Vec2Encoder             | x           |



**Decoder**

| arch name                                  | supported  |
| ------------------------------------------ | ---------- |
| RNNDecoder                                 | ◯          |
| TransformerDecoder                         | ◯          |
| LightweightConvolutionTransformerDecoder   | ◯         |
| LightweightConvolution2DTransformerDecoder | ◯         |
| DynamicConvolutionTransformerDecoder       | x          |
| DynamicConvolution2DTransformerDecoder     | x          |



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
| StreamPositionalEncoding    | ◯          |



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
not supported.

**post encoder**  
not supported.

**transducer**  
not supported.
