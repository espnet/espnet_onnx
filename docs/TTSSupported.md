# Supported Model architecture

**TTS**

| arch name     | supported |
| ------------- | --------- |
| Tacotron2     | x         |
| Transformer   | x         |
| FastSpeech    | x         |
| FastSpeech2   | x         |
| VITS          |  ◯         |
| JointText2Wav | x         |

**Vocoder**

| arch name                                  | supported |
| ------------------------------------------ | --------- |
| ParallelWaveGAN                            | x         |

**feats_extractl**

| arch name       | supported |
| --------------- | --------- |
| LogMelFbank | x         |
| LogSpectrogram   | x        |
| LinearSpectrogram   | x         |

**pitch_extract**

| arch name       | supported |
| --------------- | --------- |
| Dio | x         |

**energy_extract**

| arch name    | supported |
| ------------ | --------- |
| Energy    | x         |

**PositionalEmbedding**

| arch name                   | supported |
| --------------------------- | --------- |
| PositionalEncoding          | ◯         |
| ScaledPositionalEncoding    | ◯         |
| LegacyRelPositionalEncoding | ◯         |
| RelPositionalEncoding       | ◯         |
| StreamPositionalEncoding    | ◯         |

**normalize**

| arch name               | supported |
| ----------------------- | --------- |
| GlobalMVN                   | ◯         |

