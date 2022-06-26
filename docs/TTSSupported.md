# Supported Model architecture

**TTS**

| arch name     | supported |
| ------------- | --------- |
| Tacotron2     | ◯         |
| Transformer   | x         |
| FastSpeech    | x         |
| FastSpeech2   | ◯         |
| VITS          | ◯         |
| JointText2Wav | ◯         |
| JETS          | ◯         |

**Vocoder**

| arch name       | supported |
| --------------- | --------- |
| ParallelWaveGAN | ◯         |
| MelGAN          | ◯         |
| HiFiGAN         | ◯         |
| StyleMelGAN     | x         |

**Attention**

| arch name    | supported |
| ------------ | --------- |
| AttLoc       | ◯         |
| AttForward   | x         |
| AttForwardTA | x         |

**PositionalEmbedding**

| arch name                   | supported |
| --------------------------- | --------- |
| PositionalEncoding          | ◯         |
| ScaledPositionalEncoding    | ◯         |
| LegacyRelPositionalEncoding | ◯         |
| RelPositionalEncoding       | ◯         |
| StreamPositionalEncoding    | ◯         |
