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
| ParallelWaveGAN | yes       |
| MelGAN          | yes       |
| HiFiGAN         | yes       |
| StyleMelGAN     | no        |

**Attention**

| arch name    | supported |
| ------------ | --------- |
| AttLoc       | yes       |
| AttForward   | no        |
| AttForwardTA | no        |

**PositionalEmbedding**

| arch name                   | supported |
| --------------------------- | --------- |
| PositionalEncoding          | yes       |
| ScaledPositionalEncoding    | yes       |
| LegacyRelPositionalEncoding | yes       |
| RelPositionalEncoding       | yes       |
| StreamPositionalEncoding    | yes       |
