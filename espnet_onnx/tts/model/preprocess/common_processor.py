import numpy as np

from espnet_onnx.tts.model.preprocess.text_cleaner import TextCleaner


class CommonPreprocessor:
    def __init__(
        self,
        tokenizer,
        token_id_converter,
        cleaner_config,
    ):
        if cleaner_config is None:
            self.text_cleaner = None
        else:
            self.text_cleaner = TextCleaner(cleaner_config.cleaner_types)

        self.tokenizer = tokenizer
        self.token_id_converter = token_id_converter

    def __call__(self, text: str) -> np.ndarray:
        if self.text_cleaner is not None:
            text = self.text_cleaner(text)

        tokens = self.tokenizer.text2tokens(text)
        text_ints = self.token_id_converter.tokens2ids(tokens)
        return np.array(text_ints, dtype=np.int64)
