
from typing import Union
from pathlib import Path
from typeguard import check_argument_types

import numpy as np
import librosa

from espnet_onnx.asr.model.encoder import Encoder
from espnet_onnx.asr.model.decoder import get_decoder
from espnet_onnx.asr.model.lm import LM
from espnet_onnx.asr.scorers import CTCPrefixScorer


class Speech2Text:
    """Wrapper class for espnet2.asr.bin.asr_infer.Speech2Text
    
    """
    def __init__(self,
        model_dir: Union[Path, str] = None,
        quantized: bool = False
        ):
        assert check_argument_types()
        
        # 1. Build asr model
        config = get_config(os.path.join(model_dir, 'config.json'))
        self.encoder = Encoder(config.encoder)
        decoder = get_decoder(config.decoder, config.token)
        ctc = CTCPrefixScorer(config.ctc.model_path, config.token.eos)
        
        scorers = {}
        scorers.update(
            decoder=decoder,
            ctc=ctc,
            length_bonus=LengthBonus(len(config.token.list))
        )
        
        # 2. Build lm model
        if config.use_lm:
            scorers.update(
                lm=LM(config.lm, config.token)
            )
        
        # 3. Build ngram model
        if config.ngram.use_ngram:
            if config.ngram.scorer_type == 'full':
                scorers.update(
                    ngram=NgramFullScorer(
                        config.ngram.ngram_file,
                        config.token.list
                    )
                )
            else:
                scorers.update(
                    ngram=NgramPartialScorer(
                        config.ngram.ngram_file,
                        config.token.list
                    )
                )
        
        # 4. Build beam search object
        weights = dict(
            decoder=1.0 - config.weights.ctc_weight,
            ctc=config.weights.ctc_weight,
            lm=config.weights.lm_weight,
            length_bonus=config.weights.penalty,
        )
        if config.use_transducer:
            self.beam_search = BSTransducer(
                config.beam_search,
                config.transducer,
                config.token,
                decoder=decoder,
                # lm=scorers["lm"] if "lm" in scorers else None,
                weights=weights
            )
        else:
            self.beam_search = BeamSearch(
                config.beam_search,
                config.token,
                scorers=scorers,
                weights=weights,
            )
        
        # 5. Build text converter
        self.tokenizer = Tokenizer(config.token, config.bpemodel)
        self.converter = TokenIdConverter(token_list=config.token.list)
        
        self.config = config
    
    def __call__(self, speech: np.ndarray) -> List[
        Tuple[
            Optional[str],
            List[str],
            List[int],
            Union[Hypothesis, ExtTransHypothesis, TransHypothesis],
        ]
    ]:
        """Inference
        Args:
            data: Input speech data
        Returns:
            text, token, token_int, hyp
        """
        assert check_argument_types()
        
        # check dtype
        if speech.dtype != np.float32:
            speech = speech.astype(np.float32)
            
        # data: (Nsamples,) -> (1, Nsamples)
        speech = speech[np.newaxis, :]
        # lengths: (1,)
        lengths = np.array([speech.shape[1]]).astype(np.int64)
        
        # b. Forward Encoder
        enc, _ = self.encoder(speech=speech, speech_length=lengths)
        if isinstance(enc, tuple):
            enc = enc[0]
        assert len(enc) == 1, len(enc)
        
        nbest_hyps = self.beam_search(enc[0])[: self.nbest]
        
        results = []
        for hyp in nbest_hyps:
            # remove sos/eos and get results
            last_pos = -1
            if isinstance(hyp.yseq, list):
                token_int = hyp.yseq[1:last_pos].astype(np.int64)
            else:
                token_int = hyp.yseq[1:last_pos].astype(np.int64).tolist()

            # remove blank symbol id, which is assumed to be 0
            token_int = list(filter(lambda x: x != 0, token_int))

            # Change integer-ids to tokens
            token = self.converter.ids2tokens(token_int)
            token = token[1:] # since I add 'blank' before sos for onnx computing

            if self.tokenizer is not None:
                text = self.tokenizer.tokens2text(token)
            else:
                text = None
            results.append((text, token, token_int, hyp))

        return results
        
    def from_wav(self, audio_path: Union[Path, str]):
        assert check_argument_types()
        y, sr = librosa.load(audio_path, 16000)
        y = librosa.util.normalize(y)
        sound_idx = librosa.effects.split(y)
        text = ""
        for sidx in sound_idx:
            text += " " + self(y[sidx[0]:sidx[1]])[0][0]
        return text.capitalize()
    
    
    