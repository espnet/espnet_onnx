"""Beam search module."""
import logging
from itertools import chain
from typing import Any, Dict, List, Tuple

import numpy as np
from typeguard import check_argument_types

from espnet_onnx.asr.beam_search.hyps import Hypothesis
from espnet_onnx.asr.scorer.interface import (PartialScorerInterface,
                                              ScorerInterface)
from espnet_onnx.utils.config import Config
from espnet_onnx.utils.function import end_detect, topk


class BeamSearch:
    """Beam search implementation."""

    def __init__(
        self,
        bs_config: Config,
        token_config: Config,
        scorers: Dict[str, ScorerInterface],
        weights: Dict[str, float],
    ):
        """Initialize beam search.
        Args:
            scorers (dict[str, ScorerInterface]): Dict of decoder modules
                e.g., Decoder, CTCPrefixScorer, LM
                The scorer will be ignored if it is `None`
            weights (dict[str, float]): Dict of weights for each scorers
                The scorer will be ignored if its weight is 0
            beam_size (int): The number of hypotheses kept during search
            vocab_size (int): The number of vocabulary
            sos (int): Start of sequence id
            eos (int): End of sequence id
            token_list (list[str]): List of tokens for debug log
            pre_beam_score_key (str): key of scores to perform pre-beam search
            pre_beam_ratio (float): beam size in the pre-beam search
                will be `int(pre_beam_ratio * beam_size)`
        """
        assert check_argument_types()

        # set scorers
        self.weights = weights
        self.scorers = dict()
        self.full_scorers = dict()
        self.part_scorers = dict()
        # this module dict is required for recursive cast
        # `self.to(device, dtype)` in `recog.py`
        # self.nn_dict = {}
        for k, v in scorers.items():
            w = weights.get(k, 0)
            if w == 0 or v is None:
                continue

            assert isinstance(
                v, ScorerInterface
            ), f"{k} ({type(v)}) does not implement ScorerInterface"

            self.scorers[k] = v
            if isinstance(v, PartialScorerInterface):
                self.part_scorers[k] = v
            else:
                self.full_scorers[k] = v

        # set configurations
        self.sos = token_config.sos
        self.eos = token_config.eos
        self.token_list = token_config.list
        self.pre_beam_size = int(bs_config.pre_beam_ratio * bs_config.beam_size)
        self.beam_size = bs_config.beam_size
        self.n_vocab = len(self.token_list)
        if (
            bs_config.pre_beam_score_key is not None
            and bs_config.pre_beam_score_key != "full"
            and bs_config.pre_beam_score_key not in self.full_scorers
        ):
            raise KeyError(
                f"{bs_config.pre_beam_score_key} is not found in {self.full_scorers}"
            )

        self.pre_beam_score_key = bs_config.pre_beam_score_key
        self.do_pre_beam = (
            self.pre_beam_score_key is not None
            and self.pre_beam_size < self.n_vocab
            and len(self.part_scorers) > 0
        )
        # maxlenratio (float): Input length ratio to obtain max output length.
        #     If maxlenratio=0.0 (default), it uses a end-detect function
        #     to automatically find maximum hypothesis lengths
        #     If maxlenratio<0.0, its absolute value is interpreted
        #     as a constant max output length.
        # minlenratio (float): Input length ratio to obtain min output length.
        self.minlenratio = bs_config.minlenratio
        self.maxlenratio = bs_config.maxlenratio

    def init_hyp(self, x: np.ndarray) -> List[Hypothesis]:
        """Get an initial hypothesis data.
        Args:
            x (np.ndarray): The encoder output feature
        Returns:
            Hypothesis: The initial hypothesis.
        """
        init_states = dict()
        init_scores = dict()
        for k, d in self.scorers.items():
            init_states[k] = d.init_state(x)
            init_scores[k] = 0.0
        return [
            Hypothesis(
                score=0.0,
                scores=init_scores,
                states=init_states,
                # Add blank token before sos for onnx inference
                yseq=np.array([self.sos]),
            )
        ]

    @staticmethod
    def append_token(xs: np.ndarray, x: int) -> np.ndarray:
        """Append new token to prefix tokens.
        Args:
            xs (np.ndarray): The prefix token
            x (int): The new token to append
        Returns:
            np.ndarray: New tensor contains: xs + [x] with xs.dtype and xs.device
        """
        x = np.array([x], dtype=xs.dtype)
        return np.concatenate([xs, x])

    def score_full(
        self, hyp: Hypothesis, x: np.ndarray
    ) -> Tuple[Dict[str, np.ndarray], Dict[str, Any]]:
        """Score new hypothesis by `self.full_scorers`.
        Args:
            hyp (Hypothesis): Hypothesis with prefix tokens to score
            x (np.ndarray): Corresponding input feature
        Returns:
            Tuple[Dict[str, np.ndarray], Dict[str, Any]]: Tuple of
                score dict of `hyp` that has string keys of `self.full_scorers`
                and tensor score values of shape: `(self.n_vocab,)`,
                and state dict that has string keys
                and state values of `self.full_scorers`
        """
        scores = dict()
        states = dict()
        for k, d in self.full_scorers.items():
            scores[k], states[k] = d.score(hyp.yseq, hyp.states[k], x)
        return scores, states

    def score_partial(
        self, hyp: Hypothesis, ids: np.ndarray, x: np.ndarray
    ) -> Tuple[Dict[str, np.ndarray], Dict[str, Any]]:
        """Score new hypothesis by `self.part_scorers`.
        Args:
            hyp (Hypothesis): Hypothesis with prefix tokens to score
            ids (np.ndarray): 1D tensor of new partial tokens to score
            x (np.ndarray): Corresponding input feature
        Returns:
            Tuple[Dict[str, np.ndarray], Dict[str, Any]]: Tuple of
                score dict of `hyp` that has string keys of `self.part_scorers`
                and tensor score values of shape: `(len(ids),)`,
                and state dict that has string keys
                and state values of `self.part_scorers`
        """
        scores = dict()
        states = dict()
        for k, d in self.part_scorers.items():
            scores[k], states[k] = d.score_partial(hyp.yseq, ids, hyp.states[k], x)
        return scores, states

    def beam(
        self, weighted_scores: np.ndarray, ids: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Compute topk full token ids and partial token ids.
        Args:
            weighted_scores (np.ndarray): The weighted sum scores for each tokens.
            Its shape is `(self.n_vocab,)`.
            ids (np.ndarray): The partial token ids to compute topk
        Returns:
            Tuple[np.ndarray, np.ndarray]:
                The topk full token ids and partial token ids.
                Their shapes are `(self.beam_size,)`
        """
        # no pre beam performed
        if weighted_scores.shape[0] == ids.shape[0]:
            top_ids = topk(weighted_scores, self.beam_size)
            return top_ids, top_ids

        # mask pruned in pre-beam not to select in topk
        tmp = weighted_scores[ids]
        weighted_scores[:] = -float("inf")
        weighted_scores[ids] = tmp
        top_ids = topk(weighted_scores, self.beam_size)
        local_ids = topk(weighted_scores[ids], self.beam_size)
        return top_ids, local_ids

    @staticmethod
    def merge_scores(
        prev_scores: Dict[str, float],
        next_full_scores: Dict[str, np.ndarray],
        full_idx: int,
        next_part_scores: Dict[str, np.ndarray],
        part_idx: int,
    ) -> Dict[str, np.ndarray]:
        """Merge scores for new hypothesis.
        Args:
            prev_scores (Dict[str, float]):
                The previous hypothesis scores by `self.scorers`
            next_full_scores (Dict[str, np.ndarray]): scores by `self.full_scorers`
            full_idx (int): The next token id for `next_full_scores`
            next_part_scores (Dict[str, np.ndarray]):
                scores of partial tokens by `self.part_scorers`
            part_idx (int): The new token id for `next_part_scores`
        Returns:
            Dict[str, np.ndarray]: The new score dict.
                Its keys are names of `self.full_scorers` and `self.part_scorers`.
                Its values are scalar tensors by the scorers.
        """
        new_scores = dict()
        for k, v in next_full_scores.items():
            new_scores[k] = prev_scores[k] + v[full_idx]
        for k, v in next_part_scores.items():
            new_scores[k] = prev_scores[k] + v[part_idx]
        return new_scores

    def merge_states(self, states: Any, part_states: Any, part_idx: int) -> Any:
        """Merge states for new hypothesis.
        Args:
            states: states of `self.full_scorers`
            part_states: states of `self.part_scorers`
            part_idx (int): The new token id for `part_scores`
        Returns:
            Dict[str, np.ndarray]: The new score dict.
                Its keys are names of `self.full_scorers` and `self.part_scorers`.
                Its values are states of the scorers.
        """
        new_states = dict()
        for k, v in states.items():
            new_states[k] = v
        for k, d in self.part_scorers.items():
            new_states[k] = d.select_state(part_states[k], part_idx)
        return new_states

    def search(self, running_hyps: List[Hypothesis], x: np.ndarray) -> List[Hypothesis]:
        """Search new tokens for running hypotheses and encoded speech x.
        Args:
            running_hyps (List[Hypothesis]): Running hypotheses on beam
            x (np.ndarray): Encoded speech feature (T, D)
        Returns:
            List[Hypotheses]: Best sorted hypotheses
        """
        best_hyps = []
        part_ids = np.arange(self.n_vocab)  # no pre-beam
        for hyp in running_hyps:
            # scoring
            weighted_scores = np.zeros(self.n_vocab, dtype=x.dtype)
            scores, states = self.score_full(hyp, x)
            for k in self.full_scorers:
                weighted_scores += self.weights[k] * scores[k]
            # partial scoring
            if self.do_pre_beam:
                pre_beam_scores = (
                    weighted_scores
                    if self.pre_beam_score_key == "full"
                    else scores[self.pre_beam_score_key]
                )
                part_ids = topk(pre_beam_scores, self.pre_beam_size)
            part_scores, part_states = self.score_partial(hyp, part_ids, x)
            for k in self.part_scorers:
                weighted_scores[part_ids] += self.weights[k] * part_scores[k]
            # add previous hyp score
            weighted_scores += hyp.score

            # update hyps
            for j, part_j in zip(*self.beam(weighted_scores, part_ids)):
                # will be (2 x beam at most)
                best_hyps.append(
                    Hypothesis(
                        score=weighted_scores[j],
                        yseq=self.append_token(hyp.yseq, j),
                        scores=self.merge_scores(
                            hyp.scores, scores, j, part_scores, part_j
                        ),
                        states=self.merge_states(states, part_states, part_j),
                    )
                )
            # sort and prune 2 x beam -> beam
            best_hyps = sorted(best_hyps, key=lambda x: x.score, reverse=True)[
                : min(len(best_hyps), self.beam_size)
            ]
        return best_hyps

    def __call__(self, x: np.ndarray) -> List[Hypothesis]:
        """Perform beam search.
        Args:
            x (np.ndarray): Encoded speech feature (T, D)
        Returns:
            list[Hypothesis]: N-best decoding results
        """
        # set length bounds
        if self.maxlenratio == 0:
            maxlen = x.shape[0]
        elif self.maxlenratio < 0:
            maxlen = -1 * int(self.maxlenratio)
        else:
            maxlen = max(1, int(self.maxlenratio * x.shape[0]))
        minlen = int(self.minlenratio * x.shape[0])
        logging.debug("decoder input length: " + str(x.shape[0]))
        logging.debug("max output length: " + str(maxlen))
        logging.debug("min output length: " + str(minlen))

        # main loop of prefix search
        running_hyps = self.init_hyp(x)
        ended_hyps = []
        for i in range(maxlen):
            logging.debug("position " + str(i))
            best = self.search(running_hyps, x)
            # post process of one iteration
            running_hyps = self.post_process(i, maxlen, best, ended_hyps)
            # end detection
            if self.maxlenratio == 0.0 and end_detect(
                [h.asdict() for h in ended_hyps], i
            ):
                logging.debug(f"End detected at {i}")
                break
            if len(running_hyps) == 0:
                logging.debug("No hypothesis. Finish decoding.")
                break
            else:
                logging.debug(f"Remained hypotheses: {len(running_hyps)}")

        nbest_hyps = sorted(ended_hyps, key=lambda x: x.score, reverse=True)
        # check the number of hypotheses reaching to eos
        if len(nbest_hyps) == 0:
            logging.warning(
                "There is no N-best results, perform recognition "
                "again with smaller minlenratio."
            )
            return (
                []
                if minlenratio < 0.1
                else self(x, self.maxlenratio, max(0.0, minlenratio - 0.1))
            )

        # report the best result
        best = nbest_hyps[0]
        for k, v in best.scores.items():
            logging.debug(
                f"{v:6.2f} * {self.weights[k]:3} = {v * self.weights[k]:6.2f} for {k}"
            )
        logging.debug(f"Total log probability: {best.score:.2f}")
        logging.debug(f"Normalized log probability: {best.score / len(best.yseq):.2f}")
        logging.debug(f"Total number of ended hypotheses: {len(nbest_hyps)}")
        if self.token_list is not None:
            logging.debug(
                "Best hypo: "
                + "".join([self.token_list[int(x)] for x in best.yseq[1:-1]])
                + "\n"
            )
        return nbest_hyps

    def post_process(
        self,
        i: int,
        maxlen: int,
        running_hyps: List[Hypothesis],
        ended_hyps: List[Hypothesis],
    ) -> List[Hypothesis]:
        """Perform post-processing of beam search iterations.
        Args:
            i (int): The length of hypothesis tokens.
            maxlen (int): The maximum length of tokens in beam search.
            running_hyps (List[Hypothesis]): The running hypotheses in beam search.
            ended_hyps (List[Hypothesis]): The ended hypotheses in beam search.
        Returns:
            List[Hypothesis]: The new running hypotheses.
        """
        logging.debug(f"The number of running hypotheses: {len(running_hyps)}")
        if self.token_list is not None:
            logging.debug(
                "Best hypo: "
                + "".join([self.token_list[x] for x in running_hyps[0].yseq[1:]])
            )
        # add eos in the final loop to avoid that there are no ended hyps
        if i == maxlen - 1:
            logging.debug("Adding <eos> in the last position in the loop")
            running_hyps = [
                h._replace(yseq=self.append_token(h.yseq, self.eos))
                for h in running_hyps
            ]

        # add ended hypotheses to a final list, and removed them from current hypotheses
        # (this will be a problem, number of hyps < beam)
        remained_hyps = []
        for hyp in running_hyps:
            if hyp.yseq[-1] == self.eos:
                # e.g., Word LM needs to add final <eos> score
                for k, d in chain(self.full_scorers.items(), self.part_scorers.items()):
                    s = d.final_score(hyp.states[k])
                    hyp.scores[k] += s
                    hyp = hyp._replace(score=hyp.score + self.weights[k] * s)
                ended_hyps.append(hyp)
            else:
                remained_hyps.append(hyp)
        return remained_hyps
