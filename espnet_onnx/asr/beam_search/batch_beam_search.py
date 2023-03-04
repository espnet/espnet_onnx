"""Parallel beam search module."""

import logging
from typing import Any, Dict, List, Tuple

import numpy as np

from espnet_onnx.asr.beam_search.beam_search import BeamSearch
from espnet_onnx.asr.beam_search.hyps import BatchHypothesis, Hypothesis
from espnet_onnx.utils.function import pad_sequence, topk


class BatchBeamSearch(BeamSearch):
    """Batch beam search implementation."""

    def batchfy(self, hyps: List[Hypothesis]) -> BatchHypothesis:
        """Convert list to batch."""
        if len(hyps) == 0:
            return BatchHypothesis()
        return BatchHypothesis(
            yseq=pad_sequence(
                [h.yseq for h in hyps], batch_first=True, padding_value=self.eos
            ),
            length=np.array([len(h.yseq) for h in hyps], dtype=np.int64),
            score=np.array([h.score for h in hyps]),
            scores={k: np.array([h.scores[k] for h in hyps]) for k in self.scorers},
            states={k: [h.states[k] for h in hyps] for k in self.scorers},
        )

    def _batch_select(self, hyps: BatchHypothesis, ids: List[int]) -> BatchHypothesis:
        return BatchHypothesis(
            yseq=hyps.yseq[ids],
            score=hyps.score[ids],
            length=hyps.length[ids],
            scores={k: v[ids] for k, v in hyps.scores.items()},
            states={
                k: [self.scorers[k].select_state(v, i) for i in ids]
                for k, v in hyps.states.items()
            },
        )

    def _select(self, hyps: BatchHypothesis, i: int) -> Hypothesis:
        return Hypothesis(
            yseq=hyps.yseq[i, : hyps.length[i]],
            score=hyps.score[i],
            scores={k: v[i] for k, v in hyps.scores.items()},
            states={
                k: self.scorers[k].select_state(v, i) for k, v in hyps.states.items()
            },
        )

    def unbatchfy(self, batch_hyps: BatchHypothesis) -> List[Hypothesis]:
        """Revert batch to list."""
        return [
            Hypothesis(
                yseq=batch_hyps.yseq[i][: batch_hyps.length[i]],
                score=batch_hyps.score[i],
                scores={k: batch_hyps.scores[k][i] for k in self.scorers},
                states={
                    k: v.select_state(batch_hyps.states[k], i)
                    for k, v in self.scorers.items()
                },
            )
            for i in range(len(batch_hyps.length))
        ]

    def batch_beam(
        self, weighted_scores: np.ndarray, ids: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Batch-compute topk full token ids and partial token ids.
        Args:
            weighted_scores (np.ndarray): The weighted sum scores for each tokens.
                Its shape is `(n_beam, self.vocab_size)`.
            ids (np.ndarray): The partial token ids to compute topk.
                Its shape is `(n_beam, self.pre_beam_size)`.
        Returns:
            Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
                The topk full (prev_hyp, new_token) ids
                and partial (prev_hyp, new_token) ids.
                Their shapes are all `(self.beam_size,)`
        """
        top_ids = topk(weighted_scores.reshape(-1), self.beam_size)
        prev_hyp_ids = top_ids // self.n_vocab
        new_token_ids = top_ids % self.n_vocab
        return prev_hyp_ids, new_token_ids, prev_hyp_ids, new_token_ids

    def init_hyp(self, x: np.ndarray) -> BatchHypothesis:
        """Get an initial hypothesis data.
        Args:
            x (np.ndarray): The encoder output feature
        Returns:
            Hypothesis: The initial hypothesis.
        """
        init_states = dict()
        init_scores = dict()
        for k, d in self.scorers.items():
            init_states[k] = d.batch_init_state(x)
            init_scores[k] = 0.0
        return self.batchfy(
            [
                Hypothesis(
                    score=0.0,
                    scores=init_scores,
                    states=init_states,
                    yseq=np.array([self.sos], dtype=np.int64),
                )
            ]
        )

    def score_full(
        self, hyp: BatchHypothesis, x: np.ndarray
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
            scores[k], states[k] = d.batch_score(hyp.yseq, hyp.states[k], x)
        return scores, states

    def score_partial(
        self, hyp: BatchHypothesis, ids: np.ndarray, x: np.ndarray
    ) -> Tuple[Dict[str, np.ndarray], Dict[str, Any]]:
        """Score new hypothesis by `self.full_scorers`.
        Args:
            hyp (Hypothesis): Hypothesis with prefix tokens to score
            ids (np.ndarray): 2D tensor of new partial tokens to score
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
        for k, d in self.part_scorers.items():
            scores[k], states[k] = d.batch_score_partial(
                hyp.yseq, ids, hyp.states[k], x
            )
        return scores, states

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
        for k, v in part_states.items():
            new_states[k] = v
        return new_states

    def search(self, running_hyps: BatchHypothesis, x: np.ndarray) -> BatchHypothesis:
        """Search new tokens for running hypotheses and encoded speech x.
        Args:
            running_hyps (BatchHypothesis): Running hypotheses on beam
            x (np.ndarray): Encoded speech feature (T, D)
        Returns:
            BatchHypothesis: Best sorted hypotheses
        """
        n_batch = len(running_hyps)
        part_ids = None  # no pre-beam
        # batch scoring
        weighted_scores = np.zeros((n_batch, self.n_vocab), dtype=x.dtype)
        scores, states = self.score_full(
            running_hyps,
            np.vstack([x for _ in range(n_batch)]).reshape(n_batch, *x.shape),
        )
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
        part_scores, part_states = self.score_partial(running_hyps, part_ids, x)
        for k in self.part_scorers:
            weighted_scores += self.weights[k] * part_scores[k]
        # add previous hyp scores
        weighted_scores += np.expand_dims(running_hyps.score, 1)

        best_hyps = []
        prev_hyps = self.unbatchfy(running_hyps)
        for (
            full_prev_hyp_id,
            full_new_token_id,
            part_prev_hyp_id,
            part_new_token_id,
        ) in zip(*self.batch_beam(weighted_scores, part_ids)):
            prev_hyp = prev_hyps[full_prev_hyp_id]
            best_hyps.append(
                Hypothesis(
                    score=weighted_scores[full_prev_hyp_id, full_new_token_id],
                    yseq=self.append_token(prev_hyp.yseq, full_new_token_id),
                    scores=self.merge_scores(
                        prev_hyp.scores,
                        {k: v[full_prev_hyp_id] for k, v in scores.items()},
                        full_new_token_id,
                        {k: v[part_prev_hyp_id] for k, v in part_scores.items()},
                        part_new_token_id,
                    ),
                    states=self.merge_states(
                        {
                            k: self.full_scorers[k].select_state(v, full_prev_hyp_id)
                            for k, v in states.items()
                        },
                        {
                            k: self.part_scorers[k].select_state(
                                v, part_prev_hyp_id, part_new_token_id
                            )
                            for k, v in part_states.items()
                        },
                        part_new_token_id,
                    ),
                )
            )
        return self.batchfy(best_hyps)

    def post_process(
        self,
        i: int,
        maxlen: int,
        running_hyps: BatchHypothesis,
        ended_hyps: List[Hypothesis],
    ) -> BatchHypothesis:
        """Perform post-processing of beam search iterations.
        Args:
            i (int): The length of hypothesis tokens.
            maxlen (int): The maximum length of tokens in beam search.
            running_hyps (BatchHypothesis): The running hypotheses in beam search.
            ended_hyps (List[Hypothesis]): The ended hypotheses in beam search.
        Returns:
            BatchHypothesis: The new running hypotheses.
        """
        n_batch = running_hyps.yseq.shape[0]
        logging.debug(f"the number of running hypothes: {n_batch}")
        if self.token_list is not None:
            logging.debug(
                "best hypo: "
                + "".join(
                    [
                        self.token_list[int(x)]
                        for x in running_hyps.yseq[0, 1 : running_hyps.length[0]]
                    ]
                )
            )
        # add eos in the final loop to avoid that there are no ended hyps
        if i == maxlen - 1:
            logging.debug("adding <eos> in the last position in the loop")
            yseq_eos = np.hstack(
                (
                    running_hyps.yseq,
                    np.full(
                        (n_batch, 1),
                        self.eos,
                        dtype=np.int64,
                    ),
                )
            )

            running_hyps.yseq.resize(yseq_eos.shape, refcheck=False)
            running_hyps.yseq[:] = yseq_eos
            running_hyps.length[:] = yseq_eos.shape[1]

        # add ended hypotheses to a final list, and removed them from current hypotheses
        # (this will be a probmlem, number of hyps < beam)
        is_eos = (
            running_hyps.yseq[
                np.arange(n_batch), (running_hyps.length - 1).astype(np.int64)
            ]
            == self.eos
        )
        for b in np.transpose(np.nonzero(is_eos)).reshape(-1):
            hyp = self._select(running_hyps, b)
            ended_hyps.append(hyp)
        remained_ids = np.transpose(np.nonzero(is_eos == 0)).reshape(-1)
        return self._batch_select(running_hyps, remained_ids)
