import six

from typing import (
    Any,
    List,
    Tuple
)
from typeguard import check_argument_types

import numpy as np
import onnxruntime
from scipy.special import (
    logsumexp,
    log_softmax
)

from espnet_onnx.utils.config import Config
from .interface import BatchPartialScorerInterface


class CTCPrefixScore:
    """Compute CTC label sequence scores
    which is based on Algorithm 2 in WATANABE et al.
    "HYBRID CTC/ATTENTION ARCHITECTURE FOR END-TO-END SPEECH RECOGNITION,"
    but extended to efficiently compute the probablities of multiple labels
    simultaneously
    """

    def __init__(self, x: np.ndarray, blank: int, eos: float, xp: np):
        assert check_argument_types()
        self.xp = xp
        self.logzero = -10000000000.0
        self.blank = blank
        self.eos = eos
        self.input_length = len(x)
        self.x = x

    def initial_state(self):
        """Obtain an initial CTC state
        :return: CTC state
        """
        # initial CTC state is made of a frame x 2 tensor that corresponds to
        # r_t^n(<sos>) and r_t^b(<sos>), where 0 and 1 of axis=1 represent
        # superscripts n and b (non-blank and blank), respectively.
        r = self.xp.full((self.input_length, 2),
                         self.logzero, dtype=np.float32)
        r[0, 1] = self.x[0, self.blank]
        for i in six.moves.range(1, self.input_length):
            r[i, 1] = r[i - 1, 1] + self.x[i, self.blank]
        return r

    def __call__(self, y, cs, r_prev):
        """Compute CTC prefix scores for next labels
        :param y     : prefix label sequence
        :param cs    : array of next labels
        :param r_prev: previous CTC state
        :return ctc_scores, ctc_states
        """
        # initialize CTC states
        output_length = len(y) - 1  # ignore sos
        # new CTC states are prepared as a frame x (n or b) x n_labels tensor
        # that corresponds to r_t^n(h) and r_t^b(h).
        r = self.xp.ndarray((self.input_length, 2, len(cs)), dtype=np.float32)
        xs = self.x[:, cs]
        if output_length == 0:
            r[0, 0] = xs[0]
            r[0, 1] = self.logzero
        else:
            r[output_length - 1] = self.logzero

        # prepare forward probabilities for the last label
        r_sum = self.xp.logaddexp(
            r_prev[:, 0], r_prev[:, 1]
        )  # log(r_t^n(g) + r_t^b(g))
        last = y[-1]
        if output_length > 0 and last in cs:
            log_phi = self.xp.ndarray(
                (self.input_length, len(cs)), dtype=np.float32)
            for i in six.moves.range(len(cs)):
                log_phi[:, i] = r_sum if cs[i] != last else r_prev[:, 1]
        else:
            log_phi = r_sum

        # compute forward probabilities log(r_t^n(h)), log(r_t^b(h)),
        # and log prefix probabilities log(psi)
        start = max(output_length, 1)
        log_psi = r[start - 1, 0]
        for t in six.moves.range(start, self.input_length):
            r[t, 0] = self.xp.logaddexp(r[t - 1, 0], log_phi[t - 1]) + xs[t]
            r[t, 1] = (
                self.xp.logaddexp(r[t - 1, 0], r[t - 1, 1]
                                  ) + self.x[t, self.blank]
            )
            log_psi = self.xp.logaddexp(log_psi, log_phi[t - 1] + xs[t])

        # get P(...eos|X) that ends with the prefix itself
        eos_pos = self.xp.where(cs == self.eos)[0]
        if len(eos_pos) > 0:
            log_psi[eos_pos] = r_sum[-1]  # log(r_T^n(g) + r_T^b(g))

        # exclude blank probs
        blank_pos = self.xp.where(cs == self.blank)[0]
        if len(blank_pos) > 0:
            log_psi[blank_pos] = self.logzero

        # return the log prefix probability and CTC states, where the label axis
        # of the CTC states is moved to the first axis to slice it easily
        return log_psi, self.xp.rollaxis(r, 2)


class CTCPrefixScorer(BatchPartialScorerInterface):
    """Decoder interface wrapper for CTCPrefixScore."""

    def __init__(self, ctc: Config, eos: int, providers: List[str], use_quantized: bool = False):
        """Initialize class.
        Args:
            ctc (np.ndarray): The CTC implementation.
                For example, :class:`espnet.nets.pytorch_backend.ctc.CTC`
            eos (int): The end-of-sequence id.
        """
        assert check_argument_types()
        if use_quantized:
            self.ctc = onnxruntime.InferenceSession(
                ctc.quantized_model_path,
                providers=providers
            )
        else:
            self.ctc = onnxruntime.InferenceSession(
                ctc.model_path,
                providers=providers
            )
        self.eos = eos
        self.impl = None

    def init_state(self, x: np.ndarray):
        """Get an initial state for decoding.
        Args:
            x (np.ndarray): The encoded feature tensor
        Returns: initial state
        """
        x = self.ctc.run(["ctc_out"], {"x": x[None, :]})[0]
        logp = np.squeeze(x, axis=0)
        # TODO(karita): use CTCPrefixScoreTH
        self.impl = CTCPrefixScore(logp, 0, self.eos, np)
        return 0, self.impl.initial_state()

    def select_state(self, state, i, new_id=None):
        """Select state with relative ids in the main beam search.
        Args:
            state: Decoder state for prefix tokens
            i (int): Index to select a state in the main beam search
            new_id (int): New label id to select a state if necessary
        Returns:
            state: pruned state
        """
        if type(state) == tuple:
            if len(state) == 2:  # for CTCPrefixScore
                sc, st = state
                return sc[i], st[i]
            else:  # for CTCPrefixScoreTH (need new_id > 0)
                r, log_psi, f_min, f_max, scoring_idmap = state
                s = log_psi[i, new_id].repeat(log_psi.shape[1])
                if scoring_idmap is not None:
                    return r[:, :, i, scoring_idmap[i, new_id]], s, f_min, f_max
                else:
                    return r[:, :, i, new_id], s, f_min, f_max
        return None if state is None else state[i]

    def score_partial(self, y, ids, state, x):
        """Score new token.
        Args:
            y (np.ndarray): 1D prefix token
            next_tokens (np.ndarray): torch.int64 next token to score
            state: decoder state for prefix tokens
            x (np.ndarray): 2D encoder feature that generates ys
        Returns:
            tuple[np.ndarray, Any]:
                Tuple of a score tensor for y that has a shape `(len(next_tokens),)`
                and next state for ys
        """
        prev_score, state = state
        presub_score, new_st = self.impl(y, ids, state)
        tscore = np.array(
            presub_score - prev_score, dtype=x.dtype
        )
        return tscore, (presub_score, new_st)

    def batch_init_state(self, x: np.ndarray):
        """Get an initial state for decoding.
        Args:
            x (np.ndarray): The encoded feature tensor
        Returns: initial state
        """
        logp = self.ctc.run(["ctc_out"], {"x": x[None, :]})[0]
        xlen = np.array([logp.shape[1]])
        self.impl = CTCPrefixScoreTH(logp, xlen, 0, self.eos)
        return None

    def batch_score_partial(self, y, ids, state, x):
        """Score new token.
        Args:
            y (np.ndarray): 1D prefix token
            ids (np.ndarray): torch.int64 next token to score
            state: decoder state for prefix tokens
            x (np.ndarray): 2D encoder feature that generates ys
        Returns:
            tuple[np.ndarray, Any]:
                Tuple of a score tensor for y that has a shape `(len(next_tokens),)`
                and next state for ys
        """
        if state[0] is not None:
            batch_state = (
                np.concatenate([s[0][..., None] for s in state], axis=2),
                np.concatenate([s[1][None, :] for s in state]),
                state[0][2],
                state[0][3],
            )
        else:
            batch_state = None
        return self.impl(y, batch_state, ids)
    
    def extend_prob(self, x: np.ndarray):
        """Extend probs for decoding.

        This extension is for streaming decoding
        as in Eq (14) in https://arxiv.org/abs/2006.14941

        Args:
            x (np.ndarray): The encoded feature tensor

        """
        x = self.ctc.run(["ctc_out"], {"x": x[None, :]})[0]
        logp = log_softmax(x, axis=-1)
        self.impl.extend_prob(logp)

    def extend_state(self, state):
        """Extend state for decoding.

        This extension is for streaming decoding
        as in Eq (14) in https://arxiv.org/abs/2006.14941

        Args:
            state: The states of hyps

        Returns: exteded state

        """
        new_state = []
        for s in state:
            new_state.append(self.impl.extend_state(s))

        return new_state


class CTCPrefixScoreTH:
    """Batch processing of CTCPrefixScore
    which is based on Algorithm 2 in WATANABE et al.
    "HYBRID CTC/ATTENTION ARCHITECTURE FOR END-TO-END SPEECH RECOGNITION,"
    but extended to efficiently compute the label probablities for multiple
    hypotheses simultaneously
    See also Seki et al. "Vectorized Beam Search for CTC-Attention-Based
    Speech Recognition," In INTERSPEECH (pp. 3825-3829), 2019.
    """

    def __init__(self, x: np.ndarray, xlens: np.ndarray, blank: int, eos: int, margin: int = 0):
        """Construct CTC prefix scorer
        :param np.ndarray x: input label posterior sequences (B, T, O)
        :param np.ndarray xlens: input lengths (B,)
        :param int blank: blank label id
        :param int eos: end-of-sequence id
        :param int margin: margin parameter for windowing (0 means no windowing)
        """
        assert check_argument_types()
        # In the comment lines,
        # we assume T: input_length, B: batch size, W: beam width, O: output dim.
        self.logzero = -10000000000.0
        self.blank = blank
        self.eos = eos
        self.batch = x.shape[0]
        self.input_length = x.shape[1]
        self.odim = x.shape[2]
        self.dtype = x.dtype

        # Pad the rest of posteriors in the batch
        # TODO(takaaki-hori): need a better way without for-loops
        for i, l in enumerate(xlens):
            if l < self.input_length:
                x[i, l:, :] = self.logzero
                x[i, l:, blank] = 0
        # Reshape input x
        xn = x.transpose(1, 0, 2)  # (B, T, O) -> (T, B, O)
        xb = xn[:, :, None, self.blank].repeat(self.odim, axis=2)
        # operation is faster than np.stack
        self.x = np.concatenate([xn[None, :], xb[None, :]])
        self.end_frames = xlens - 1

        # Setup CTC windowing
        self.margin = margin
        if margin > 0:
            self.frame_ids = np.arange(
                self.input_length, dtype=self.dtype
            )
        # Base indices for index conversion
        self.idx_bh = None
        self.idx_b = np.arange(self.batch)
        self.idx_bo = (self.idx_b * self.odim)[:, None]

    def __call__(self, y, state, scoring_ids=None, att_w=None):
        """Compute CTC prefix scores for next labels
        :param list y: prefix label sequences
        :param tuple state: previous CTC state
        :param np.ndarray pre_scores: scores for pre-selection of hypotheses (BW, O)
        :param np.ndarray att_w: attention weights to decide CTC window
        :return new_state, ctc_local_scores (BW, O)
        """
        output_length = len(y[0]) - 1  # ignore blank and sos
        last_ids = [yi[-1] for yi in y]  # last output label ids
        n_bh = len(last_ids)  # batch * hyps
        n_hyps = n_bh // self.batch  # assuming each utterance has the same # of hyps
        self.scoring_num = scoring_ids.shape[-1] if scoring_ids is not None else 0
        # prepare state info
        if state is None:
            r_prev = np.full(
                (self.input_length, 2, self.batch, n_hyps),
                self.logzero,
                dtype=self.dtype,
            )
            r_prev[:, 1] = np.cumsum(self.x[0, :, :, self.blank], 0)[
                :, :, None]
            r_prev = r_prev.reshape(-1, 2, n_bh)
            s_prev = 0.0
            f_min_prev = 0
            f_max_prev = 1
        else:
            r_prev, s_prev, f_min_prev, f_max_prev = state

        # select input dimensions for scoring
        if self.scoring_num > 0:
            scoring_idmap = np.full(
                (n_bh, self.odim), -1, dtype=np.int64
            )
            snum = self.scoring_num
            if self.idx_bh is None or n_bh > len(self.idx_bh):
                self.idx_bh = np.arange(n_bh).reshape(-1, 1)
            scoring_idmap[self.idx_bh[:n_bh], scoring_ids] = np.arange(snum)
            scoring_idx = (
                scoring_ids + self.idx_bo.repeat(n_hyps, axis=1).reshape(-1, 1)
            ).reshape(-1)
            x_ = np.take(
                self.x.reshape(2, -1, self.batch * self.odim), scoring_idx, axis=2
            ).reshape(2, -1, n_bh, snum)
        else:
            scoring_ids = None
            scoring_idmap = None
            snum = self.odim
            x_ = self.x[:, :, :, None].repeat(
                n_hyps, axis=3).reshape(2, -1, n_bh, snum)

        # new CTC forward probs are prepared as a (T x 2 x BW x S) tensor
        # that corresponds to r_t^n(h) and r_t^b(h) in a batch.
        r = np.full(
            (self.input_length, 2, n_bh, snum),
            self.logzero,
            dtype=self.dtype
        )
        if output_length == 0:
            r[0, 0] = x_[0, 0]

        r_sum = logsumexp(r_prev, axis=1)
        log_phi = r_sum[:, :, None].repeat(snum, 2)
        if scoring_ids is not None:
            for idx in range(n_bh):
                pos = scoring_idmap[idx, int(last_ids[idx])]
                if pos >= 0:
                    log_phi[:, idx, pos] = r_prev[:, 1, idx]
        else:
            for idx in range(n_bh):
                log_phi[:, idx, last_ids[idx]] = r_prev[:, 1, idx]

        # decide start and end frames based on attention weights
        if att_w is not None and self.margin > 0:
            f_arg = np.matmul(att_w, self.frame_ids)
            f_min = max(int(f_arg.min()), f_min_prev)
            f_max = max(int(f_arg.max()), f_max_prev)
            start = min(f_max_prev, max(f_min - self.margin, output_length, 1))
            end = min(f_max + self.margin, self.input_length)
        else:
            f_min = f_max = 0
            start = max(output_length, 1)
            end = self.input_length

        # compute forward probabilities log(r_t^n(h)) and log(r_t^b(h))
        for t in range(start, end):
            rp = r[t - 1]
            rr = np.concatenate([rp[0:1], log_phi[t - 1:t], rp[0:1], rp[1:2]]).reshape(
                2, 2, n_bh, snum
            )
            r[t] = logsumexp(rr, 1) + x_[:, t]

        # compute log prefix probabilities log(psi)
        log_phi_x = np.concatenate(
            (log_phi[0][None, :], log_phi[:-1]), axis=0) + x_[0]
        if scoring_ids is not None:
            log_psi = np.full(
                (n_bh, self.odim), self.logzero, dtype=self.dtype
            )
            log_psi_ = logsumexp(
                np.concatenate(
                    (log_phi_x[start:end], r[start - 1, 0][None, :]), axis=0),
                axis=0,
            )
            for si in range(n_bh):
                log_psi[si, scoring_ids[si].astype(np.int64)] = log_psi_[si]
        else:
            log_psi = logsumexp(
                np.concatenate(
                    (log_phi_x[start:end], r[start - 1, 0][None, :]), axis=0),
                axis=0,
            )

        for si in range(n_bh):
            log_psi[si, self.eos] = r_sum[self.end_frames[si // n_hyps], si]

        # exclude blank probs
        log_psi[:, self.blank] = self.logzero

        return (log_psi - s_prev), (r, log_psi, f_min, f_max, scoring_idmap)

    def index_select_state(self, state, best_ids):
        """Select CTC states according to best ids
        :param state    : CTC state
        :param best_ids : index numbers selected by beam pruning (B, W)
        :return selected_state
        """
        r, s, f_min, f_max, scoring_idmap = state
        # convert ids to BHO space
        n_bh = len(s)
        n_hyps = n_bh // self.batch
        vidx = (best_ids + (self.idx_b * (n_hyps * self.odim)
                            ).reshape(-1, 1)).reshape(-1)
        # select hypothesis scores
        s_new = np.take(s.reshape(-1), vidx, axis=0)
        s_new = s_new.reshape(-1, 1).repeat(self.odim,
                                            axis=1).reshape(n_bh, self.odim)
        # convert ids to BHS space (S: scoring_num)
        if scoring_idmap is not None:
            snum = self.scoring_num
            hyp_idx = (best_ids // self.odim + (self.idx_b * n_hyps).reshape(-1, 1)).reshape(
                -1
            )
            label_ids = np.fmod(best_ids, self.odim).reshape(-1)
            score_idx = scoring_idmap[hyp_idx, label_ids]
            score_idx[score_idx == -1] = 0
            vidx = score_idx + hyp_idx * snum
        else:
            snum = self.odim
        # select forward probabilities
        r_new = np.take(r.reshape(-1, 2, n_bh * snum), vidx, axis=2).reshape(
            -1, 2, n_bh
        )
        return r_new, s_new, f_min, f_max

    def extend_prob(self, x):
        """Extend CTC prob.
        :param np.ndarray x: input label posterior sequences (B, T, O)
        """
        if self.x.shape[1] < x.shape[1]:  # self.x (2,T,B,O); x (B,T,O)
            # Pad the rest of posteriors in the batch
            # TODO(takaaki-hori): need a better way without for-loops
            xlens = np.array([x.shape[1]])
            for i, l in enumerate(xlens):
                if l < self.input_length:
                    x[i, l:, :] = self.logzero
                    x[i, l:, self.blank] = 0
            tmp_x = self.x
            xn = x.transpose(1, 0, 2)  # (B, T, O) -> (T, B, O)
            xb = xn[:, :, None, self.blank].repeat(self.odim, axis=2)
            self.x = np.concatenate([xn[None, :], xb[None, :]])  # (2, T, B, O)
            self.x[:, : tmp_x.shape[1], :, :] = tmp_x
            self.input_length = x.shape[1]
            self.end_frames = xlens - 1

    def extend_state(self, state):
        """Compute CTC prefix state.
        :param state    : CTC state
        :return ctc_state
        """

        if state is None:
            # nothing to do
            return state
        else:
            r_prev, s_prev, f_min_prev, f_max_prev = state

            r_prev_new = np.full(
                (self.input_length, 2),
                self.logzero,
                dtype=self.dtype,
            )
            start = max(r_prev.shape[0], 1)
            r_prev_new[0:start] = r_prev
            for t in six.moves.range(start, self.input_length):
                r_prev_new[t, 1] = r_prev_new[t - 1, 1] + \
                    self.x[0, t, :, self.blank]

            return (r_prev_new, s_prev, f_min_prev, f_max_prev)
