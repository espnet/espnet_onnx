from typing import List

import numpy as np

from espnet_onnx.asr.beam_search.batch_beam_search import BatchBeamSearch
from espnet_onnx.asr.beam_search.hyps import Hypothesis


class BatchBeamSearchOnlineSim(BatchBeamSearch):
    """Online beam search implementation.

    This implementation is a modification version
    from espnet.nets.batch_beam_search_online_sim.py
    """

    def set_streaming_config(self, block_size, hop_size, look_ahead):
        """Set config file for streaming decoding.

        Args:
            config (Config): The config for asr training

        """
        self.block_size = block_size
        self.hop_size = hop_size
        self.look_ahead = look_ahead
        self.process_idx = 0
        self.max_frame_len = 10000

    def set_block_size(self, block_size: int):
        """Set block size for streaming decoding.

        Args:
            block_size (int): The block size of encoder
        """
        self.block_size = block_size

    def set_hop_size(self, hop_size: int):
        """Set hop size for streaming decoding.

        Args:
            hop_size (int): The hop size of encoder
        """
        self.hop_size = hop_size

    def set_look_ahead(self, look_ahead: int):
        """Set look ahead size for streaming decoding.

        Args:
            look_ahead (int): The look ahead size of encoder
        """
        self.look_ahead = look_ahead

    def start(self):
        self.conservative = True  # always true
        assert (
            self.block_size and self.hop_size and self.look_ahead
        ), "block_size, hop_size, and look_ahead must be set."
        self.cur_end_frame = int(self.block_size - self.look_ahead)

        # main loop of prefix search
        self.running_hyps = []
        self.ended_hyps = []
        self.prev_repeat = False
        self._init_hyp = True

    def end(self):
        # main loop of prefix search
        self.running_hyps = []
        self.ended_hyps = []
        self.prev_repeat = False
        self._init_hyp = False

    def __call__(self, h: np.ndarray) -> List[Hypothesis]:
        # extend states for ctc
        if self._init_hyp:
            self.minlen = int(self.minlenratio * h.shape[0])
            self.running_hyps = self.init_hyp(h)
            self.prev_hyps = []
            self._init_hyp = False

        maxlen = (
            h.shape[0]
            if self.maxlenratio == 0
            else max(1, int(maxlenratio * x.shape[0]))
        )
        move_to_next_block = False
        best = None
        # extend states for ctc
        self.extend(h, self.running_hyps)

        while self.process_idx < h.shape[0]:
            best = self.search(self.running_hyps, h)

            n_batch = best.yseq.shape[0]
            local_ended_hyps = []
            is_local_eos = best.yseq[np.arange(n_batch), best.length - 1] == self.eos
            for i in range(is_local_eos.shape[0]):
                if is_local_eos[i]:
                    hyp = self._select(best, i)
                    local_ended_hyps.append(hyp)
                elif (
                    not self.prev_repeat
                    and best.yseq[i, -1] in best.yseq[i, :-1]
                    and self.cur_end_frame < h.shape[0]
                ):
                    move_to_next_block = True
                    self.prev_repeat = True

            if len(local_ended_hyps) > 0 and self.cur_end_frame < self.max_frame_len:
                move_to_next_block = True

            if move_to_next_block:
                if (
                    self.hop_size
                    and self.cur_end_frame + int(self.hop_size) + int(self.look_ahead)
                    < self.max_frame_len
                ):
                    self.cur_end_frame += int(self.hop_size)
                else:
                    self.cur_end_frame = h.shape[0]

                if (
                    self.process_idx > 1
                    and len(self.prev_hyps) > 0
                    and self.conservative
                ):
                    self.running_hyps = self.prev_hyps
                    self.process_idx -= 1
                    self.prev_hyps = []
                break

            self.prev_repeat = False
            self.prev_hyps = self.running_hyps
            self.running_hyps = self.post_process(
                self.process_idx, maxlen, best, self.ended_hyps
            )

            if self.cur_end_frame >= h.shape[0]:
                for hyp in local_ended_hyps:
                    self.ended_hyps.append(hyp)

            # increment number
            self.process_idx += 1

        if best is not None:
            return sorted(self.unbatchfy(best), key=lambda x: x.score, reverse=True)
        else:
            return []

    def extend(self, x: np.ndarray, hyps: Hypothesis) -> List[Hypothesis]:
        for k, d in self.scorers.items():
            if hasattr(d, "extend_prob"):
                d.extend_prob(x)
            if hasattr(d, "extend_state"):
                hyps.states[k] = d.extend_state(hyps.states[k])
