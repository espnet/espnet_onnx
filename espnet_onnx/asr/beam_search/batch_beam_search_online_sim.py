import logging
from pathlib import Path
from typing import (
    List,
    Union
)

import numpy as np

from espnet_onnx.utils.config import Config, get_config
from espnet_onnx.utils.function import (
    narrow,
    end_detect
)
from .batch_beam_search import BatchBeamSearch
from .hyps import Hypothesis


class BatchBeamSearchOnlineSim(BatchBeamSearch):
    """Online beam search implementation.

    This implementation is a modification version
    from espnet.nets.batch_beam_search_online_sim.py
    """
    
    def set_streaming_config(self, config_path: Union[str, Path]):
        """Set config file for streaming decoding.

        Args:
            config (Config): The config for asr training

        """
        self.block_size = None
        self.hop_size = None
        self.look_ahead = None
        config = get_config(config_path)
        if "encoder_conf" not in config.keys():
            if "config" in config.keys():
                config = get_config(config.config)
            else:
                raise ValueError(
                    "Cannot find config file for streaming decoding: "
                    + "apply batch beam search instead."
                )
        
        if "encoder_conf" in config.keys():
            if "block_size" in config.encoder_conf.keys():
                self.block_size = config.encoder_conf.block_size
            if "hop_size" in config.encoder_conf.keys():
                self.hop_size = config.encoder_conf.hop_size
            if "look_ahead" in config.encoder_conf..keys():
                self.look_ahead = config.encoder_conf.look_ahead

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

    def forward(
        self, x: torch.Tensor
    ) -> List[Hypothesis]:
        """Perform beam search.

        Args:
            x (torch.Tensor): Encoded speech feature (T, D)

        Returns:
            list[Hypothesis]: N-best decoding results

        """
        self.conservative = True  # always true

        if self.block_size and self.hop_size and self.look_ahead:
            cur_end_frame = int(self.block_size - self.look_ahead)
        else:
            cur_end_frame = x.shape[0]
        process_idx = 0
        if cur_end_frame < x.shape[0]:
            h = narrow(x, 0, 0, cur_end_frame)
        else:
            h = x

        # set length bounds
        if self.maxlenratio == 0:
            maxlen = x.shape[0]
        else:
            maxlen = max(1, int(self.maxlenratio * x.size(0)))
        minlen = int(self.minlenratio * x.size(0))
        print("decoder input length: " + str(x.shape[0]))
        print("max output length: " + str(maxlen))
        print("min output length: " + str(minlen))

        # main loop of prefix search
        running_hyps = self.init_hyp(h)
        prev_hyps = []
        ended_hyps = []
        prev_repeat = False

        continue_decode = True

        while continue_decode:
            move_to_next_block = False
            if cur_end_frame < x.shape[0]:
                h = narrow(x, 0, 0, cur_end_frame)
            else:
                h = x

            print(('len(h) : ', len(h)))
            # extend states for ctc
            self.extend(h, running_hyps)

            while process_idx < maxlen:
                print("position " + str(process_idx))
                best = self.search(running_hyps, h)

                if process_idx == maxlen - 1:
                    # end decoding
                    running_hyps = self.post_process(
                        process_idx, maxlen, self.maxlenratio, best, ended_hyps
                    )
                n_batch = best.yseq.shape[0]
                local_ended_hyps = []
                is_local_eos = (
                    best.yseq[torch.arange(n_batch), best.length - 1] == self.eos
                )
                for i in range(is_local_eos.shape[0]):
                    if is_local_eos[i]:
                        hyp = self._select(best, i)
                        local_ended_hyps.append(hyp)
                    # NOTE(tsunoo): check repetitions here
                    # This is a implicit implementation of
                    # Eq (11) in https://arxiv.org/abs/2006.14941
                    # A flag prev_repeat is used instead of using set
                    elif (
                        not prev_repeat
                        and best.yseq[i, -1] in best.yseq[i, :-1]
                        and cur_end_frame < x.shape[0]
                    ):
                        move_to_next_block = True
                        prev_repeat = True
                if self.maxlenratio == 0.0 and end_detect(
                    [lh.asdict() for lh in local_ended_hyps], process_idx
                ):
                    print(f"end detected at {process_idx}")
                    continue_decode = False
                    break
                if len(local_ended_hyps) > 0 and cur_end_frame < x.shape[0]:
                    move_to_next_block = True

                if move_to_next_block:
                    if (
                        self.hop_size
                        and cur_end_frame + int(self.hop_size) + int(self.look_ahead)
                        < x.shape[0]
                    ):
                        cur_end_frame += int(self.hop_size)
                    else:
                        cur_end_frame = x.shape[0]
                    print("Going to next block: %d", cur_end_frame)
                    if process_idx > 1 and len(prev_hyps) > 0 and self.conservative:
                        running_hyps = prev_hyps
                        process_idx -= 1
                        prev_hyps = []
                    break

                prev_repeat = False
                prev_hyps = running_hyps
                running_hyps = self.post_process(
                    process_idx, maxlen, self.maxlenratio, best, ended_hyps
                )

                if cur_end_frame >= x.shape[0]:
                    for hyp in local_ended_hyps:
                        ended_hyps.append(hyp)

                if len(running_hyps) == 0:
                    print("no hypothesis. Finish decoding.")
                    continue_decode = False
                    break
                else:
                    print(f"remained hypotheses: {len(running_hyps)}")
                # increment number
                process_idx += 1

        nbest_hyps = sorted(ended_hyps, key=lambda x: x.score, reverse=True)
        # check the number of hypotheses reaching to eos
        if len(nbest_hyps) == 0:
            logging.warning(
                "there is no N-best results, perform recognition "
                "again with smaller self.minlenratio."
            )
            return (
                []
                if self.minlenratio < 0.1
                else self.forward(x, self.maxlenratio, max(0.0, self.minlenratio - 0.1))
            )

        # report the best result
        best = nbest_hyps[0]
        for k, v in best.scores.items():
            print(
                f"{v:6.2f} * {self.weights[k]:3} = {v * self.weights[k]:6.2f} for {k}"
            )
        print(f"total log probability: {best.score:.2f}")
        print(f"normalized log probability: {best.score / len(best.yseq):.2f}")
        print(f"total number of ended hypotheses: {len(nbest_hyps)}")
        if self.token_list is not None:
            print(
                "best hypo: "
                + "".join([self.token_list[x] for x in best.yseq[1:-1]])
                + "\n"
            )
        return nbest_hyps

    def extend(self, x: np.ndarray, hyps: Hypothesis) -> List[Hypothesis]:
        """Extend probabilities and states with more encoded chunks.

        Args:
            x (torch.Tensor): The extended encoder output feature
            hyps (Hypothesis): Current list of hypothesis

        Returns:
            Hypothesis: The extended hypothesis

        """
        for k, d in self.scorers.items():
            if hasattr(d, "extend_prob"):
                d.extend_prob(x)
            if hasattr(d, "extend_state"):
                hyps.states[k] = d.extend_state(hyps.states[k])
