import numpy as np
import six


def subsequent_mask(size):
    """Create mask for subsequent steps (size, size).
    Modified from the original mask function to apply for fix-length mask.

    Args:
        size(int) : size of mask

    Examples:
        >>> subsequent_mask(3)
        [[1, 0, 0],
        [1, 1, 0],
        [1, 1, 1]]
    """
    return np.tril(np.ones((size, size))).astype(np.float32)


def mask_fill(arr, mask, mask_value):
    """Numpy implementation of torch.Tensor.masked_fill

    Args:
        arr (np.ndarray): Array to mask.
        mask (np.ndarray): Every value with 1 (or True) will mask with mask_value
        mask_value (number): Value to apply for mask

    Returns:
        np.ndarray: Masked array
    """
    arr[mask.astype(np.bool) == True] = mask_value
    return arr


def make_pad_mask(lengths, xs=None, dim=-1, xs_shape=None):
    """Make mask tensor containing indices of padded part.

    Args:
        lengths (np.ndarray or List): Batch of lengths (B,).
        xs (np.ndarray, optional): The reference tensor.
            If set, masks will be the same shape as this tensor.
        dim (int, optional): Dimension indicator of the above tensor.
            See the example.

    Returns:
        np.ndarray: Mask tensor containing indices of padded part.

    Examples:
        With only lengths.
        >>> lengths = [5, 3, 2]
        >>> make_pad_mask(lengths)
        masks = [[0, 0, 0, 0 ,0],
                 [0, 0, 0, 1, 1],
                 [0, 0, 1, 1, 1]]

        With the reference tensor.
        >>> xs = np.zeros((3, 2, 4))
        >>> make_pad_mask(lengths, xs)
        array([[[0, 0, 0, 0],
                 [0, 0, 0, 0]],
                [[0, 0, 0, 1],
                 [0, 0, 0, 1]],
                [[0, 0, 1, 1],
                 [0, 0, 1, 1]]])

        >>> xs = np.zeros((3, 2, 6))
        >>> make_pad_mask(lengths, xs)
        array([[[0, 0, 0, 0, 0, 1],
                 [0, 0, 0, 0, 0, 1]],
                [[0, 0, 0, 1, 1, 1],
                 [0, 0, 0, 1, 1, 1]],
                [[0, 0, 1, 1, 1, 1],
                 [0, 0, 1, 1, 1, 1]]])

        With the reference tensor and dimension indicator.
        >>> xs = np.zeros((3, 6, 6))
        >>> make_pad_mask(lengths, xs, 1)
        array([[[0, 0, 0, 0, 0, 0],
                 [0, 0, 0, 0, 0, 0],
                 [0, 0, 0, 0, 0, 0],
                 [0, 0, 0, 0, 0, 0],
                 [0, 0, 0, 0, 0, 0],
                 [1, 1, 1, 1, 1, 1]],
                [[0, 0, 0, 0, 0, 0],
                 [0, 0, 0, 0, 0, 0],
                 [0, 0, 0, 0, 0, 0],
                 [1, 1, 1, 1, 1, 1],
                 [1, 1, 1, 1, 1, 1],
                 [1, 1, 1, 1, 1, 1]],
                [[0, 0, 0, 0, 0, 0],
                 [0, 0, 0, 0, 0, 0],
                 [1, 1, 1, 1, 1, 1],
                 [1, 1, 1, 1, 1, 1],
                 [1, 1, 1, 1, 1, 1],
                 [1, 1, 1, 1, 1, 1]]])

        >>> make_pad_mask(lengths, xs, 2)
        array([[[0, 0, 0, 0, 0, 1],
                 [0, 0, 0, 0, 0, 1],
                 [0, 0, 0, 0, 0, 1],
                 [0, 0, 0, 0, 0, 1],
                 [0, 0, 0, 0, 0, 1],
                 [0, 0, 0, 0, 0, 1]],
                [[0, 0, 0, 1, 1, 1],
                 [0, 0, 0, 1, 1, 1],
                 [0, 0, 0, 1, 1, 1],
                 [0, 0, 0, 1, 1, 1],
                 [0, 0, 0, 1, 1, 1],
                 [0, 0, 0, 1, 1, 1]],
                [[0, 0, 1, 1, 1, 1],
                 [0, 0, 1, 1, 1, 1],
                 [0, 0, 1, 1, 1, 1],
                 [0, 0, 1, 1, 1, 1],
                 [0, 0, 1, 1, 1, 1],
                 [0, 0, 1, 1, 1, 1]]])
    """
    if xs is not None:
        base = np.zeros(xs.shape)
    else:
        base = np.zeros((len(lengths), max(lengths)))

    if len(base.shape) == 3 and dim == 1:
        base = base.transpose(0, 2, 1)

    for i in range(len(base)):
        base[i][..., lengths[i] :] = 1

    if len(base.shape) == 3 and dim == 1:
        base = base.transpose(0, 2, 1)

    return base


def topk(x: np.ndarray, k: int, require_value: bool = False):
    """Get indicies of topk.

    Args:
        x (np.ndarray)
        k (int)

    Examples:
        >>> a = np.array([3,6,1,7])
        >>> topk(a, 2)
        array([3, 1])
        >>> b = np.array([[3,6,2,7],
                          [6,2,4,8],
                          [1,1,7,3]])
        >>> topk(b, 2)
        array([[3,1],
               [3,0],
               [2,3]])
    """
    topk_index = np.argpartition(x, x.shape[-1] - k, axis=-1)[..., -k:]
    if require_value:
        return np.take_along_axis(x, topk_index, axis=-1), topk_index
    else:
        return topk_index


def pad_sequence(yseqs, batch_first=False, padding_value=0):
    """Numpy implementation of torch.pad_sequence

    Args:
        yseqs (np.ndarray): List of array. (B, *)
        batch_first (bool):
        padding_value (int, optional): Padding value. Defaults to 0.

    Returns:
        np.ndarray

    Examples:
        >>> a = np.ones(25, 300)
        >>> b = np.ones(22, 300)
        >>> c = np.ones(15, 300)
        >>> pad_sequence([a, b, c]).size()
        (25, 3, 300)

        >>> pad_sequence([a, b, c], batch_first=True).size()
        (3, 25, 300)
    """
    if len(yseqs) == 1:
        return np.array(yseqs)

    max_idx = np.argmax([y.shape[0] for y in yseqs])
    max_shape = yseqs[max_idx].shape
    base = np.ones((len(yseqs), *max_shape)) * padding_value
    for i, y in enumerate(yseqs):
        base[i][: y.shape[0]] = y
    if batch_first:
        return base
    else:
        return base.transpose(1, 0, *np.arange(2, len(base.shape)))


def is_prefix(x, pref) -> bool:
    """Check if pref is a prefix of x.

    Args:
        x: Label ID sequence.
        pref: Prefix label ID sequence.

    Returns:
        : Whether pref is a prefix of x.
    """
    if len(pref) >= len(x):
        return False

    for i in range(len(pref)):
        if pref[i] != x[i]:
            return False

    return True


def recombine_hyps(hyps):
    """Recombine hypotheses with same label ID sequence.

    Args:
        hyps: Hypotheses.

    Returns:
       final: Recombined hypotheses.
    """
    final = []

    for hyp in hyps:
        seq_final = [f.yseq for f in final if f.yseq]

        if hyp.yseq in seq_final:
            seq_pos = seq_final.index(hyp.yseq)

            final[seq_pos].score = np.logaddexp(final[seq_pos].score, hyp.score)

        else:
            final.append(hyp)

    return final


def select_k_expansions(
    hyps,
    logps,
    beam_size,
    gamma,
    beta,
):
    """Return K hypotheses candidates for expansion from a list of hypothesis.
    K candidates are selected according to the extended hypotheses probabilities
    and a prune-by-value method. Where K is equal to beam_size + beta.

    Args:
        hyps: Hypotheses.
        beam_logp: Log-probabilities for hypotheses expansions.
        beam_size: Beam size.
        gamma: Allowed logp difference for prune-by-value method.
        beta: Number of additional candidates to store.

    Return:
        k_expansions: Best K expansion hypotheses candidates.
    """
    k_expansions = []

    for i, hyp in enumerate(hyps):
        hyp_i = [(int(k), hyp.score + float(logp)) for k, logp in enumerate(logps[i])]
        k_best_exp = max(hyp_i, key=lambda x: x[1])[1]

        k_expansions.append(
            sorted(
                filter(lambda x: (k_best_exp - gamma) <= x[1], hyp_i),
                key=lambda x: x[1],
                reverse=True,
            )[: beam_size + beta]
        )

    return k_expansions


def subtract(x, subset):
    """Remove elements of subset if corresponding label ID sequence already exist in x.

    Args:
        x: Set of hypotheses.
        subset: Subset of x.

    Returns:
       final: New set of hypotheses.
    """
    final = []

    for x_ in x:
        if any(x_.yseq == sub.yseq for sub in subset):
            continue
        final.append(x_)

    return final


def narrow(arr: np.ndarray, axis: int, start: int, length: int):
    """Numpy implementation of torch.narrow

    Args:
        arr (np.ndarray): the array to narrow
        axis (int): the dimension along which to narrow
        start (int): the starting dimension
        length (int): the distance to the ending dimension
    """
    return arr.take(np.arange(start, start + length), axis=axis)


def end_detect(ended_hyps, i, M=3, D_end=np.log(1 * np.exp(-10))):
    """End detection.
    described in Eq. (50) of S. Watanabe et al
    "Hybrid CTC/Attention Architecture for End-to-End Speech Recognition"
    :param ended_hyps:
    :param i:
    :param M:
    :param D_end:
    :return:
    """
    if len(ended_hyps) == 0:
        return False
    count = 0
    best_hyp = sorted(ended_hyps, key=lambda x: x["score"], reverse=True)[0]
    for m in six.moves.range(M):
        # get ended_hyps with their length is i - m
        hyp_length = i - m
        hyps_same_length = [x for x in ended_hyps if len(x["yseq"]) == hyp_length]
        if len(hyps_same_length) > 0:
            best_hyp_same_length = sorted(
                hyps_same_length, key=lambda x: x["score"], reverse=True
            )[0]
            if best_hyp_same_length["score"] - best_hyp["score"] < D_end:
                count += 1

    if count == M:
        return True
    else:
        return False
