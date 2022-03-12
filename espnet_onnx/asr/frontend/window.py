from typeguard import check_argument_types
import numpy as np


def get_window(
    window_type: str,
    length: int,
    periodic: bool = True
):
    assert check_argument_types()
    N = length + 1 if periodic else length
    if window_type == 'bartlett':
        w = np.bartlett(N)
    elif window_type == 'blackman':
        w = np.blackman(N)
    elif window_type == 'hamming':
        w = np.hamming(N)
    elif window_type == 'hann':
        w = np.hanning(N)
    elif window_type == 'kaiser':
        w = np.kaiser(N)
    else:
        raise Error(f'Window type ${window_type} is not supported.')
    w = w[:-1] if periodic else w
    return w
