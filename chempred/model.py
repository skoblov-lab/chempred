"""



"""

from typing import Sequence, Tuple, Optional, List
from functools import reduce
from itertools import chain
from fn import F

from keras import layers
import numpy as np

from chempred.chemdner import Interval


def stack_conv(prev: layers.Layer, param: Tuple[str, int, int]):
    name, nfilt, kern_size = param
    return layers.Convolution1D(
        nfilt, kern_size, activation="relu", name=name,
    )(prev)


def build_conv(incomming,
               filters: Optional[Sequence[int]],
               kernels: Optional[Sequence[int]]):
    filters = filters or []
    kernels = kernels or []
    assert len(filters) == len(kernels)

    conv_names = ("conv_{}".format(i) for i in range(1, len(kernels)+1))
    conv = reduce(stack_conv, zip(conv_names, filters, kernels), incomming)
    return conv


def stack_lstm(prev: layers.Layer, param: Tuple[str, int, float, float],
               bidirectional: bool, stateful: bool):
    name, units, indrop, recdrop = param
    layer = layers.LSTM(units, dropout=indrop, recurrent_dropout=recdrop,
                        return_sequences=True, stateful=stateful)
    return (layers.Bidirectional(layer) if bidirectional else layer)(prev)


def build_rec(nsteps: Optional[Sequence[int]],
              lstm_inp_drop: Optional[Sequence[float]],
              lstm_rec_drop: Optional[Sequence[float]],
              bidirectional=True,
              stateful=False):

    nsteps = nsteps or []
    lstm_inp_drop = lstm_inp_drop or []
    lstm_rec_drop = lstm_rec_drop or []
    assert len(nsteps) == len(lstm_rec_drop) == len(lstm_inp_drop)

    def rec(incomming):
        rec_names = ("rec_{}".format(i) for i in range(1, len(nsteps)+1))
        recur = reduce(
            F(stack_lstm, bidirectional=bidirectional, stateful=stateful),
            zip(rec_names, nsteps, lstm_inp_drop, lstm_rec_drop),
            incomming
        )
        return recur

    return rec


def merge_predictions(intervals: List[Interval], predictions: np.ndarray) \
        -> np.ndarray:
    """
    :param intervals: intervals (non-inclusive on the right side)
    :param predictions:
    :return:
    >>> randints = np.random.randint(0, 1000, size=20)
    >>> intervals = sorted([sorted(randints[i:i+2])
    ...                     for i in range(0, len(randints), 2)])
    >>> maxlen = max(end - start for start, end in intervals)
    >>> predictions = np.zeros((len(intervals), maxlen), dtype=float)
    >>> for i, (start, end) in enumerate(intervals):
    ...     predictions[i, :end-start] = np.random.uniform(0, 1, size=end-start)
    >>> manual = [[] for _ in range(max(chain.from_iterable(intervals)))]
    >>> for (i, (start, end)), pred in zip(enumerate(intervals), predictions):
    ...     for j, value in zip(range(start, end), pred[:end-start]):
    ...         manual[j].append(value)
    >>> means_man =  np.array([np.mean(values) if values else np.nan
    ...                       for values in manual])
    >>> means_func = merge_predictions(intervals, predictions)
    >>> nan_man = np.isnan(means_man)
    >>> nan_func = np.isnan(means_func)
    >>> (nan_man == nan_func).all()
    True
    >>> (means_man[~nan_man].round(3) == means_func[~nan_func].round(3)).all()
    True
    """
    # the intervals are half-inclusive and zero-indexed
    length = max(chain.from_iterable(intervals))
    buckets = np.zeros(length, dtype=np.float64)
    nsamples = np.zeros(length, dtype=np.int32)
    for (start, end), pred in zip(intervals, predictions):
        # `predictions` are zero-padded – we must remove the padded tail
        sample_length = end - start
        buckets[start:end] += pred[:sample_length]
        nsamples[start:end] += np.ones(sample_length, dtype=np.int32)
    with np.errstate(divide='ignore', invalid="ignore"):
        return buckets / nsamples


if __name__ == "__main__":
    raise RuntimeError
