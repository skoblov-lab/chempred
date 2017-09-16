import operator as op
from itertools import repeat
from typing import Sequence

import numpy as np

from sciner.util import Interval


def merge_predictions(intervals: Sequence[Interval],
                      predictions: Sequence[np.ndarray]) \
        -> np.ndarray:
    """
    :param intervals: intervals (non-inclusive on the right side)
    :param predictions:
    :return:
    # TODO update tests
    # >>> randints = np.random.randint(0, 1000, size=20)
    # >>> intervals = sorted([tuple(sorted(randints[i:i+2]))
    # ...                     for i in range(0, len(randints), 2)])
    # >>> maxlen = max(end - start for start, end in intervals)
    # >>> predictions = np.zeros((len(intervals), maxlen), dtype=float)
    # >>> for i, (start, end) in enumerate(intervals):
    # ...     predictions[i, :end-start] = np.random.uniform(0, 1, size=end-start)
    # >>> manual = [[] for _ in range(max(chain.from_iterable(intervals)))]
    # >>> for (i, (start, end)), pred in zip(enumerate(intervals), predictions):
    # ...     for j, value in zip(range(start, end), pred[:end-start]):
    # ...         manual[j].append(value)
    # >>> means_man =  np.array([np.mean(values) if values else np.nan
    # ...                       for values in manual])
    # >>> means_func = merge_predictions(intervals, predictions)
    # >>> nan_man = np.isnan(means_man)
    # >>> nan_func = np.isnan(means_func)
    # >>> (nan_man == nan_func).all()
    # True
    # >>> (means_man[~nan_man].round(3) == means_func[~nan_func].round(3)).all()
    # True
    """
    # the intervals are half-inclusive and zero-indexed
    length = max(iv.stop for iv in intervals)
    shape = (length, *predictions[0].shape[1:])
    buckets = np.zeros(shape, dtype=np.float64)
    nsamples = np.zeros(length, dtype=np.int32)
    for iv, pred in zip(intervals, predictions):
        # `predictions` are zero-padded – we must remove the padded tail
        true_length = iv.stop - iv.start
        buckets[iv.start:iv.stop] += pred[:true_length]
        nsamples[iv.start:iv.stop] += 1
    slice_ = (slice(length), *repeat(None, predictions[0].ndim-1))
    with np.errstate(divide='ignore', invalid="ignore"):
        return buckets / op.getitem(nsamples, slice_)


if __name__ == "__main__":
    raise RuntimeError
