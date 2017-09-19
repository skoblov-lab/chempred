import operator as op
from itertools import repeat, chain
from typing import Sequence

import numpy as np

from sciner.util import Interval


def combine_predictions_average(samples: Sequence[Sequence[Interval[int]]],
                                predictions: Sequence[np.ndarray]) -> np.ndarray:
    # the intervals are half-inclusive and zero-indexed
    """
    :param spans: intervals (non-inclusive on the right side)
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
    sorted_samples = [sorted(s, key=lambda x: x.start) for s in samples]
    spans = [Interval(s[0].data, s[-1].data+1) for s in sorted_samples]
    length = max(span.stop for span in spans)
    shape = (length, *predictions[0].shape[1:])
    buckets = np.zeros(shape, dtype=np.float64)
    nsamples = np.zeros(length, dtype=np.int32)
    for span, pred in zip(spans, predictions):
        # `predictions` are zero-padded – we must remove the padded tail
        pad_start = span.stop - span.start
        buckets[span.start:span.stop] += pred[:pad_start]
        nsamples[span.start:span.stop] += 1
    slice_ = (slice(length), *repeat(None, predictions[0].ndim - 1))
    with np.errstate(divide='ignore', invalid="ignore"):
        return buckets / op.getitem(nsamples, slice_)


def combine_predictions_max(samples: Sequence[Sequence[Interval[int]]],
                            predictions: Sequence[np.ndarray]) -> np.ndarray:
    # the intervals are half-inclusive and zero-indexed
    """
    :param spans: intervals (non-inclusive on the right side)
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
    sorted_samples = [sorted(s, key=lambda x: x.start) for s in samples]
    spans = [Interval(s[0].data, s[-1].data+1) for s in sorted_samples]
    length = max(span.stop for span in spans)
    buckets = np.zeros(length, dtype=np.float64)
    for span, pred in zip(spans, predictions):
        # `predictions` are zero-padded – we must remove the padded tail
        pad_start = span.stop - span.start
        current = buckets[span.start:span.stop]
        new = pred[:pad_start, 1]
        buckets[span.start:span.stop] = np.vstack((current, new)).max(0)
    return buckets


if __name__ == "__main__":
    raise RuntimeError
