"""

Data preprocessing routines

"""


from typing import List, Sequence, Iterator, Iterable

from chempred.intervals import Interval, Intervals, T


def sample(intervals: Intervals[T], window: int, dropcropped=True) \
        -> Iterator[Intervals[T]]:
    """
    Sample windows using a sliding window approach. Sampling windows start at
    the beginning of each interval in `intervals`
    :param intervals: intervals to sample
    :param window: sampling window length
    >>> from itertools import starmap
    >>> ranges = [(1, 2), (3, 4), (5, 6), (7, 8), (9, 30)]
    >>> regions = list(starmap(Interval, ranges))
    >>> intervals = Intervals(regions)
    >>> samples = [[(i.start, i.stop) for i in sample]
    ...            for sample in sample(intervals, 10)]
    >>> samples == [ranges[i:-1] for i in range(len(ranges)-1)]
    True
    """
    start_points = (interval.start for interval in intervals)
    return filter(bool, (intervals.within(start, start+window, dropcropped)
                         for start in start_points))


def find_unsampled(samples: Iterable[Intervals],
                   targets: Sequence[Interval[T]]) -> List[Interval[T]]:
    """
    Find intervals (`targets`) missing in `samples`
    :param samples: sampled intervals (see `sample`)
    :param targets: intervals to search
    :return: a list of missing intervals
    >>> from itertools import starmap
    >>> ranges = [(1, 2), (3, 4), (5, 6), (7, 8), (9, 30)]
    >>> regions = list(starmap(Interval, ranges))
    >>> intervals = Intervals(regions)
    >>> samples = sample(intervals, 10)
    >>> find_unsampled(samples, intervals)
    [Interval(start=9, stop=30)]
    """
    return [target for target in targets
            if not any(target in sample_ for sample_ in samples)]


if __name__ == "__main__":
    raise RuntimeError
