"""

Data sampling routines

"""


from typing import List, Iterator

from chempred.intervals import Interval, Intervals, T


def sample_windows(intervals: Intervals[Interval[T]], width: int,
                   partial=False) -> Iterator[Intervals[Interval[T]]]:
    # TODO update docs
    """
    Sample windows using a sliding window approach. Sampling windows start at
    the beginning of each interval in `intervals`
    :param intervals: intervals to sample
    :param width: sampling window width
    >>> from itertools import starmap
    >>> ranges = [(1, 2), (3, 4), (5, 6), (7, 8), (9, 30)]
    >>> regions = list(starmap(Interval, ranges))
    >>> intervals = Intervals(regions)
    >>> sample_spans = [[(i.start, i.stop) for i in sample]
    ...            for sample in sample_windows(intervals, 10)]
    >>> sample_spans == [ranges[i:-1] for i in range(len(ranges)-1)]
    True
    """
    start_points = (interval.start for interval in intervals)
    samples = (intervals.within(start, start+width, partial)
               for start in start_points)
    return (sample for sample in samples if sample)


if __name__ == "__main__":
    raise RuntimeError
