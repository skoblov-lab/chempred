"""

Data preprocessing routines

"""


from typing import List, Sequence, Generator
from numbers import Integral
from itertools import chain

import numpy as np

from chempred.intervals import Interval, Intervals, T


def sample(intervals: Intervals[T], window: int, dropcropped=True) \
        -> Generator[Intervals[T]]:
    # TODO test
    # TODO docs
    """
    Sample windows.
    """
    start_points = (interval.start for interval in intervals)
    return (intervals.within(start, start+window, dropcropped)
            for start in start_points)


def find_unsampled(samples: List[Intervals], targets: Sequence[Interval[T]]) \
        -> List[Interval[T]]:
    # TODO test
    # TODO docs
    return [target for target in targets
            if not any(target in sample_ for sample_ in samples)]


if __name__ == "__main__":
    raise RuntimeError
