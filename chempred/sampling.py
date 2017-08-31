"""

Data sampling routines

"""


from typing import List, Sequence, Iterator, Iterable, Text, Tuple

import numpy as np

from chempred import chemdner, util, encoding
from chempred.intervals import Interval, Intervals, T


def sample_windows(intervals: Intervals[Interval[T]], width: int, partial=False) \
        -> Iterator[Intervals[Interval[T]]]:
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
    return filter(bool, (intervals.within(start, start + width, partial)
                         for start in start_points))


def find_unsampled(samples: Iterable[Intervals],
                   targets: Sequence[Interval[T]]) -> List[Interval[T]]:
    # TODO make it smarter, i.e. handle partially covered targets
    """
    Find intervals (`targets`) missing in `samples`
    :param samples: sampled intervals (see `sample`)
    :param targets: intervals to search
    :return: a list of missing intervals
    >>> from itertools import starmap
    >>> ranges = [(1, 2), (3, 4), (5, 6), (7, 8), (9, 30)]
    >>> regions = list(starmap(Interval, ranges))
    >>> intervals = Intervals(regions)
    >>> samples = sample_windows(intervals, 10)
    >>> find_unsampled(samples, intervals)
    [Interval(start=9, stop=30)]
    """
    return [target for target in targets
            if not any(target in sample_ for sample_ in samples)]


def process_text(text: Text, annotation: Intervals[chemdner.ClassifiedInterval],
                 tokeniser: util.Tokeniser, width: int, minlen: int,
                 default: int=0, annotated_only=True) \
        -> Tuple[List[Intervals], np.ndarray, np.ndarray, np.ndarray]:
    # TODO docs
    """
    :param text:
    :param annotation:
    :param width: context window width (in charactes)
    :param minlen: minimum sample span
    :param default: default class encoding
    :return: samples, encoded text, encoded annotations, padding mask
    >>> import random
    >>> from chempred.intervals import Interval, Intervals
    >>> anno = Intervals([Interval(4, 10, 1), Interval(20, 25, 2)])
    >>> text = "".join(random.choice("abc ") for _ in range(len(anno.span)+9))
    >>> samples, text_e, cls_e, mask = process_text(text, anno,
    ...                                             util.tokenise, 10, 5)
    >>> text_e.shape == cls_e.shape == mask.shape
    True
    >>> len(samples) == len(text_e) == len(cls_e) == len(mask)
    True
    """
    # TODO return failures
    tokenised_text = tokeniser(text)

    samples = [sample for sample in sample_windows(tokenised_text, width)]
    if annotated_only:
        # remove samples with no annotated regions and insufficient length
        passing = [sample for sample in samples if len(sample.span) >= minlen
                   and annotation.covers(sample.span)]
    else:
        # remove samples with insufficient length
        passing = [sample for sample in samples if len(sample.span) >= minlen]

    if not passing:
        return [], np.array([]), np.array([]), np.array([])

    encoded_text = [
        encoding.encode_characters(text, sample.span) for sample in passing
    ]
    encoded_classes = [
        encoding.encode_annotation(sample.span, annotation, default=default)
        for sample in passing
    ]

    joined_text, text_mask = util.join(encoded_text, width)
    joined_cls, cls_mask = util.join(encoded_classes, width)
    # sanity check
    assert (text_mask == cls_mask).all()
    return passing, joined_text, joined_cls, text_mask


if __name__ == "__main__":
    raise RuntimeError