"""

Data preprocessing routines

"""


from itertools import chain
from typing import List, Tuple, Mapping, Callable, Set, Union

import numpy as np

from chempred.chemdner import Annotation, Interval


Sampler = Callable[[int, List[Annotation]], List[List[Annotation]]]


def slide(center: int, width: int, lastpos: int=None, flanking: bool=False) \
        -> List[Interval]:
    """
    Slide through the `center` index and return the resulting intervals
    :param center:
    :param width: sliding-window's length
    :param lastpos: the end position
    :param flanking: include the flanking windows
    >>> slide(-1, 3, 10, True)
    [(0, 3)]
    >>> slide(0, 3, 10, False)
    [(0, 3)]
    >>> slide(0, 3, 10, True)
    [(0, 3), (1, 4)]
    >>> slide(8, 3, 10, False)
    [(6, 9), (7, 10)]
    >>> slide(8, 3, 10, True)
    [(5, 8), (6, 9), (7, 10)]
    >>> slide(8, 3, None, True)
    [(5, 8), (6, 9), (7, 10), (8, 11)]
    >>> slide(10, 3, 10, False)
    []
    >>> slide(10, 3, 10, True)
    [(7, 10)]
    >>> slide(0, 10, 10, False) == slide(0, 10, 10, True) == [(0, 10)]
    True
    >>> slide(0, 11, 10, False) == slide(0, 11, 10, True) == []
    True
    """
    lastpos_ = lastpos or center + width
    first = max(center - (width if flanking else width - 1), 0)
    last = min(center + 2 if flanking else center + 1, lastpos_ - width + 1)
    return [(i, i + width) for i in range(first, last)]


def make_sampler(width: int, maxlen: int, flanking: bool) \
        -> Sampler:
    """
    Create a context-sampler.
    :param width: the desired number of context tokens to sample; e.g. for a
    positive token at index `i` and window `3` the function will try to create
    samples [(i-2, i-1, i), (i-1, i, i+1), (i, i+1, i+2)] if flanking == False
    :param maxlen: the maximum length of a sample in unicode codes.
    :param flanking: include windows adjacent to central words; note that
    each positive token is an independent central word
    >>> text = "abcdefjhijklmnop"
    >>> extractor = lambda x: text[x[0].start: x[-1].end]
    >>> annotations = [Annotation(None, 0, 4, None, None),
    ...                Annotation(None, 5, 8, None, None),
    ...                Annotation(None, 8, 10, None, None),
    ...                Annotation(None, 11, 12, None, None),
    ...                Annotation(None, 13, 16, None, None)]
    >>> sampler1 = make_sampler(3, len(text), flanking=False)
    >>> len(sampler1(0, annotations)) == 1
    True
    >>> len(sampler1(2, annotations)) == 3
    True
    >>> extractor(sampler1(0, annotations)[0]) == text[0:10]
    True
    >>> make_sampler(3, 8, flanking=False)(0, annotations)
    []
    >>> len(make_sampler(3, 8, flanking=False)(2, annotations)) == 2
    True
    >>> len(make_sampler(3, 7, flanking=False)(2, annotations)) == 1
    True
    >>> len(make_sampler(3, 6, flanking=False)(2, annotations)) == 0
    True
    """
    # TODO although flanking sampling is implemented, the feautre is
    # TODO deliberately disabled, because `sample_windows` (see below)
    # TODO doesn't handle it properly, yet
    if flanking:
        raise NotImplemented("`flanking` is deliberately disabled")

    def sampler(target: int, annotations: List[Annotation]) \
            -> List[List[Annotation]]:
        """
        Samples context windows around the target
        """
        windows = slide(target, width, len(annotations), flanking)
        samples = [annotations[first:last] for first, last in windows]
        lens = [annotations[last-1].end - annotations[first].start
                for first, last in windows]
        return [sample for sample, length in zip(samples, lens)
                if length <= maxlen]

    return sampler


def sample_windows(targets: List[int], annotations: List[Annotation],
                   sampler: Sampler) \
        -> Tuple[List[List[Annotation]], List[Annotation]]:
    # TODO tests
    """
    Sample context windows around positive tokens.
    :return: (list[sampled windows], list[failed target words]);
    failed target words – positive words with no samples of length <= `maxlen`
    """
    samples = [sampler(i, annotations) for i in targets]
    # TODO flanking=True breaks this check; see make_sampler
    failed_targets = [annotations[i] for i, samples in zip(targets, samples)
                      if not samples]
    return list(chain.from_iterable(samples)), failed_targets


def sample_targets(positive_classes: Union[Set[str], Mapping[str, int]],
                   annotations: List[Annotation], nonpos: int) -> List[int]:
    # TODO tests
    """
    Extract all positions of positive annotations and add `nonpos` randomly
    selected non-positive annotations
    :param positive_classes: a set of class strings to be considered positive
    :param annotations: a list of annotations (usually from a sample)
    :param nonpos: the maximum number of nonpositive targets to sample
    :return: extracted indices
    """
    indices = np.arange(len(annotations))
    mask = np.array([anno.cls in positive_classes for anno in annotations])
    positive = indices[mask]
    other = indices[~positive]
    nonpos_sample = np.random.choice(
        other, nonpos if nonpos <= len(other) else len(other), False)
    return list(positive) + list(nonpos_sample)


if __name__ == "__main__":
    raise RuntimeError
