"""

"""


from typing import List, Tuple, Mapping, Callable
from chempred.chemdner import Annotation, Interval
from itertools import chain

import numpy as np


PADDING_VAL = 0
MAXCHAR = 127
Sampler = Callable[[int, List[Annotation]], List[List[Annotation]]]


def slide(center: int, width: int, lastpos: int, flanking: bool) \
        -> List[Interval]:
    """
    :param center:
    :param width:
    :param lastpos:
    :param flanking:
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
    >>> slide(10, 3, 10, False)
    []
    >>> slide(10, 3, 10, True)
    [(7, 10)]
    >>> slide(0, 10, 10, False) == slide(0, 10, 10, True) == [(0, 10)]
    True
    >>> slide(0, 11, 10, False) == slide(0, 11, 10, True) == []
    True
    """
    first = max(center - (width if flanking else width - 1), 0)
    last = min(center + 2 if flanking else center + 1, lastpos - width + 1)
    return [(i, i + width) for i in range(first, last)]


def make_sampler(width: int, maxlen: int, flanking: bool) \
        -> Sampler:
    """
    :type width: int
    :param width: the desired number of context tokens to sample; e.g. for a
    positive token at index `i` and window `3` the function will try to create
    samples [(i-2, i-1, i), (i-1, i, i+1), (i, i+1, i+2)] if flanking == False
    :param maxlen: the maximum length of a sample in unicode codes.
    :type flanking: bool
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
    def sampler(target: int, annotations: List[Annotation]) \
            -> List[List[Annotation]]:
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
    """
    Sample context windows around positive tokens.
    :return: (list[sampled windows], list[failed target words]);
    failed target words – positive words with no samples of length <= `maxlen`
    """
    samples = [sampler(i, annotations) for i in targets]
    failed_targets = [annotations[i] for i, samples in zip(targets, samples)
                      if not samples]
    return list(chain.from_iterable(samples)), failed_targets


def encode_samples(text: str, samples: List[List[Annotation]], length: int) \
        -> np.ndarray:
    encoded_samples = np.zeros((len(samples), length), dtype=np.int64)
    for i, sample in enumerate(samples):
        start, end = sample[0].start, sample[-1].end
        sample_length = end - start
        if sample_length > length:
            raise ValueError("Sample exceeds the length limit")
        encoded_samples[i, :length] = list(map(ord, text[start:end]))
    encoded_samples[encoded_samples > (MAXCHAR - 1)] = MAXCHAR
    return encoded_samples.astype(np.int16)


def encode_classes(mapping: Mapping[str, int], samples: List[List[Annotation]],
                   length: int,) -> np.ndarray:
    if 0 in mapping.values():
        raise ValueError("0 is an invalid class mapping")
    encoded_classes = np.zeros((len(samples), length), dtype=np.int16)
    try:
        for i, sample in enumerate(samples):
            offset = sample[0].start
            codes = (mapping[anno.cls] for anno in sample)
            intervals = ((anno.start - offset, anno.end - offset)
                         for anno in sample)
            for code, (start, end) in zip(codes, intervals):
                encoded_classes[i, start:end] = code
    except KeyError:
        raise ValueError("Missing a class in the mapping")

    return encoded_classes


if __name__ == "__main__":
    raise RuntimeError
