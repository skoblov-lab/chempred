"""

"""


from typing import List, Tuple, Iterator, Mapping, Iterable, Callable, Any
from chempred.chemdner import Annotation, Interval, TITLE, BODY, OTHER
from itertools import chain


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


def generate_samples(text: str, annotations: List[Annotation],
                     positive: Mapping[str, int], sampler: Sampler):
    """
    Sample context windows around positive tokens.
    :type annotations: list[(str, int, int, str, str)]
    :param annotations: list[(source, start, end, text, type)], e.g.
    the output from `chemdner.tokenise_abstracts`
    :type positive: dict[str, int]
    :param positive: a mapping from type strings to integer-encoded classes;
    any unspecified type string is mapped into 0, hence the mapping must contain
    no keys with zero values
    :return: (list[sample boundaries], list[samples], list[sample classes],
    list[failed centered words]); failed centered words – positive words with no
     samples of length
    <= `maxlen` with at least `mincontext` context tokens
    :rtype: (list[(int, int)], list[list[int]], list[list[int]], list[str])
    """
    pass


if __name__ == "__main__":
    raise RuntimeError
