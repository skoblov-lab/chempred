from numbers import Integral
from typing import Sequence, NamedTuple, Text, Iterable, Tuple, List, \
    Mapping, Pattern, Callable
from functools import reduce
from itertools import chain
from pyrsistent import PVector, pvector
import re

import numpy as np
import spacy
from fn import F

from sciner import intervals

OTHER = "OTHER"
TITLE = "T"
BODY = "A"
ClassMapping = Mapping[Text, Integral]
ClassifiedInterval = intervals.Interval[Integral]
Annotation = Sequence[ClassifiedInterval]
AbstractAnnotation = NamedTuple("AbstractAnnotation", [("id", int),
                                                       ("title", Annotation),
                                                       ("body", Annotation)])
Abstract = NamedTuple("Abstract",
                      [("id", int), ("title", Text), ("body", Text)])
spacy_tokeniser = F(spacy.load("en")) >> (map, lambda tk: tk.text)


class AnnotationError(ValueError):
    pass


NO_WS_PATT = re.compile("\S+")
WORD_LIKE_PATT = re.compile(r"[\w]+|[^\s\w]")


def flatten_aligned_pair(pair: Tuple[Abstract, AbstractAnnotation]) \
        -> List[Tuple[int, Text, Text, Sequence[intervals.Interval]]]:
    # TODO tests
    """
    :return: list[(abstract id, source, text, annotation)]
    """
    (abstract_id, title, body), (anno_id, title_anno, body_anno) = pair
    if abstract_id != anno_id:
        raise AnnotationError("Abstract ids do not match")
    return [(abstract_id, TITLE, title, title_anno),
            (abstract_id, BODY, body, body_anno)]


def parse_mapping(classmaps: Iterable[str]) -> ClassMapping:
    """
    :param classmaps:
    :return:
    >>> classmaps = ["a:1", "b:1", "c:2"]
    >>> parse_mapping(classmaps) == dict(a=1, b=1, c=2)
    True
    """
    try:
        return {cls: int(val)
                for cls, val in [classmap.split(":") for classmap in classmaps]}
    except ValueError as err:
        raise AnnotationError("Badly formatted mapping: {}".format(err))


def parser(tokeniser: Callable[[str], Iterable[str]], text: Text) -> np.ndarray:

    def mark_boundaries(boundaries: PVector, token: str):
        if not boundaries:
            return boundaries.append(intervals.Interval(0, len(token), token))
        prev = boundaries[-1]
        start = prev.stop
        stop = start + len(token)
        return boundaries.append(intervals.Interval(start, stop, token))

    if not text:
        return np.array([])
    all_tk = re.compile("\S+|\s+")
    ws = re.compile("\s")
    tokens = all_tk.findall(text)
    fine_grained = chain.from_iterable(
        tokeniser(tk) if not ws.match(tk) else [tk] for tk in tokens)
    intervals_ = reduce(mark_boundaries, fine_grained, pvector())
    ws_less = (iv for iv in intervals_ if not ws.match(iv.data))
    return np.array([iv.reload(i) for i, iv in enumerate(ws_less)])


if __name__ == "__main__":
    raise RuntimeError
