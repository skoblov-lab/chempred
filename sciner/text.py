from numbers import Integral
from typing import Sequence, NamedTuple, Text, Iterable, Tuple, List, \
    Mapping, Callable, Optional
from functools import reduce
from itertools import chain
from pyrsistent import PVector, pvector
import re

import numpy as np
import spacy
from fn import F

from sciner import intervals
from sciner.util import flatmap

OTHER = "OTHER"
TITLE = "T"
BODY = "A"
ClassMapping = Mapping[Text, Integral]
ClassifiedInterval = intervals.Interval[Integral]
Annotation = Sequence[ClassifiedInterval]
SentenceBorders = intervals.Intervals

AbstractText = NamedTuple("Abstract",
                          [("id", int), ("title", Text), ("body", Text)])
AbstractAnnotation = NamedTuple("AbstractAnnotation", [("id", int),
                                                       ("title", Annotation),
                                                       ("body", Annotation)])
AbstractSentenceBorders = NamedTuple("AbstractSentenceBorders",
                                     [("id", int), ("title", SentenceBorders),
                                      ("body", SentenceBorders)])
Abstract = Tuple[AbstractText, Optional[AbstractAnnotation],
                 Optional[AbstractSentenceBorders]]
Record = Tuple[int, Text, Text, Optional[Annotation], Optional[SentenceBorders]]

spacy_tokeniser = (F(spacy.load("en").tokenizer) >>
                   (map, lambda tk: tk.text) >>
                   (flatmap, re.compile(r"[&/|]|[^&/|]+").findall))
fine_tokeniser = F(flatmap, re.compile(r"[\w]+|[^\s\w]").findall)


class AnnotationError(ValueError):
    pass


def flatten_abstract(abstract: Abstract) -> List[Record]:
    """
    :return: list[(abstract id, source, text, annotation)]
    """
    abstract_id, title, body = abstract[0]
    anno_id, title_anno, body_anno = abstract[1]
    borders_id, title_borders, body_borders = abstract[2]
    if abstract_id != anno_id:
        raise AnnotationError("Abstract ids do not match")
    return [(abstract_id, TITLE, title, title_anno, title_borders),
            (abstract_id, BODY, body, body_anno, body_borders)]


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


def tointervals(tokeniser: Callable[[str], Iterable[str]], text: Text) \
        -> intervals.Intervals:
    # TODO docs
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
