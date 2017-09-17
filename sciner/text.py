from numbers import Integral
from typing import Sequence, NamedTuple, Text, Iterable, Tuple, List, \
    Mapping, Pattern

import numpy as np

from sciner.util import Interval

OTHER = "OTHER"
TITLE = "T"
BODY = "A"
ClassMapping = Mapping[Text, Integral]
ClassifiedInterval = Interval[Integral]
Annotation = Sequence[ClassifiedInterval]
AbstractAnnotation = NamedTuple("AbstractAnnotation", [("id", int),
                                                       ("title", Annotation),
                                                       ("body", Annotation)])
Abstract = NamedTuple("Abstract",
                      [("id", int), ("title", Text), ("body", Text)])


class AnnotationError(ValueError):
    pass


def flatten_aligned_pair(pair: Tuple[Abstract, AbstractAnnotation]) \
        -> List[Tuple[int, Text, Text, Sequence[Interval]]]:
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


def parse_text(text: Text, pattern: Pattern) -> np.ndarray:
    # TODO tests
    """
    Parse text into a sequence (numpy array) of intervals. Each interval
    contains its index in the `data` attribute
    :param text: text to parse
    :param pattern: token pattern
    :return: a sorted array of matches intervals
    """
    try:
        intervals = [m.span() for m in pattern.finditer(text)]
        return np.array([Interval(start, end, i)
                         for i, (start, end) in enumerate(intervals)])
    except TypeError:
        raise TypeError("`{}` is not a valid unicode string".format(repr(text)))


if __name__ == "__main__":
    raise RuntimeError


