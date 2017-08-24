"""

Parsers, preprocessors and type annotations for the chemdner dataset.

"""

import operator as op
import re
from typing import List, Tuple, Iterator, Text, Iterable, NamedTuple, Mapping
from numbers import Integral
from itertools import groupby

from fn import F

from chempred.annotation import Annotation, Interval


TOKEN_PATT = re.compile("\S+")
OTHER = "OTHER"
TITLE = "T"
BODY = "A"

AbstractAnnotation = NamedTuple(
    "AbstractAnnotation", [
        ("id", int),
        ("title", Annotation[Interval]),
        ("body", Annotation[Interval])
    ]
)
Abstract = NamedTuple("Abstract",
                      [("id", int), ("title", Text), ("body", Text)])


def read_abstracts(path: Text) -> List[Abstract]:
    """
    Read chemdner abstracts
    :return: list[(abstract id, title, body)]
    >>> path = "testdata/abstracts.txt"
    >>> abstracts = read_abstracts(path)
    >>> ids = {21826085, 22080034, 22080035, 22080037}
    >>> all(id_ in ids for id_, *_ in abstracts)
    True
    """
    with open(path) as buffer:
        parsed_buffer = (line.strip().split("\t") for line in buffer)
        return [Abstract(int(abstract_n), title, abstract)
                for abstract_n, title, abstract in parsed_buffer]


def read_annotations(path: Text, mapping: Mapping[Text, Integral],
                     default: Integral=0) \
        -> List[AbstractAnnotation]:
    # TODO more tests
    """
    Read chemdner annotations
    :param path: path to a CHEMDNER-formatted annotation files
    :param mapping: a class mapping
    :param default: default class integer value for out-of-mapping classes
    >>> path = "testdata/annotations.txt"
    >>> anno = read_annotations(path, {"SYSTEMATIC": 1}, 0)
    >>> ids = {21826085, 22080034, 22080035, 22080037}
    >>> all(id_ in ids for id_, *_ in anno)
    True
    >>> nonempty_anno = [id_ for id_, title, _ in anno if title]
    >>> nonempty_anno
    [22080037]
    >>> [len(title) for _, title, _ in anno]
    [0, 0, 0, 2]
    >>> [len(body) for _, _, body in anno]
    [1, 6, 9, 5]
    """
    def wrap_interval(record: Tuple[str, str, str, str, str, str]) -> Interval:
        _, _, start, stop, _, cls = record
        return Interval(int(start), int(stop), mapping.get(cls, default))

    with open(path) as buffer:
        parsed_lines = (l.strip().split("\t") for l in buffer)
        # separate abstracts
        abstract_groups = groupby(parsed_lines, op.itemgetter(0))
        # separate parts (title and body)
        part_groups = ((id_, groupby(group, op.itemgetter(1)))
                       for id_, group in abstract_groups)
        wrapper = F(map, wrap_interval) >> list
        mapped_parts = ((id_, {part: wrapper(recs) for part, recs in parts})
                        for id_, parts in part_groups)
        return [AbstractAnnotation(int(id_),
                                   Annotation(parts.get(TITLE, [])),
                                   Annotation(parts.get(BODY, [])))
                for id_, parts in mapped_parts]


def tokenise(text: Text) -> Annotation[Interval]:
    # TODO tests
    """
    Tokenise text and classify each token
    :param text: text to parse
    :return: a sorted list of annotated tokens
    """
    # mask text if needed
    intervals = [m.span() for m in TOKEN_PATT.finditer(text)]
    return Annotation(Interval(start, end, text[start:end])
                      for start, end in intervals)


def align_abstracts_and_annotations(abstracts: Iterable[Abstract],
                                    annotations: Iterable[AbstractAnnotation]) \
        -> Iterator[Tuple[Abstract, AbstractAnnotation]]:
    # TODO tests
    """
    Align abstracts and annotations (i.e. match abstract ids)
    :param abstracts: parsed abstracts (e.g. produces by `read_abstracts`)
    :param annotations: parsed annotations (e.g. produces by `read_annotations`)
    :return: Iterator[(parsed abstract, parsed annotation)]
    """
    def empty(id_: int) -> AbstractAnnotation:
        return AbstractAnnotation(id_, Annotation([]), Annotation([]))

    anno_mapping = {anno.id: anno for anno in annotations}
    return ((abstract, anno_mapping.get(abstract.id, empty(abstract.id)))
            for abstract in abstracts)


def flatten_aligned_pair(pair: Tuple[Abstract, AbstractAnnotation]) \
        -> List[Tuple[int, Text, Text, Annotation[Interval]]]:
    # TODO tests
    """
    :return: list[(abstract id, source, text, annotation)]
    """
    (abstract_id, title, body), (anno_id, title_anno, body_anno) = pair
    if abstract_id != anno_id:
        raise ValueError("Abstract ids do not match")
    return [(abstract_id, TITLE, title, title_anno),
            (abstract_id, BODY, body, body_anno)]


if __name__ == "__main__":
    raise RuntimeError
