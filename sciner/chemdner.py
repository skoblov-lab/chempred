"""

Parsers, preprocessors and type annotations for the chemdner dataset.

"""

import operator as op
from itertools import groupby
from numbers import Integral
from typing import List, Tuple, Text

from fn import F

from sciner.text import TITLE, BODY, AbstractAnnotation, Abstract, ClassMapping
from sciner.util import Interval


def parse_abstracts(path: Text) -> List[Abstract]:
    """
    Read chemdner abstracts
    :return: list[(abstract id, title, body)]
    >>> path = "testdata/abstracts.txt"
    >>> abstracts = parse_abstracts(path)
    >>> ids = {21826085, 22080034, 22080035, 22080037}
    >>> all(id_ in ids for id_, *_ in abstracts)
    True
    """
    with open(path) as buffer:
        parsed_buffer = (line.strip().split("\t") for line in buffer)
        return [Abstract(int(abstract_n), title, abstract)
                for abstract_n, title, abstract in parsed_buffer]


def parse_annotations(path: Text, mapping: ClassMapping, default: Integral=None) \
        -> List[AbstractAnnotation]:
    # TODO log empty annotations
    # TODO more tests
    """
    Read chemdner annotations
    :param path: path to a CHEMDNER-formatted annotation files
    :param mapping: a class mapping
    :param default: default class integer value for out-of-mapping classes; if
    None is given, objects with out-of-mapping classes are discarded
    >>> path = "testdata/annotations.txt"
    >>> anno = parse_annotations(path, {"SYSTEMATIC": 1}, 0)
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
    def wrap_interval(record: Tuple[str, str, str, str, str, str]) \
            -> Interval:
        _, _, start, stop, _, cls = record
        value = mapping.get(cls, default)
        return None if value is None else Interval(int(start), int(stop), value)

    with open(path) as buffer:
        parsed_lines = (l.strip().split("\t") for l in buffer)
        # separate abstracts
        abstract_groups = groupby(parsed_lines, op.itemgetter(0))
        # separate parts (title and body)
        part_groups = ((id_, groupby(group, op.itemgetter(1)))
                       for id_, group in abstract_groups)
        # filter zero-length intervals and `None`s
        wrapper = F(map, wrap_interval) >> (filter, bool) >> list
        mapped_parts = ((id_, {part: wrapper(recs) for part, recs in parts})
                        for id_, parts in part_groups)
        return [AbstractAnnotation(int(id_),
                                   list(parts.get(TITLE, [])),
                                   list(parts.get(BODY, [])))
                for id_, parts in mapped_parts]


if __name__ == "__main__":
    raise RuntimeError
