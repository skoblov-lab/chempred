"""

"""

from collections import Iterator, deque
from functools import reduce
from itertools import groupby
import operator as op
import re

from pyrsistent import pvector
import numpy as np


TOKEN_PATT = re.compile("\S+")
OTHER = "OTHER"


def read_abstracts(path):
    """
    Read chemdner abstracts
    :type path: str
    :rtype: list[(int, str, str)]
    :return: list[(abstract id, title, body)]
    """
    with open(path) as buffer:
        parsed_buffer = (line.strip().split("\t") for line in buffer)
        return [(int(abstract_n), title, abstact)
                for abstract_n, title, abstact in parsed_buffer]


def read_annotations(path):
    """
    Read chemdner annotations
    :type path: str
    :rtype: list[tuple[int, list[(str, int, int, str, str)]]
    :return: list[(abstract_id, (source, start, end, text, class))]
    """
    with open(path) as buffer:
        parsed_buffer = (lines.strip().split("\t") for lines in buffer)
        groups = groupby(parsed_buffer, op.itemgetter(0))
        return  [(int(id_), [(src, int(start), int(end), text, type_)
                             for _, src, start, end, text, type_ in anno])
                 for id_, anno in groups]


def mask_annotated_regions(text, annotations):
    """
    Replace annotated regions with white space
    :type text: str
    :param annotations: list[(source, start, end, text, class)]
    :type annotations: list[(str, int, int, str, str)]
    :return: masked `text`
    :rtype: str
    """
    charcodes = np.array([ord(ch) for ch in text], dtype=int)
    mask = np.zeros(len(text), dtype=bool)
    for _, start, end, *_ in annotations:
        mask[start:end] = True
    charcodes[mask] = ord(" ")
    return "".join([chr(num) for num in charcodes])


def annotate_intervals(intervals, annotations):
    """
    :type intervals: list[(int, int)]
    :param intervals: list[(start, end)]
    :type annotations: list[(str, int, int, str, str)]
    :param annotations: list[(source, start, end, text, class)]
    :rtype: list[str]
    :return: a list of token classes
    """

    def intersects(interval, annotation):
        start_a, end_a = interval
        start_b, end_b = annotation[1:3]
        return (start_a <= start_b < end_a) or (start_b <= start_a < end_b)

    def appears_after(interval, annotation):
        start_a = interval[0]
        end_b = annotation[2]
        return start_a >= end_b

    anno = deque(annotations)

    def accumulate_classes(classes, interval):
        if anno and appears_after(interval, anno[0]):
            anno.popleft()
        if anno and intersects(interval, anno[0]):
            return classes.append(anno[0][-1])
        return classes.append(OTHER)

    return list(reduce(accumulate_classes, intervals, pvector()))


def tokenise(text, annotations, source, guided=True):
    """
    :param text: text
    :type text: str
    :param annotations: [(source, start, end, text, class)]
    :type annotations: list[(str, int, int, str, str)]
    :type source: str
    :param source: text source type (i.e. title or body)
    :type guided: bool
    :param guided: use guided (masked) tokenisation; uses `annotation`
    to guide tokenisation (preserves correct segmentation of annotated tokens)
    :return: a sorted list of annotated tokens, i.e. [(source, start, end, text, class)]
    :rtype: list[(str, int, int, str, str)]
    """
    # mask text if needed
    text_ = mask_annotated_regions(text, annotations) if guided else text
    # tokenise, annotate and sort tokens
    intervals = [m.span() for m in TOKEN_PATT.finditer(text_)]
    if any(src != source for src, *_ in annotations):
        raise ValueError("source mismatch")
    classes = annotate_intervals(intervals, annotations)
    tokens = [(source, start, end, text_[start:end], cls) for
              (start, end), cls in zip(intervals, classes)]
    return sorted(tokens + annotations if guided else tokens,
                  key=lambda x: x[1:3])


def align_abstracts_and_annotations(abstacts, abstact_annotations):
    """
    Align abstracts and annotations (i.e. match abstract ids)
    :param abstacts: parsed abstracts (e.g. produces by `read_abstracts`)
    :type abstacts: list[(int, str, str)]
    :param abstact_annotations: parsed annotations (e.g. produces by
    `read_annotations`)
    :type abstact_annotations: list[tuple[int, list[(str, int, int, str, str)]]
    :rtype: Iterator[(list[(int, str, str)], list[tuple[int, list[(str, int, int, str, str)]])]
    :return: Iterator[(parsed abstract, parsed annotation)]
    """
    id_anno_mapping = dict(abstact_annotations)
    abstact_annotations_extented = [(abst[0], id_anno_mapping.get(abst[0], []))
                                    for abst in abstacts]
    return zip(abstacts, abstact_annotations_extented)
