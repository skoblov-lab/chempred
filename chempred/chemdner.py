"""

"""

from typing import NamedTuple, List, Tuple, Iterator
from collections import deque
from functools import reduce
from itertools import groupby
import operator as op
import re

from pyrsistent import pvector
import numpy as np


TOKEN_PATT = re.compile("\S+")
OTHER = "OTHER"
TITLE = "T"
BODY = "A"


Interval = Tuple[int, int]
Annotation = NamedTuple("Annotation", [("source", str), ("start", int),
                                       ("end", int), ("text", str),
                                       ("cls", str)])
Annotations = Tuple[int, List[Annotation], List[Annotation]]
Abstract = Tuple[int, str, str]


def read_abstracts(path: str) -> List[Tuple[int, str, str]]:
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
        return [(int(abstract_n), title, abstact)
                for abstract_n, title, abstact in parsed_buffer]


def read_annotations(path: str) \
        -> List[Tuple[int, List[Annotation], List[Annotation]]]:
    """
    Read chemdner annotations
    :return: list[(abstract_id, title annotations, body_annotations])]
    >>> path = "testdata/annotations.txt"
    >>> anno = read_annotations(path)
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
    def pack_annotations(anno: List[Tuple[str, str, str, str, str, str]]) \
            -> List[Annotation]:
        return [Annotation(src, int(start), int(end), text, cls)
                for _, src, start, end, text, cls in anno]

    def split_sources(annotations: List[Annotation]) \
            -> Tuple[List[Annotation], List[Annotation]]:
        """
        Separate title and body annotations
        :return: (title annotations, body annotations)
        """
        source_groups = {source: list(anno) for source, anno in
                         groupby(annotations, lambda x: x.source)}
        title_anno = source_groups.get(TITLE, [])
        body_anno = source_groups.get(BODY, [])
        return title_anno, body_anno

    with open(path) as buffer:
        parsed_buffer = (lines.strip().split("\t") for lines in buffer)
        abstract_groups = groupby(parsed_buffer, op.itemgetter(0))
        abstract_anno = [(int(id_), split_sources(pack_annotations(anno)))
                         for id_, anno in abstract_groups]
        return [(id_, title_anno, body_anno)
                for id_, (title_anno, body_anno) in abstract_anno]


def mask_annotated_regions(text: str, annotations: List[Annotation]) -> str:
    """
    Replace annotated regions with white space
    >>> annotations = [Annotation(TITLE, 5, 10, "t", "a"),
    ...                Annotation(TITLE, 15, 20, "t", "b")]
    >>> text = "a" * 20
    >>> masked_text = mask_annotated_regions(text, annotations)
    >>> masked_regions = [masked_text[anno.start:anno.end]
    ...                   for anno in annotations]
    >>> all(reg == " " * (anno.end - anno.start)
    ...     for anno, reg in zip(annotations, masked_regions))
    True
    >>> mask_annotated_regions(text, [Annotation(TITLE, 20, 21, "t", "a")])
    Traceback (most recent call last):
    ...
    ValueError: An annotated region is out of boundaries
    """
    charcodes = np.array([ord(ch) for ch in text], dtype=int)
    textlen = len(text)
    mask = np.zeros(textlen, dtype=bool)
    for _, start, end, *_ in annotations:
        if end > textlen:
            raise ValueError("An annotated region is out of boundaries")
        mask[start:end] = True
    charcodes[mask] = ord(" ")
    return "".join([chr(num) for num in charcodes])


def classify_intervals(intervals: List[Interval],
                       annotations: List[Annotation]) -> List[str]:
    """
    Return class annotations for `intervals`. Any interval has class OTHER
    unless it intersects an annotated region, then it gets that region's class
    :return: a list of token classes
    >>> annotations = [Annotation(TITLE, 5, 10, "t", "a"),
    ...                Annotation(TITLE, 15, 20, "t", "b")]
    >>> intervals = [(2, 6), (9, 11), (12, 15), (15, 21)]
    >>> classify_intervals(intervals, annotations)
    ['a', 'a', 'OTHER', 'b']
    """

    def intersects(interval: Interval, annotation: Annotation) -> bool:
        start_a, end_a = interval
        start_b, end_b = annotation.start, annotation.end
        return (start_a <= start_b < end_a) or (start_b <= start_a < end_b)

    def appears_after(interval: Interval, annotation: Annotation) -> bool:
        return interval[0] >= annotation.end

    anno = deque(annotations)

    def accumulate_classes(classes: pvector, interval: Interval) -> pvector:
        if anno and appears_after(interval, anno[0]):
            anno.popleft()
        if anno and intersects(interval, anno[0]):
            return classes.append(anno[0].cls)
        return classes.append(OTHER)

    return list(reduce(accumulate_classes, intervals, pvector()))


def annotate_text(text: str, annotations: List[Annotation], source: str,
                  guided: bool=True) -> List[Annotation]:
    """
    Tokenise text and classify each token
    :param text: text to parse
    :param annotations: chemdner annotations
    :param source: text source type (i.e. title or body)
    :param guided: use guided (masked) tokenisation; uses `annotation`
    to guide tokenisation (preserves correct segmentation of annotated tokens)
    :return: a sorted list of annotated tokens
    >>> id_abstract, title, _ = read_abstracts("testdata/abstracts.txt")[-1]
    >>> id_anno, t_anno, _ = read_annotations("testdata/annotations.txt")[-1]
    >>> id_abstract == id_anno
    True
    >>> guided_anno = annotate_text(title, t_anno, TITLE, guided=True)
    >>> t_anno == [anno for anno in guided_anno if anno.cls != OTHER]
    True
    >>> unguided_anno = annotate_text(title, t_anno, TITLE, guided=False)
    >>> t_anno != [anno for anno in unguided_anno if anno.cls != OTHER]
    True
    >>> [anno.text for anno in guided_anno if anno.cls != OTHER]
    ['Mercury', 'nitric oxide']
    >>> [anno.text for anno in unguided_anno if anno.cls != OTHER]
    ['Mercury', 'nitric', 'oxide']
    """
    # mask text if needed
    text_ = mask_annotated_regions(text, annotations) if guided else text
    # tokenise, annotate and sort tokens
    intervals = [m.span() for m in TOKEN_PATT.finditer(text_)]
    if any(src != source for src, *_ in annotations):
        raise ValueError("source mismatch")
    classes = classify_intervals(intervals, annotations)
    text_anno = [Annotation(source, start, end, text_[start:end], cls)
                 for (start, end), cls in zip(intervals, classes)]
    return sorted(text_anno + annotations if guided else text_anno,
                  key=lambda x: (x.start, x.end))


def align_abstracts_and_annotations(
        abstacts: List[Tuple[int, str, str]],
        abstract_annotations: List[Tuple[int, List[Annotation], List[Annotation]]]) \
        -> Iterator[Tuple[Tuple[int, str, str],
                          Tuple[int, List[Annotation], List[Annotation]]]]:
    """
    Align abstracts and annotations (i.e. match abstract ids)
    :param abstacts: parsed abstracts (e.g. produces by `read_abstracts`)
    :param abstract_annotations: parsed annotations (e.g. produces by
    `read_annotations`)
    :return: Iterator[(parsed abstract, parsed annotation)]
    """
    id_anno_mapping = dict(abstract_annotations)
    aligned_anno = (
        (abst[0], id_anno_mapping.get(abst[0], ([], []))) for abst in abstacts
    )
    flattened_aligned_anno = ((id_, title, body)
                              for id_, (title, body) in aligned_anno)
    return zip(abstacts, flattened_aligned_anno)


def flatten_aligned_pair(pair: Tuple[Abstract, Annotations]) \
        -> List[Tuple[int, str, List[Annotation]]]:
    """
    :return: list[(abstract id, text, token annotations)]
    """
    (abstract_id, title, body), (anno_id, title_anno, body_anno) = pair
    if abstract_id != anno_id:
        raise ValueError("Abstract ids do not match")
    return [(abstract_id, title, title_anno), (abstract_id, body, body_anno)]


if __name__ == "__main__":
    raise RuntimeError
