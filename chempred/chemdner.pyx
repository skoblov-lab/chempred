"""

"""

from collections import Iterator
from itertools import groupby
import operator as op
import re

import numpy as np

from cpython cimport bool as pybool_t
from libc.stdint cimport int64_t


TOKEN_PATT = re.compile("\S+")

cdef str OTHER = "OTHER"
cdef str TITLE = "T"
cdef str BODY = "A"


def read_abstracts(str path):
    """
    Read chemdner abstracts
    :type path: str
    :rtype: list[(int, str, str)]
    :return: [(abstract id, title, body)]
    """
    with open(path) as buffer:
        parsed_buffer = (line.strip().split("\t") for line in buffer)
        return [(int(abstract_n), title, abstact)
                for abstract_n, title, abstact in parsed_buffer]


def read_annotations(str path):
    """
    Read chemdner annotations
    :type path: str
    :rtype: list[tuple[int, list[(str, int, int, str, str)]]
    :return: [(abstract_id, (source, start, end, text, type))]
    """
    with open(path) as buffer:
        parsed_buffer = (lines.strip().split("\t") for lines in buffer)
        groups = groupby(parsed_buffer, op.itemgetter(0))
        return  [(int(id_), [(src, int(start), int(end), text, type_)
                             for _, src, start, end, text, type_ in anno])
                 for id_, anno in groups]


cpdef list tokenise(str text):
    """
    Extract tokens separated by any white-space character.
    :param text: 
    :return: [(start, end, text)]
    :rtype: list[(int, int, str)]
    """
    cdef list matched = [m.span() for m in TOKEN_PATT.finditer(text)]
    return [(start, end, text[start:end]) for start, end in matched]


cdef inline pybool_t intersects(tuple token, tuple annotation):
    """
    Check whether a token intersects with an annotated region
    :param token: (start, end, text)
    :type token: (int, int, str)
    :param annotation: (source, start, end, text, type)
    :type annotation: (str, int, int, str, str)
    :rtype: bool
    """
    cdef int token_start = token[0]
    cdef int token_end = token[1]
    cdef int anno_start = annotation[1]
    cdef int anno_end = annotation[2]
    return (token_start <= anno_start < token_end) or (anno_start <= token_start < anno_end)


cdef inline pybool_t appears_after(tuple token, tuple annotation):
    """
    Check whether a token appears after an annotated region
    :param token: (start, end, text)
    :type token: (int, int, str)
    :param annotation: (source, start, end, text, type)
    :type annotation: (str, int, int, str, str)
    :rtype: bool
    """
    cdef int token_start = token[0]
    cdef int anno_end = annotation[2]
    return token_start >= anno_end


cpdef list annotate_text(str text, str source, list annotations):
    """
    Tokenise text and annotate the tokens
    :type text: str
    :type source: str
    :param annotations: list[(source, start, end, text, type)]
    :type annotations: list[(str, int, int, str, str)]
    :return: list[(source, start, end, text, type)]
    :rtype: list[(str, int, int, str, str)]
    """
    cdef list tokens = tokenise(text)
    cdef list types = [OTHER] * (len(tokens))
    anno_it = iter(annotations)
    cdef tuple anno = next(anno_it, None)
    cdef int anno_size = len(annotations)
    for i in range(len(tokens)):
        if not anno:
            break
        if appears_after(tokens[i], anno):
            anno = next(anno_it, None)
        if anno and intersects(tokens[i], anno):
            types[i] = anno[-1]
    return [(source, start, end, text, type_)
            for (start, end, text), type_ in zip(tokens, types)]


cpdef str mask_annotated_regions(str text, list annotations):
    """
    Replace annotated regions with white space
    :type text: str
    :param annotations: (source, start, end, text, type)
    :type annotations: (str, int, int, str, str)
    :return: masked `text`
    :rtype: str
    """
    charcodes = np.array([ord(ch) for ch in text], dtype=int)
    mask = np.zeros(len(text), dtype=bool)
    for _, start, end, *_ in annotations:
        mask[start:end] = True
    charcodes[mask] = ord(" ")
    return "".join([chr(num) for num in charcodes])


cpdef list annotate_abstract(tuple abstract, tuple abstract_annotation,
                             pybool_t guided=True):
    """
    :param abstract: (abstract id, title, body)
    :type abstract: (int, str, str)
    :param abstract_annotation: (abstract id, [(src, start, end, type)])
    :type abstract_annotation: (int, [(str, int, int, str)])
    :type guided: bool
    :param guided: use guided (masked) tokenisation; uses `abstract_annotation`
    to guide tokenisation (preserves correct segmentation of annotated tokens)
    :return: a sorted list of annotated tokens, i.e. [(source, start, end, text, type)]
    :rtype: list[(str, int, int, str, str)]
    """
    # check whether abstract ids match
    cdef int anno_id = abstract_annotation[0]
    cdef list annotations = abstract_annotation[1]
    if not abstract[0] == anno_id:
        raise ValueError("Abstract IDs do not match")

    # separate annotations
    cdef list title_anno = []
    cdef list body_anno = []
    groups = groupby(sorted(annotations), op.itemgetter(0))
    for source, group in groups:
        if source == TITLE:
            title_anno = list(group)
        elif source == BODY:
            body_anno = list(group)
        else:
            raise ValueError("Unknown source")

    # mask text if needed
    cdef str title = mask_annotated_regions(abstract[1], title_anno) if guided else abstract[1]
    cdef str body = mask_annotated_regions(abstract[2], body_anno) if guided else abstract[2]

    # tokenise, annotate and sort tokens
    cdef list title_tokens = annotate_text(title, TITLE, title_anno)
    cdef list body_tokens = annotate_text(body, BODY, body_anno)
    if guided:
        title_tokens.extend(title_anno)
        body_tokens.extend(body_anno)
    return sorted(title_tokens) + sorted(body_tokens)


def align_abstracts_and_annotations(list abstacts, list abstact_annotations):
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
    cdef dict id_anno_mapping = dict(abstact_annotations)
    cdef list abstact_annotations_extented = [
        (abst[0], id_anno_mapping.get(abst[0], [])) for abst in abstacts
    ]
    return zip(abstacts, abstact_annotations_extented)
