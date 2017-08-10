"""

"""

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
    :param text: 
    :return: [start, end, text]
    :rtype: list[(int, int, str)]
    """
    cdef list matched = [m.span() for m in TOKEN_PATT.finditer(text)]
    return [(start, end, text[start:end]) for start, end in matched]


cdef inline pybool_t intersects(tuple token, tuple annotation):
    cdef int token_start = token[0]
    cdef int token_end = token[1]
    cdef int anno_start = annotation[1]
    cdef int anno_end = annotation[2]
    return (token_start <= anno_start < token_end) or (anno_start <= token_start < anno_end)


cdef inline pybool_t appears_after(tuple token, tuple annotation):
    """
    Does the token appears after the annotated region
    :param token: 
    :param annotation: 
    :return: 
    """
    cdef int token_start = token[0]
    cdef int anno_end = annotation[2]
    return token_start >= anno_end


cpdef list annotate_text(str text, str source, list annotations):
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


cpdef str mask_text(str text, list annotations):
    """
    Replace annotated regions with white space
    :param text: 
    :param annotations: (src, start, end, text, type)
    :return: 
    """
    charcodes = np.array([ord(ch) for ch in text], dtype=int)
    mask = np.zeros(len(text), dtype=bool)
    for _, start, end, *_ in annotations:
        mask[start:end] = True
    charcodes[mask] = ord(" ")
    return "".join([chr(num) for num in charcodes])


cpdef list annotate_abstract(tuple abstract, tuple abstract_annotation,
                             pybool_t guided=False):
    # TODO
    """
    :param abstract: (abstract id, title, body)
    :type abstract: (int, str, str)
    :param abstract_annotation: (abstract id, [(src, start, end, type)])
    :type abstract_annotation: (int, [(str, int, int, str)])
    :type guided: bool
    :param guided: use guided (masked) tokenisation
    :return: 
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
    cdef str title = mask_text(abstract[1], title_anno) if guided else abstract[1]
    cdef str body = mask_text(abstract[2], body_anno) if guided else abstract[2]

    # tokenise, annotate and sort tokens
    cdef list title_tokens = annotate_text(title, TITLE, title_anno)
    cdef list body_tokens = annotate_text(body, BODY, body_anno)
    if guided:
        title_tokens.extend(title_anno)
        body_tokens.extend(body_anno)
    return sorted(title_tokens) + sorted(body_tokens)



def pair(list abstacts, list abstact_annotations):
    # todo
    """
    :param abstacts:
    :param abstact_annotations:
    :return:
    """
    cdef dict id_anno_mapping = dict(abstact_annotations)
    cdef list abstact_annotations_extented = [
        (abst[0], id_anno_mapping.get(abst[0], [])) for abst in abstacts
    ]
    return abstact_annotations_extented
