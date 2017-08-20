"""



"""


from typing import List, Mapping, Tuple

import numpy as np

from chempred.chemdner import Annotation

MAXCHAR = 127


def encode_sample_chars(text: str, sample: List[Annotation], dtype=np.int32) \
        -> np.ndarray:
    # TODO tests
    """
    Encode text characters at each position of the sample
    :param text: the complete text from which the sample was drawn
    :param sample: a list of annotations
    :param dtype: output data type; it must be an integral numpy dtype
    :return: an integer array
    """
    if not np.issubdtype(dtype, np.int):
        raise ValueError("`dtype` must be integral")
    start, end = sample[0].start, sample[-1].end
    length = end - start
    encoded = np.fromiter(map(ord, text[start:end]), dtype, length)
    encoded[encoded >= MAXCHAR] = MAXCHAR
    return encoded


def encode_sample_classes(mapping: Mapping[str, int], sample: List[Annotation],
                          dtype=np.int32) \
        -> np.array:
    # TODO tests
    """
    Encode classes at each position of the sample. The function maps any class
    not in `mapping` into 0.
    :param mapping: a mapping from string classes into integers
    :param sample: a list of annotations
    :param dtype: output data type; it must be an integral numpy dtype
    :return: an integer array
    >>> sample = [Annotation(None, 0, 5, None, "a"),
    ...           Annotation(None, 6, 10, None, "b")]
    >>> (np.unique(encode_sample_classes(dict(a=1, b=2), sample), return_counts=True)[1]
    ...  == np.array([1, 5, 4])).all()
    True
    >>> (np.unique(encode_sample_classes(dict(a=1), sample), return_counts=True)[1]
    ...  == np.array([5, 5])).all()
    True
    >>> (np.unique(encode_sample_classes(dict(), sample), return_counts=True)[1]
    ...  == np.array([10])).all()
    True
    """
    if not np.issubdtype(dtype, np.int):
        raise ValueError("`dtype` must be integral")
    offset = sample[0].start
    length = sample[-1].end - offset
    encoded = np.zeros(length, dtype=dtype)
    for _, start, end, _, cls in sample:
        encoded[start-offset:end-offset] = mapping.get(cls, 0)
    return encoded


def encode_annotation(mapping: Mapping[str, int], anno: Annotation) \
        -> Tuple[np.ndarray, np.ndarray]:
    # TODO docs
    # TODO tests
    length = anno.end - anno.start
    txt = np.fromiter(map(ord, anno.text), dtype=np.int32, count=length)
    txt[txt > MAXCHAR] = MAXCHAR
    cls = np.array([mapping.get(anno.cls, 0)] * length, dtype=np.int32)
    return txt, cls


if __name__ == "__main__":
    raise RuntimeError
