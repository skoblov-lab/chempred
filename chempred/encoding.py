"""



"""


from typing import Text, Sequence, Tuple

import numpy as np

from chempred.chemdner import Annotation
from chempred.intervals import Intervals, Interval
from chempred.util import Vocabulary

MAXCHAR = 127


class EncodingError(ValueError):
    pass


def encode_sample(sample: Intervals[Interval], text: Text, vocab: Vocabulary,
                  dtype=np.int32) -> Tuple[np.ndarray, np.ndarray]:
    # TODO tests
    """
    Encode text characters at each position of the sample
    :param text: the complete text from which the sample was drawn
    :param sample: sample's span
    :param dtype: output data type; it must be an integral numpy dtype
    :return: an integer array
    """
    if not np.issubdtype(dtype, np.int):
        raise ValueError("`dtype` must be integral")
    encoded = np.fromiter(map(ord, text[sample.start:sample.stop]),
                          dtype, len(sample))
    encoded[encoded > MAXCHAR] = MAXCHAR
    return encoded


def encode_sample_annotation(sample: Intervals[Interval],
                             annotation: Annotation,
                             default=0, dtype=np.int32) \
        -> Tuple[np.ndarray, np.ndarray]:
    # TODO tests
    """
    :param sample: a sequence of intervals
    :param annotation: annotated intervals
    :param default: default value for unannotated regions
    :param dtype: output data type; it must be an integral numpy dtype
    :return: an integer array
    """
    if not np.issubdtype(dtype, np.int):
        raise ValueError("`dtype` must be integral")
    # encode character annotations
    fst, lst = sample[0], sample[-1]
    annotated_intervals = annotation.within(fst.start, lst.stop, partial=True)
    char_annotation = np.repeat([default], len(sample.span)).astype(dtype)
    offset = fst.stop
    for iv in annotated_intervals:
        char_annotation[iv.start-offset:iv.stop-offset] = iv.data
    # encode token annotations
    token_annotation = np.repeat([default], len(sample)).astype(dtype)
    for i, iv in enumerate(sample):
        anno = set(anno_iv.data
                   for anno_iv in annotation.within(iv.start, iv.stop))
        if len(anno) > 1:
            raise EncodingError("ambiguous annotations")
        if anno:
            token_annotation[i] = anno.pop()
    return char_annotation, token_annotation


if __name__ == "__main__":
    raise RuntimeError
