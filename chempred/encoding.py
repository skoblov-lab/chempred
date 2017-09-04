"""



"""


from typing import Text, Sequence, Tuple
from numbers import Integral

import numpy as np

from chempred.chemdner import Annotation
from chempred.intervals import Intervals, Interval
from chempred.util import Vocabulary

MAXCHAR = 127
MAXCLS = 255

EncodedSample = Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]

class EncodingError(ValueError):
    pass


def encode_annotation(annotations: Intervals, size: int) -> np.ndarray:
    # TODO update docs
    """
    Default class is 0.
    :param annotations:
    :param size:
    :return:
    """
    encoded_anno = np.zeros(size, dtype=np.uint8)
    for anno in annotations:
        if anno.stop > size:
            raise EncodingError("annotation `size` is insufficient")
        cls = anno.data
        if not 0 <= cls <= MAXCLS:
            raise EncodingError("class codes must be in [0, {}]".format(MAXCLS))
        encoded_anno[anno.start:anno.stop] = anno.data
    return encoded_anno


def encode_sample(sample: Intervals, text: Text, vocab: Vocabulary,
                  annotation: np.ndarray, dtype=np.int32) -> EncodedSample:
    # TODO update docs
    # TODO tests
    """
    :param text: the complete text from which the sample was drawn
    :param sample: sample's span
    :param dtype: output data type; it must be an integral numpy dtype
    :return: (encoded tokens, token anno), (encoded characters, character anno)
    """
    if not np.issubdtype(dtype, np.int):
        raise ValueError("`dtype` must be integral")
    # encode tokens
    tokens = (text[iv.start:iv.stop].lower() for iv in sample)
    encoded_tokens = np.fromiter(map(vocab.get, tokens), dtype, len(sample))
    token_anno = np.zeros(len(sample), dtype=np.int32)
    for i, iv in enumerate(sample):
        tk_anno = np.unique(annotation[iv.start:iv.stop])
        positive_anno = tk_anno[tk_anno > 0]
        if len(positive_anno) > 1:
            raise EncodingError("ambiguous annotation")
        token_anno[i] = positive_anno[0] if positive_anno else 0
    # encode characters
    span = sample.span
    characters = text[span.start:span.stop]
    encoded_characters = np.fromiter(map(ord, characters), dtype, len(span))
    encoded_characters[encoded_characters > MAXCHAR] = MAXCHAR
    char_anno = annotation[span.start:span.stop]
    return encoded_tokens, token_anno, encoded_characters, char_anno


if __name__ == "__main__":
    raise RuntimeError
