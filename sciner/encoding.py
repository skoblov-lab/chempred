"""



"""


from typing import Sequence, Iterable, Text, Mapping, Union

import numpy as np

from sciner.intervals import Interval

MAXCHAR = 127
MAXCLS = 255


Encoder = Union[Mapping[Text, np.ndarray],
                Mapping[Sequence[Text], np.ndarray]]


class EncodingError(ValueError):
    pass


def encode_annotation(annotations: Iterable[Interval], size: int) -> np.ndarray:
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


def encode_tokens(encoder: Encoder, tokens: Iterable[Text], dtype=np.float32) \
        -> np.ndarray:
    # TODO update docs
    # TODO tests
    """
    :param text: the complete text from which the sample was drawn
    :param sample: a sample of intervals
    :return: (encoded tokens, token anno), (encoded characters, character anno)
    """
    tokens_ = list(tokens)
    if not len(tokens_):
        raise EncodingError("The tokens is empty")
    try:
        return np.array(encoder[tokens_]).astype(dtype)
    except (TypeError, KeyError, ValueError):
        return np.array([encoder[tk] for tk in tokens_]).astype(dtype)


def encode_characters(characters: Text) -> np.ndarray:
    codes = np.fromiter(map(ord, characters), np.int32, len(characters))
    return np.clip(codes, 0, MAXCHAR)


if __name__ == "__main__":
    raise RuntimeError
