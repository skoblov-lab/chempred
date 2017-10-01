"""



"""


from typing import Sequence, Iterable, Text, Mapping, Union
from numbers import Integral
from itertools import groupby
import operator as op

import numpy as np
from fn import F

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


# def encode_entity_borders(step_annotation: Sequence[Integral]) -> np.ndarray:
#     grouped = groupby(enumerate(step_annotation), op.itemgetter(1))
#     positive_runs = (list(run) for cls, run in grouped if cls)
#     # col1 - start, col2 - end, col0 - other
#     position_types = np.zeros((len(step_annotation), 3), dtype=np.int32)
#     position_types[:, 0] = 1
#     for run in positive_runs:
#         first, _ = run[0]
#         last, _ = run[-1]
#         position_types[first, 1] = 1
#         position_types[last, 2] = 1
#     return position_types


def encode_entity_borders(step_annotation: Sequence[Integral]) -> np.ndarray:
    grouped = groupby(enumerate(step_annotation), op.itemgetter(1))
    positive_runs = (list(run) for cls, run in grouped if cls)
    border_positions = np.zeros(len(step_annotation), dtype=np.int32)
    for run in positive_runs:
        first, _ = run[0]
        last, _ = run[-1]
        border_positions[first] = 1
        border_positions[last] = 1
    return border_positions


if __name__ == "__main__":
    raise RuntimeError
