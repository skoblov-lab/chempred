"""



"""


from typing import Sequence, Iterable, Text, Mapping

import numpy as np

from chempred.util import Interval, sample_span, \
    extract_intervals

MAXCHAR = 127
MAXCLS = 255


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


def annotate_sample(sample: Sequence[Interval], annotation: np.ndarray,
                    dtype=np.int32) -> np.ndarray:
    # TODO update docs
    # TODO tests
    """
    :param text: the complete text from which the sample was drawn
    :param sample: a sequence of Intervals
    :param dtype: output data type; it must be an integral numpy dtype
    :return: encoded annotation
    """
    if not np.issubdtype(dtype, np.int):
        raise EncodingError("`dtype` must be integral")
    span = sample_span(sample)
    if span is None:
        raise EncodingError("The sample is empty")
    if span.stop > len(annotation):
        raise EncodingError("The annotation doesn't fully cover the sample")
    token_annotations = map(np.unique, extract_intervals(annotation, sample))
    encoded_token_anno = np.zeros(len(sample), dtype=np.int32)
    for i, tk_anno in enumerate(token_annotations):
        positive_anno = tk_anno[tk_anno > 0]
        if len(positive_anno) > 1:
            raise EncodingError("ambiguous annotation")
        encoded_token_anno[i] = positive_anno[0] if positive_anno else 0
    return encoded_token_anno


def encode_sample(sample: Sequence[Interval], text: Text,
                  encoder: Mapping[Text, np.ndarray]) \
        -> np.ndarray:
    # TODO update docs
    # TODO tests
    """
    :param text: the complete text from which the sample was drawn
    :param sample: a sample of intervals
    :return: (encoded tokens, token anno), (encoded characters, character anno)
    """
    if not len(sample):
        raise EncodingError("The sample is empty")
    tokens = extract_intervals(text, sample)
    return np.array(encoder[tk] for tk in tokens)



if __name__ == "__main__":
    raise RuntimeError
