"""



"""


from typing import Sequence, Iterable, Text, Mapping, Union, Iterator, Optional

import numpy as np

from sciner.util import Interval, extract_intervals

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


def annotate_sample(annotation: np.ndarray, sample: Sequence[Interval],
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


def sample_windows(intervals: Sequence[Interval], window: int) \
        -> Iterator[Sequence[Interval]]:
    # TODO update docs
    # TODO test
    """
    Sample windows using a sliding window approach. Sampling windows start at
    the beginning of each interval in `intervals`
    :param intervals: a sequence (preferable a numpy array) of interval objects
    :param window: sampling window width in tokens
    """
    samples = (
        iter([intervals]) if len(intervals) <= window else
        (intervals[i:i+window] for i in range(len(intervals)-window+1))
    )
    return samples


def sample_length(sample: Sequence[Interval]) -> int:
    # TODO docs
    return 0 if not len(sample) else sample[-1].stop - sample[0].start


def sample_span(sample: Sequence[Interval]) -> Optional[Interval]:
    return Interval(sample[0].start, sample[-1].stop) if len(sample) else None


if __name__ == "__main__":
    raise RuntimeError


