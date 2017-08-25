"""



"""


from typing import List, Mapping, Tuple, Text

import numpy as np

from chempred.intervals import Intervals, Interval
from chempred.chemdner import Annotation


MAXCHAR = 127


def encode_sample_chars(text: Text, span: Interval, dtype=np.int32) \
        -> np.ndarray:
    # TODO tests
    """
    Encode text characters at each position of the sample
    :param text: the complete text from which the sample was drawn
    :param span: sample's span
    :param dtype: output data type; it must be an integral numpy dtype
    :return: an integer array
    """
    if not np.issubdtype(dtype, np.int):
        raise ValueError("`dtype` must be integral")
    encoded = np.fromiter(map(ord, text[span.start:span.stop]),
                          dtype, len(span))
    encoded[encoded >= MAXCHAR] = MAXCHAR
    return encoded


def encode_sample_classes(span: Interval, annotation: Annotation, default=0,
                          dtype=np.int32) -> np.ndarray:
    # TODO tests
    # TODO docs
    """
    :param span: sample's span
    :param dtype: output data type; it must be an integral numpy dtype
    :return: an integer array

    """
    if not np.issubdtype(dtype, np.int):
        raise ValueError("`dtype` must be integral")
    intervals = annotation.within(span.start, span.stop)
    encoded = np.repeat([default], len(span)).astype(dtype)
    offset = span.start
    for interval in intervals:
        encoded[interval.start-offset:interval.end-offset] = interval.data
    return encoded


if __name__ == "__main__":
    raise RuntimeError
