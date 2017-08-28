"""



"""


from typing import List, Mapping, Tuple, Text

import numpy as np

from chempred.intervals import Intervals, Interval
from chempred.chemdner import Annotation


MAXCHAR = 127


def encode_text(text: Text, span: Interval, dtype=np.int32) \
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
    encoded[encoded > MAXCHAR] = MAXCHAR
    return encoded


def encode_annotation(span: Interval, annotation: Annotation, default=0,
                      dtype=np.int32) -> np.ndarray:
    """
    :param span: sample's span
    :param annotation: annotated intervals
    :param default: default value for unannotated regions
    :param dtype: output data type; it must be an integral numpy dtype
    :return: an integer array
    >>> anno = Intervals([Interval(3, 10, 1), Interval(12, 15, 2)])
    >>> samples = [Interval(0, 10), Interval(2, 10), Interval(2, 20),
    ...            Interval(11, 16)]
    >>> encoded = [encode_annotation(sample, anno) for sample in samples]
    >>> all(len(s) == len(e) for s, e in zip(samples, encoded))
    True
    >>> all(sum(e) == sum(len(i) * i.data for i in anno.within(s.start, s.stop))
    ...     for s, e in zip(samples, encoded))
    True
    >>> encode_annotation(Interval(10, 13), anno).sum() == 2
    True
    >>> encode_annotation(Interval(10, 13), anno)[-1] == 2
    True
    """
    if not np.issubdtype(dtype, np.int):
        raise ValueError("`dtype` must be integral")
    intervals = annotation.within(span.start, span.stop, partial=True)
    encoded = np.repeat([default], len(span)).astype(dtype)
    offset = span.start
    for interval in intervals:
        encoded[interval.start-offset:interval.stop-offset] = interval.data
    return encoded


if __name__ == "__main__":
    raise RuntimeError
