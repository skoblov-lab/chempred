from typing import Sequence, Iterator, Iterable, cast
import warnings
import sys

import numpy as np

from sciner.encoding import EncodingError
from sciner.intervals import Interval, extract, span


class AmbiguousAnnotation(EncodingError):
    pass


def annotate_sample(annotation: np.ndarray, nlabels: int,
                    sample: Sequence[Interval],
                    dtype=np.int32) -> np.ndarray:
    # TODO update docs
    """
    :param sample: a sequence of Intervals
    :param dtype: output data type; it must be an integral numpy dtype
    :return: encoded annotation
    """
    if not np.issubdtype(dtype, np.int):
        raise EncodingError("`dtype` must be integral")
    span_ = span(sample)
    if span_ is None:
        raise EncodingError("The sample is empty")
    if span_.stop > len(annotation):
        raise EncodingError("The annotation doesn't fully cover the sample")
    tk_annotations = extract(annotation, sample)
    encoded_token_anno = np.zeros((len(sample), nlabels), dtype=np.int32)
    for i, tk_anno in enumerate(tk_annotations):
        encoded_token_anno[i, tk_anno] = 1
    return encoded_token_anno


def sample_windows(intervals: Sequence[Interval], window: int, step: int=1) \
        -> Iterator[Sequence[Interval]]:
    # TODO update docs
    # TODO test
    """
    Sample windows using a sliding window approach. Sampling windows start at
    the beginning of each interval in `intervals`
    :param intervals: a sequence (preferable a numpy array) of interval objects
    :param window: sampling window width in tokens
    """
    if len(intervals) <= window:
        return iter([intervals])
    steps = list(range(0, len(intervals)-window+1, step))
    if steps[-1] + window < len(intervals):
        steps.append(steps[-1] + step)
    return (intervals[i:i+window] for i in steps)


def flatten_multilabel_annotation(sample_annotation: np.ndarray) -> np.ndarray:
    """
    Flatten 2D multi-label annotation. Time-steps are supposed to be encoded
    along the first axis, i.e. sample_annotation[i] encodes the i-th sample. If
    a time-step has two labels, one of which corresponds to the negative class
    (sample_annotation[:, 0]), the function returns the other (positive) label.
    If there are several positive labels for a time-step, the function raises
    an error
    :param sample_annotation: a 2D array
    :return:
    """
    time_steps = cast(Iterable[np.ndarray], sample_annotation)
    labels = [step.nonzero()[-1] for step in time_steps]
    positive_labels = [step_labels[step_labels > 0] for step_labels in labels]
    if any(len(pos_labels) > 1 for pos_labels in positive_labels):
            raise AmbiguousAnnotation("Couldn't flatten multilabel annotation"
                                      "due to ambiguity")
    return np.concatenate([pos_labels or [0] for pos_labels in positive_labels])


if __name__ == "__main__":
    raise RuntimeError
