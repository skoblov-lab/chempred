from typing import Sequence, Iterator, Iterable, Callable, cast

import numpy as np

from sciner.encoding import EncodingError
from sciner.intervals import Interval, Intervals, extract, span

Sample = Sequence[Interval]
Sampler = Callable[[Sequence[Interval]], Iterable[Sample]]
Annotator = Callable[[Sample], np.ndarray]


class AmbiguousAnnotation(EncodingError):
    pass


def annotate_sample(nlabels: int, annotation: np.ndarray,
                    sample: Intervals, dtype=np.int32) -> np.ndarray:
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


def sample_windows(window: int, step: int, text_intervals: Intervals) \
        -> Iterator[Intervals]:
    # TODO update docs
    # TODO test
    """
    Sample windows using a sliding window approach. Sampling windows start at
    the beginning of each interval in `intervals`
    :param text_intervals: a sequence (preferable a numpy array) of interval objects
    :param window: sampling window width in tokens
    """
    if len(text_intervals) <= window:
        return iter([text_intervals])
    steps = list(range(0, len(text_intervals) - window + 1, step))
    if steps[-1] + window < len(text_intervals):
        steps.append(steps[-1] + step)
    return (text_intervals[i:i + window] for i in steps)


def sample_sentences(borders, text_intervals):
    # TODO docs
    # TODO tests
    if not len(text_intervals) or not len(borders):
        raise ValueError("empty intervals and/or borders")
    ends = iter(sorted(border.stop for border in borders))
    end = next(ends)
    samples = [[]]
    for iv in sorted(text_intervals, key=lambda x: x.start):
        if iv.stop <= end:
            samples[-1].append(iv)
        else:
            end = next(ends, None)
            if end is None:
                raise RuntimeError
            samples.append([iv])
    if len(samples) != len(borders):
        raise RuntimeError
    return samples


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
