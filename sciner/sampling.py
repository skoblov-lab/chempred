from typing import Sequence, Iterator, Optional

import numpy as np

from sciner.encoding import EncodingError
from sciner.intervals import Interval, extract_intervals


class AmbiguousAnnotation(EncodingError):
    pass


def annotate_sample(annotation: np.ndarray, nlabels: int,
                    sample: Sequence[Interval],
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
    if nlabels > 1:
        tk_annotations = extract_intervals(annotation, sample)
        encoded_token_anno = np.zeros((len(sample), nlabels), dtype=np.int32)
        for i, tk_anno in enumerate(tk_annotations):
            encoded_token_anno[i, tk_anno] = 1
    else:
        tk_annotations = map(np.unique, extract_intervals(annotation, sample))
        encoded_token_anno = np.zeros(len(sample), dtype=np.int32)
        for i, tk_anno in enumerate(tk_annotations):
            positive_anno = tk_anno[tk_anno > 0]
            if len(positive_anno) > 1:
                raise AmbiguousAnnotation(
                    "ambiguous annotation: {}".format(positive_anno))
            encoded_token_anno[i] = positive_anno[0] if positive_anno else 0
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


def sample_length(sample: Sequence[Interval]) -> int:
    # TODO docs
    return 0 if not len(sample) else sample[-1].stop - sample[0].start


def sample_span(sample: Sequence[Interval]) -> Optional[Interval]:
    return Interval(sample[0].start, sample[-1].stop) if len(sample) else None


if __name__ == "__main__":
    raise RuntimeError
