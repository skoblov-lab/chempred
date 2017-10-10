import operator as op
from itertools import groupby
from typing import Tuple, Text, Sequence, Optional

import numpy as np

from sciner.intervals import Interval, span, extract
from sciner.preprocessing.encoding import EncodingError

ProcessedSample = Tuple[int, Text, Sequence[Interval], Sequence[Text],
                        Optional[np.ndarray]]


def group(ids, sources, *args):
    """
    Group args by id and source
    :param ids:
    :param sources:
    :param args:
    :return:
    """
    records = zip(ids, sources, *args)
    id_groups = groupby(records, op.itemgetter(0))
    return [[list(grp) for _, grp in src_grps] for src_grps in
            (groupby(list(grp), op.itemgetter(1)) for _, grp in id_groups)]


def annotate_sample(nlabels: int, annotation: np.ndarray,
                    sample: Sequence[Interval], dtype=np.int32) -> np.ndarray:
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


if __name__ == "__main__":
    raise RuntimeError
