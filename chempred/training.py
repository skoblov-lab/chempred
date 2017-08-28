import glob
from contextlib import contextmanager
from itertools import chain, starmap
from typing import Tuple, List, Mapping, Set, Union, Sequence, Text, Iterator

import numpy as np
import operator as op
import os
import re
import shutil
from fn import F

from chempred import intervals
from chempred import chemdner
from chempred import encoding
from chempred import sampling
from chempred import util
from chempred import chemdner

# Data = Union[Mapping[str, str], Sequence[str]]
Sample = List[intervals.Interval]
# Failure = Tuple[int, ClassifiedRegion]


def process_text(text: Text,
                 annotation: intervals.Intervals[chemdner.ClassifiedInterval],
                 width: int, minlen: int, default: int=0) \
        -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    :param text:
    :param annotation:
    :param width: context window width (in charactes)
    :param minlen: minimum sample span
    :param default: default class encoding
    :return: encoded text samples, encoded annotations, padding mask
    >>> import random
    >>> from chempred.intervals import Interval, Intervals
    >>> anno = Intervals([Interval(4, 10, 1), Interval(20, 25, 2)])
    >>> text = "".join(random.choice("abc ") for _ in range(len(anno.span)+9))
    >>> text_e, cls_e, mask = process_text(text, anno, 10, 5)
    >>> text_e.shape == cls_e.shape == mask.shape
    True
    """
    # TODO return failures
    tokenised_text = util.tokenise(text)
    sample_spans = [sample.span for sample in
                    sampling.sample_windows(tokenised_text, width)]
    # remove samples with no annotated regions and insufficient length
    passing = [span for span in sample_spans
               if len(span) >= minlen and annotation.covers(span)]
    encoded_text = [
        encoding.encode_text(text, span) for span in passing
    ]
    encoded_classes = [
        encoding.encode_annotation(span, annotation, default=default)
        for span in passing
    ]
    if not encoded_text:
        return np.array([]), np.array([]), np.array([])

    joined_text, text_mask = util.join(encoded_text, width)
    joined_cls, cls_mask = util.join(encoded_classes, width)
    # sanity check
    assert (text_mask == cls_mask).all()
    return joined_text, joined_cls, text_mask


def process_data(abstracts: List[chemdner.Abstract],
                 abstract_annotations: List[chemdner.AbstractAnnotation],
                 window_width: int, minlen: int, default: int=0) \
        -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    # TODO update docs
    # TODO tests
    """
    :param abstracts:
    :param abstract_annotations:
    :param window_width: context window width (in charactes)
    :param minlen: minimum sample span
    :param default: default class encoding
    :return: encoded and padded text, encoded and padded classes, padding masks.
    >>> mapping = {"SYSTEMATIC": 1}
    >>> abstracts = chemdner.read_abstracts("testdata/abstracts.txt")
    >>> anno = chemdner.read_annotations("testdata/annotations.txt", mapping)
    >>> x, y, mask = (
    ...     process_data(abstracts, anno, 100, 50)
    ... )
    >>> x.shape[0] == y.shape[0] == mask.shape[0]
    True
    """
    # align pairs, flatten and remove texts with no annotations
    pairs = chemdner.align_abstracts_and_annotations(abstracts,
                                                     abstract_annotations)
    flattened_pairs = list(map(chemdner.flatten_aligned_pair, pairs))
    # sample windows
    titles = [title for title, _ in flattened_pairs]
    bodies = [body for _, body in flattened_pairs]
    # sanity  check
    assert [id_ for id_, *_ in titles] == [id_ for id_, *_ in bodies]
    processed_titles = [process_text(text, anno, window_width, minlen, default)
                        for *_, text, anno in titles]
    processed_bodies = [process_text(text, anno, window_width, minlen, default)
                        for *_, text, anno in bodies]
    # merge samples from titles and bodies
    texts = (F(map, F(map, op.itemgetter(0)))
             >> chain.from_iterable
             >> (filter, lambda x: len(x) > 0)
             >> tuple
             >> np.vstack)(
        [processed_titles, processed_bodies]
    )
    cls = (F(map, F(map, op.itemgetter(1)))
           >> chain.from_iterable
           >> (filter, lambda x: len(x) > 0)
           >> tuple
           >> np.vstack)(
        [processed_titles, processed_bodies]
    )
    masks = (F(map, F(map, op.itemgetter(2)))
             >> chain.from_iterable
             >> (filter, lambda x: len(x) > 0)
             >> tuple
             >> np.vstack)(
        [processed_titles, processed_bodies]
    )
    return texts, cls, masks



def pick_best(filenames: List[str]) -> Tuple[str, Tuple[int, float]]:
    # TODO import docs
    """
    >>> fnames = ["rootdir/name/weights-improvement-16-0.99.hdf5",
    ...           "rootdir/name/weights-improvement-25-0.99.hdf5",
    ...           "rootdir/name/weights-improvement-20-0.99.hdf5"]
    >>> pick_best(fnames)
    ('rootdir/name/weights-improvement-25-0.99.hdf5', (25, 0.99))
    """
    pattern = re.compile("([0-9]+)-([0-9.]+)\.hdf5")
    stats = (F(map, pattern.findall)
             >> (map, op.itemgetter(0))
             >> (starmap, lambda epoch, acc: (int(epoch), float(acc)))
             )(filenames)
    return max(zip(filenames, stats), key=op.itemgetter(1))


@contextmanager
def training(rootdir: str, name: str):
    # TODO docs
    """
    Initialise training temporary directories and cleanup upon completion
    :param rootdir:
    :param name:
    :return:
    """
    training_dir = os.path.join(rootdir, "{}-training".format(name))
    weights_template = os.path.join(training_dir,
                                    "{epoch:02d}-{val_acc:.3f}.hdf5")
    w_destination = os.path.join(rootdir, "{}-weights.hdf5".format(name))
    os.makedirs(training_dir)
    try:
        yield weights_template
    finally:
        all_weights = glob.glob(os.path.join(training_dir, "*.hdf5"))
        if all_weights:
            best_weights, stats = pick_best(all_weights)
            shutil.move(best_weights, w_destination)
        shutil.rmtree(training_dir)


if __name__ == "__main__":
    raise RuntimeError
