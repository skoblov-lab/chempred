import glob
import operator as op
import os
import re
import shutil
import warnings
from contextlib import contextmanager
from itertools import chain, starmap
from typing import Tuple, List, Iterator, Iterable, Callable, Text, Sequence, \
    Optional, Mapping

import numpy as np
from fn import F
from sklearn.utils import class_weight

from sciner.encoding import AmbiguousAnnotation, annotate_sample, encode_annotation, \
    sample_windows
from sciner.text import AbstractAnnotation, Abstract, flatten_aligned_pair, BODY
from sciner.util import Interval, extract_intervals

ProcessedSample = Tuple[int, Text, Sequence[Interval], Sequence[Text], np.ndarray]


def process_pair(pair: Tuple[Abstract, AbstractAnnotation],
                 parser: Callable[[Text], Sequence[Interval]], window: int,
                 warn_overlapping: bool=False) \
        -> Iterator[ProcessedSample]:
    # TODO update docs
    # TODO tests
    """
    :param pair: abstract paired with its annotation
    :param window: context window width (in raw tokens)
    :return: Iterator[(text ids, sample sources (title or body),
    sampled intervals, sample tokens, sample annotations)]
    """
    def wrap_sample(id_, src, text, anno, sample):
        try:
            sample_text = extract_intervals(text, sample)
            sample_anno = annotate_sample(anno, sample)
            return id_, src, sample, sample_text, sample_anno
        except AmbiguousAnnotation as err:
            message = "Failed to annotate a sample in {}'s {} due {}".format(
                id_, "body" if src == BODY else "title", err)
            if not warn_overlapping:
                raise AmbiguousAnnotation(message)
            warnings.warn(message)
            return None

    ids, srcs, texts, annotations = zip(*flatten_aligned_pair(pair))
    sampled_ivs = (sample_windows(intervals, window)
                   for intervals in [parser(text) for text in texts])
    encoded_anno = [encode_annotation(anno, len(text))
                    for text, anno in zip(texts, annotations)]
    groups = zip(ids, srcs, texts, encoded_anno, sampled_ivs)
    processed = chain.from_iterable(
        [wrap_sample(id_, src, text, anno, s) for s in samples if len(s)]
        for id_, src, text, anno, samples in groups
    )
    return list(filter(bool, processed))


def flatten_processed_samples(processed_samples: Iterable[ProcessedSample]) \
    -> Tuple[Tuple[int], Tuple[Text], Tuple[Sequence[Interval]],
             Tuple[Sequence[Text]], Tuple[np.ndarray]]:
    ids, srcs, samples, tokens, annotations = zip(*processed_samples)
    return ids, srcs, samples, tokens, annotations


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
    Initialise temporary training directories and cleanup upon completion
    :param rootdir:
    :param name:
    :return:
    """
    training_dir = os.path.join(rootdir, "{}-training".format(name))
    weights_template = os.path.join(training_dir,
                                    "{epoch:02d}-{val_acc:.3f}.hdf5")
    destination = os.path.join(rootdir, "{}.hdf5".format(name))
    os.makedirs(training_dir)
    try:
        yield weights_template
    finally:
        all_checkpoints = glob.glob(os.path.join(training_dir, "*.hdf5"))
        if all_checkpoints:
            best_checkpoint, stats = pick_best(all_checkpoints)
            shutil.move(best_checkpoint, destination)
        shutil.rmtree(training_dir)


if __name__ == "__main__":
    raise RuntimeError
