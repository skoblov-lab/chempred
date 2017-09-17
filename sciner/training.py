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

ProcessedSample = Tuple[int, Text, Sequence[Interval], Sequence[Text],
                        Optional[np.ndarray]]


def process_pair(pair: Tuple[Abstract, AbstractAnnotation],
                 parser: Callable[[Text], Sequence[Interval]], window: int,
                 warn_overlapping: bool=False,
                 annotate: bool=True) \
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
            sample_anno = annotate_sample(anno, sample) if annotate else None
            return id_, src, sample, sample_text, sample_anno
        except AmbiguousAnnotation as err:
            message = "Failed to annotate a sample in {}'s {} due to {}".format(
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


def balance_class_weights(y: np.ndarray, mask: Optional[np.ndarray]=None) \
        -> Optional[Mapping[int, float]]:
    """
    :param y: a numpy array encoding sample classes; samples are encoded along
    the 0-axis
    :param mask: a boolean array of shape compatible with `y`, wherein True
    shows that the corresponding value(s) in `y` should be used to calculate
    weights; if `None` the function will consider all values in `y`
    :return: class weights
    """
    if not len(y):
        raise ValueError("`y` is empty")
    y_flat = (y.flatten() if mask is None else
              np.concatenate([sample[mask] for sample, mask in zip(y, mask)]))
    classes = np.unique(y_flat)
    weights = class_weight.compute_class_weight("balanced", classes, y_flat)
    weights_scaled = weights / weights.min()
    return {cls: weight for cls, weight in zip(classes, weights_scaled)}


def sample_weights(y: np.ndarray, class_weights: Mapping[int, float]) \
        -> np.ndarray:
    """
    :param y: a 2D array encoding sample classes; each sample is a row of
    integers representing class code
    :param class_weights: a class to weight mapping
    :return: a 2D array of the same shape as `y`, wherein each position stores
    a weight for the corresponding position in `y`
    """
    weights_mask = np.zeros(shape=y.shape, dtype=np.float32)
    for cls, weight in class_weights.items():
        weights_mask[y == cls] = weight
    return weights_mask


if __name__ == "__main__":
    raise RuntimeError
