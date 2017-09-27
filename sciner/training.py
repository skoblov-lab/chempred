import glob
import operator as op
import os
import re
import shutil
from contextlib import contextmanager
from itertools import starmap
from typing import Tuple, List, Iterable, Text, Sequence, \
    Optional, Mapping, cast

import numpy as np
from fn import F
from sklearn.utils import class_weight

from sciner.intervals import Interval, Intervals, extract
from sciner.sampling import Annotator, Sampler, Sample

ProcessedSample = Tuple[int, Text, Sequence[Interval], Sequence[Text],
                        Optional[np.ndarray]]


def process_record(id_: int, src: Text, text: Text, parsed_text: Intervals,
                   sampler: Sampler, annotator: Optional[Annotator]=None) \
        -> Sequence[ProcessedSample]:
    # TODO update docs
    # TODO tests
    """
    :param pair: abstract paired with its annotation
    :param window: context window width (in raw tokens)
    :return: Iterator[(text ids, sample sources (title or body),
    sampled intervals, sample tokens, sample annotations)]
    """
    def wrap_sample(sample: Sample) \
            -> Tuple[int, Text, Sample, Sequence[Text], Optional[np.ndarray]]:
        sample_text = cast(Sequence[Text], extract(text, sample))
        sample_anno = None if annotator is None else annotator(sample)
        return id_, src, sample, sample_text, sample_anno

    samples = sampler(parsed_text)
    return [wrap_sample(sample) for sample in samples if len(sample)]


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
    # TODO update docs
    # TODO tests
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
    if y.ndim == 2:
        y_flat = (y.flatten() if mask is None else
                  np.concatenate([sample[mask] for sample, mask in zip(y, mask)]))
    elif y.ndim == 3:
        y_flat = (y.nonzero()[-1] if mask is None else
                  y[mask].nonzero()[-1])
    else:
        raise ValueError("`y` should be either a 2D or a 3D array")
    classes = np.unique(y_flat)
    weights = class_weight.compute_class_weight("balanced", classes, y_flat)
    weights_scaled = weights / weights.min()
    return {cls: weight for cls, weight in zip(classes, weights_scaled)}


def sample_weights(y: np.ndarray, class_weights: Mapping[int, float]) \
        -> np.ndarray:
    # TODO update docs
    # TODO tests
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
