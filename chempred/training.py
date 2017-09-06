import glob
import operator as op
import os
import re
import shutil
from contextlib import contextmanager
from itertools import chain, starmap
from typing import Tuple, List, Iterable, Callable, Text, Sequence

import numpy as np
from fn import F

from chempred import chemdner, util, encoding
from chempred.encoding import annotate_sample
from chempred.chemdner import Abstract, AbstractAnnotation
from chempred.util import Interval


def process_data(pairs: Iterable[Tuple[Abstract, AbstractAnnotation]],
                 parser: Callable[[Text], Sequence[Interval]], window: int,
                 validator: Callable) \
        -> Tuple[Tuple[int], Tuple[str], Tuple[Interval], Tuple[np.ndarray]]:
    # TODO update docs
    # TODO tests
    """
    :param window: context window width (in raw tokens)
    :return: text ids, sample sources (title or body), sampled intervals,
    sample annotations
    # >>> mapping = {"SYSTEMATIC": 1}
    # >>> abstracts = chemdner.read_abstracts("testdata/abstracts.txt")
    # >>> anno = chemdner.read_annotations("testdata/annotations.txt", mapping)
    # >>> pairs = chemdner.align_abstracts_and_annotations(abstracts, anno)
    # >>> ids, sources, spans, x, y, mask = (
    # ...     process_data(pairs, util.tokenise, 100, 50)
    # ... )
    # >>> len(ids) == len(sources) == x.shape[0] == y.shape[0] == mask.shape[0]
    # True
    """
    ids, srcs, texts, annotations = zip(
        *chain.from_iterable(map(chemdner.flatten_aligned_pair, pairs)))
    samples = (util.sample_windows(intervals, window)
               for intervals in [parser(text) for text in texts])
    annotations = [encoding.encode_annotation(anno, len(text))
                   for text, anno in zip(texts, annotations)]
    processed = chain.from_iterable(
        ((id_, src, s, annotate_sample(s, anno)) for s in samples if len(s))
        for id_, src, samples, anno in zip(ids, srcs, samples, annotations)
    )
    ids_, src_, samples_, sample_anno = zip(*filter(validator, processed))
    return ids_, src_, samples_, sample_anno


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
