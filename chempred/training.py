import glob
from contextlib import contextmanager
from itertools import chain, starmap
from functools import reduce
from typing import Tuple, List, Mapping, Set, Union, Sequence, Text, Iterable, Callable

import numpy as np
import operator as op
import os
import re
import shutil
from fn import F

from chempred.intervals import Interval, Intervals
from chempred import chemdner
from chempred.chemdner import Abstract, AbstractAnnotation
from chempred import encoding
from chempred import sampling
from chempred import util
from chempred import chemdner

# Data = Union[Mapping[str, str], Sequence[str]]
# Sample = List[Interval]
# Failure = Tuple[int, ClassifiedRegion]


def process_text(text: Text,
                 annotation: Intervals[chemdner.ClassifiedInterval],
                 width: int, minlen: int, default: int=0) \
        -> Tuple[List[Intervals], np.ndarray, np.ndarray, np.ndarray]:
    """
    :param text:
    :param annotation:
    :param width: context window width (in charactes)
    :param minlen: minimum sample span
    :param default: default class encoding
    :return: samples, encoded text, encoded annotations, padding mask
    >>> import random
    >>> from chempred.intervals import Interval, Intervals
    >>> anno = Intervals([Interval(4, 10, 1), Interval(20, 25, 2)])
    >>> text = "".join(random.choice("abc ") for _ in range(len(anno.span)+9))
    >>> samples, text_e, cls_e, mask = process_text(text, anno, 10, 5)
    >>> text_e.shape == cls_e.shape == mask.shape
    True
    >>> len(samples) == len(text_e) == len(cls_e) == len(mask)
    True
    """
    # TODO return failures
    tokenised_text = util.tokenise(text)
    samples = [sample for sample in
               sampling.sample_windows(tokenised_text, width)]
    # remove samples with no annotated regions and insufficient length
    passing = [sample for sample in samples
               if len(sample.span) >= minlen and annotation.covers(sample.span)]

    if not passing:
        return [], np.array([]), np.array([]), np.array([])

    encoded_text = [
        encoding.encode_text(text, sample.span) for sample in passing
    ]
    encoded_classes = [
        encoding.encode_annotation(sample.span, annotation, default=default)
        for sample in passing
    ]

    joined_text, text_mask = util.join(encoded_text, width)
    joined_cls, cls_mask = util.join(encoded_classes, width)
    # sanity check
    assert (text_mask == cls_mask).all()
    return passing, joined_text, joined_cls, text_mask


def process_data(pairs: Iterable[Tuple[Abstract, AbstractAnnotation]],
                 width: int, minlen: int, default: int=0) \
        -> Tuple[List[int], List[str], List[Interval],
                 np.ndarray, np.ndarray, np.ndarray]:
    # TODO update docs
    # TODO tests
    """
    :param width: context window width (in charactes)
    :param minlen: minimum sample span
    :param default: default class encoding
    :return: text ids, samples, sample sources (title or body), encoded and
    padded text, encoded and padded classes, padding masks.
    >>> mapping = {"SYSTEMATIC": 1}
    >>> abstracts = chemdner.read_abstracts("testdata/abstracts.txt")
    >>> anno = chemdner.read_annotations("testdata/annotations.txt", mapping)
    >>> pairs = chemdner.align_abstracts_and_annotations(abstracts, anno)
    >>> ids, sources, spans, x, y, mask = (
    ...     process_data(pairs, 100, 50)
    ... )
    >>> len(ids) == len(sources) == x.shape[0] == y.shape[0] == mask.shape[0]
    True
    """
    def merge_results(field_idx: int, merger: Callable) -> Callable:
        return (F(map, F(map, op.itemgetter(field_idx)))
                >> chain.from_iterable
                >> (filter, lambda x: len(x) > 0)
                >> tuple
                >> merger)

    def list_merger(lists: Iterable[list]) -> list:
        return reduce(op.iadd, lists, [])

    # align pairs, flatten and sample windows
    flattened_pairs = list(map(chemdner.flatten_aligned_pair, pairs))
    titles = [title for title, _ in flattened_pairs]
    bodies = [body for _, body in flattened_pairs]
    # sanity  check
    assert [id_ for id_, *_ in titles] == [id_ for id_, *_ in bodies]
    proc_titles = [(id_, src, process_text(text, anno, width, minlen, default))
                    for id_, src, text, anno in titles]
    proc_bodies = [(id_, src, process_text(text, anno, width, minlen, default))
                    for id_, src, text, anno in bodies]
    # flatten processed data
    proc_titles_flat = [
        ([id_] * len(samples), [src] * len(samples), samples, txt, cls, mask)
        for id_, src, (samples, txt, cls, mask) in proc_titles
    ]
    proc_bodies_flat = [
        ([id_] * len(samples), [src] * len(samples), samples, txt, cls, mask)
        for id_, src, (samples, txt, cls, mask) in proc_bodies
    ]
    # merge data from titles and bodies
    ids = merge_results(0, list_merger)([proc_titles_flat, proc_bodies_flat])
    src = merge_results(1, list_merger)([proc_titles_flat, proc_bodies_flat])
    samples = merge_results(2, list_merger)([proc_titles_flat, proc_bodies_flat])
    texts = merge_results(3, np.vstack)([proc_titles_flat, proc_bodies_flat])
    cls = merge_results(4, np.vstack)([proc_titles_flat, proc_bodies_flat])
    masks = merge_results(5, np.vstack)([proc_titles_flat, proc_bodies_flat])
    return ids, src, samples, texts, cls, masks


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
