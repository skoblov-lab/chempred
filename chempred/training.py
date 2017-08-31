import glob
import operator as op
import os
import re
import shutil
from contextlib import contextmanager
from functools import reduce
from itertools import chain, starmap
from typing import Tuple, List, Iterable, Callable

import numpy as np
from fn import F

from chempred import chemdner, util
from chempred.chemdner import Abstract, AbstractAnnotation
from chempred.intervals import Interval
from chempred.sampling import process_text


def process_data(pairs: Iterable[Tuple[Abstract, AbstractAnnotation]],
                 tokeniser: util.Tokeniser, width: int, minlen: int,
                 default: int=0) \
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
    ...     process_data(pairs, util.tokenise, 100, 50)
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

    def lst_merger(lists: Iterable[list]) -> list:
        return reduce(op.iadd, lists, [])

    def array_merger(arrays: Tuple[np.ndarray]) -> np.ndarray:
        return np.vstack(arrays) if arrays else np.array([], dtype=np.int32)

    # align pairs, flatten and sample windows
    flattened_pairs = list(map(chemdner.flatten_aligned_pair, pairs))
    titles = [title for title, _ in flattened_pairs]
    bodies = [body for _, body in flattened_pairs]
    # sanity  check
    assert [id_ for id_, *_ in titles] == [id_ for id_, *_ in bodies]
    proc_titles = [
        (id_, src, process_text(text, anno, tokeniser, width, minlen, default))
        for id_, src, text, anno in titles
    ]
    proc_bodies = [
        (id_, src, process_text(text, anno, tokeniser, width, minlen, default))
        for id_, src, text, anno in bodies
    ]
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
    ids = merge_results(0, lst_merger)([proc_titles_flat, proc_bodies_flat])
    src = merge_results(1, lst_merger)([proc_titles_flat, proc_bodies_flat])
    samples = merge_results(2, lst_merger)([proc_titles_flat, proc_bodies_flat])
    texts = merge_results(3, array_merger)([proc_titles_flat, proc_bodies_flat])
    cls = merge_results(4, array_merger)([proc_titles_flat, proc_bodies_flat])
    masks = merge_results(5, array_merger)([proc_titles_flat, proc_bodies_flat])
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
