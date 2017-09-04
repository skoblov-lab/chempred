import glob
import operator as op
import os
import re
import shutil
from contextlib import contextmanager
from functools import reduce
from itertools import chain, starmap
from typing import Tuple, List, Iterable, Callable, cast

import numpy as np
from fn import F

from chempred import chemdner, util, encoding
from chempred.chemdner import Abstract, AbstractAnnotation
from chempred.util import WS_PATT, Interval, Vocabulary, extract_intervals, parse


def process_data(pairs: Iterable[Tuple[Abstract, AbstractAnnotation]],
                 parser, vocab: Vocabulary, window: int) \
        -> Tuple[List[int], List[str], List[Interval],
                 np.ndarray, np.ndarray, np.ndarray]:
    # TODO update docs
    # TODO tests
    """
    :param window: context window width (in raw tokens)
    :return: text ids, samples, sample sources (title or body), encoded and
    padded text, encoded and padded classes, padding masks.
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
    raw_samples = (util.sample_windows(intervals, window)
                   for intervals in [parse(text, WS_PATT) for text in texts])

    refine = F(extract_intervals) >> " ".join >> parser
    refined_samples = [[refine(txt, s) for s in samples]
                       for txt, samples in zip(texts, raw_samples)]
    encoded_anno = [encoding.encode_annotation(anno, len(text))
                    for text, anno in zip(texts, annotations)]
    encoded_samples = [
        [encoding.encode_sample(s, txt, anno, vocab) for s in samples if len(s)]
        for txt, anno, samples in zip(texts, encoded_anno, refined_samples)
    ]
    ids, srcs, enc_tk, enc_tk_anno, enc_char, enc_char_anno = zip(
        *chain.from_iterable(
            ((id_, src, *s) for s in samples) for id_, src, samples in
            zip(ids, srcs, encoded_samples)
        )
    )
    return ids, srcs, enc_tk, enc_tk_anno, enc_char, enc_char_anno


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
