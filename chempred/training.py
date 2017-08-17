import glob
import json
from contextlib import contextmanager
from functools import reduce
from itertools import chain, starmap
from typing import Tuple, List, Mapping, Set, NamedTuple, Union, Sequence

import numpy as np
import operator as op
import os
import re
import shutil
from fn import F

from chempred import preprocessing as pp
from chempred import chemdner
from chempred.chemdner import Annotation, Abstract, AbstractAnnotation


# configuration fields
TRAINING_DATA = "training_data"
TESTING_DATA = "testing_data"

ARCHITECTURE = "architecture"
NSTEPS = "nsteps"
INPUT_DROP = "input_dropout"
REC_DROP = "recurrent_dropout"
BIDIRECTIONAL = "bidirectional"
STATEFUL = "stateful"
ARCHITECTURE_FIELDS = (NSTEPS, INPUT_DROP, REC_DROP, BIDIRECTIONAL, STATEFUL)

SAMPLING = "sampling"
WINDOW = "window"
MAXLEN = "maxlen"
NONPOS = "nonpositive"
SAMPLING_FIELDS = (WINDOW, MAXLEN, NONPOS)

TRAINING = "training"
EPOCHS = "nepochs"
BATCHSIZE = "batchsize"
MAPPING = "class_mapping"
POSITIVE = "positive"
TRAINING_FIELDS = (EPOCHS, BATCHSIZE)


Data = Union[Mapping[str, str], Sequence[str]]
Configurations = NamedTuple("Configurations",
                            [("train_data", Data),
                             ("test_data", Data),
                             ("nsteps", int),
                             ("in_drop", float),
                             ("rec_drop", float),
                             ("bidir", bool),
                             ("stateful", bool),
                             ("window", int),
                             ("maxlen", int),
                             ("nonpositive", int),
                             ("epochs", int),
                             ("batchsize", int),
                             ("mapping", Mapping[str, int]),
                             ("positive", Set[str])])
Sample = List[Annotation]
Failure = Tuple[int, Annotation]


def read_config(path: str) -> Configurations:
    # TODO docs
    # TODO tests
    """

    :param path:
    :return:
    """
    with open(path) as buffer:
        config = json.load(buffer)
    try:
        train_data = config[TRAINING_DATA]
        test_data = config[TESTING_DATA]
        nsteps, in_drop, rec_drop, bidir, stateful = (
            reduce(op.getitem, [ARCHITECTURE, field], config)
            for field in ARCHITECTURE_FIELDS
        )
        window, maxlen, nonpos = (
            reduce(op.getitem, [SAMPLING, field], config)
            for field in SAMPLING_FIELDS
        )

        epochs, batchsize = (
            reduce(op.getitem, [TRAINING, field], config)
            for field in TRAINING_FIELDS
        )

        mapping = config[MAPPING]
        positive = set(config[POSITIVE])
        return Configurations(train_data, test_data, nsteps, in_drop, rec_drop,
                              bidir, stateful, window, maxlen, nonpos, epochs,
                              batchsize, mapping, positive)
    except KeyError as err:
        raise ValueError("Not all fields are present: {}".format(err))


def process_data(abstracts: List[Abstract],
                 annotations: List[AbstractAnnotation],
                 window: int, maxlen: int, nonpositive: int,
                 mapping: Mapping[str, int], positive: Set[str]) \
        -> Tuple[List[int], List[Sample],  List[Failure],
                 np.ndarray, np.ndarray, np.ndarray]:
    # TODO update docs
    # TODO tests
    """
    :param abstracts:
    :param annotations:
    :param window: context window width (in tokens)
    :param maxlen: maximum sample length (in characters)
    :param nonpositive: the number of non-positive target words per abstract
    :param mapping: сlass mapping
    :param positive: a set of positive classes
    :return: abstract ids (one per sample), samples, failed targets (abstract
    ids and token annotations), encoded and padded text, encoded and padded
    classes, padding masks.
    """
    # align pairs, flatten and remove texts with no annotations
    aligned = list(chemdner.align_abstracts_and_annotations(abstracts,
                                                            annotations))
    data = (F(map, chemdner.flatten_aligned_pair)
            >> chain.from_iterable
            >> list)(aligned)
    nonempty = [(id_, src, text, annotations)
                for id_, src, text, annotations in data if annotations]
    nonempty_ids = [id_ for id_, *_ in nonempty]
    texts = [text for *_, text, _ in nonempty]

    # annotate texts and sample windows
    text_annotations = [chemdner.annotate_text(text, annotations, src, True)
                        for _, src, text, annotations in nonempty]

    targets = [pp.sample_targets(positive, annotations, nonpositive)
               for annotations in text_annotations]
    sampler = pp.make_sampler(maxlen=maxlen, width=window, flanking=False)
    samples_and_failures = (F(zip)
                            >> (starmap, F(pp.sample_windows, sampler=sampler))
                            >> list)(targets, text_annotations)
    samples = list(map(op.itemgetter(0), samples_and_failures))
    failures = list(map(op.itemgetter(1), samples_and_failures))
    failures_with_ids = [((id_, fail) for fail in failures_)
                         for id_, failures_ in zip(nonempty_ids, failures)
                         if failures_]
    flattened_failures = list(chain.from_iterable(failures_with_ids))

    # extract each sample window's text and encode it as char-codes;
    # join encoded text (using zero-padding to match lengths)
    encoded_texts = [[pp.encode_text(text, sample) for sample in samples_]
                     for text, samples_ in zip(texts, samples)]
    ids = [[id_] * len(samples_)
           for id_, samples_ in zip(encoded_texts, nonempty_ids)]
    encoded_classes = [
        [pp.encode_classes(mapping, sample) for sample in samples_]
        for text, samples_ in zip(texts, samples)]

    joined_texts, masks_text = pp.join(
        list(chain.from_iterable(encoded_texts)))
    joined_classes, masks_classes = pp.join(
        list(chain.from_iterable(encoded_classes)))
    flattened_ids = list(chain.from_iterable(ids))

    # sanity checks
    assert (masks_text == masks_classes).all()
    assert len(flattened_ids) == len(joined_texts)

    return (flattened_ids, samples, flattened_failures, joined_texts,
            joined_classes, masks_text)


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
    w_destination = os.path.join(training_dir, "{}-weights.hdf5".format(name))
    model_destination = os.path.join(training_dir, "{}.json".format(name))
    try:
        yield model_destination, weights_template
    finally:
        all_weights = glob.glob(os.path.join(training_dir, "*.hdf5"))
        if all_weights:
            best_weights, stats = pick_best(all_weights)
            shutil.move(best_weights, w_destination)
        shutil.rmtree(training_dir)


if __name__ == "__main__":
    raise RuntimeError
