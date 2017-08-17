"""

ChemPred prototype

"""

from typing import List, Tuple
from itertools import chain, starmap
import operator as op

import numpy as np
import click
from fn import F

from chempred import chemdner
from chempred import preprocessing as pp
from chempred import model


Sample = List[chemdner.Annotation]
Failure = Tuple[int, chemdner.Annotation]
CLASS_MAPPING = {
    "OTHER": 0,
    "ABBREVIATION": 1,
    "FAMILY": 2,
    "FORMULA": 3,
    "IDENTIFIER": 4,
    "MULTIPLE": 5,
    "NO CLASS": 6,
    "SYSTEMATIC": 7,
    "TRIVIAL": 8
}
BINARY_MAPPING = {cls: 0 if cls == "OTHER" else 1 for cls in CLASS_MAPPING}
POSITIVE_CLASSES = {cls for cls in CLASS_MAPPING if cls != "OTHER"}


def prepare_training_data(abstracts: str, annotations: str, window: int,
                          maxlen: int, nonpositive: int, binary: bool=True) \
        -> Tuple[List[int], List[Sample],  List[Failure],
                 np.ndarray, np.ndarray, np.ndarray]:
    """
    :param abstracts: a path to chemdner-formatted abstracts
    :param annotations: a path to chemdner-formatted annotations
    :param window: context window width (in tokens)
    :param maxlen: maximum sample length (in characters)
    :param nonpositive: the number of non-positive target words per abstract
    :param binary: use binary token classification
    :return: abstract ids (one per sample), samples, failed targets (abstract
    ids and token annotations), encoded and padded text, encoded and padded
    classes, padding masks.
    """
    mapping = BINARY_MAPPING if binary else CLASS_MAPPING
    abstracts = chemdner.read_abstracts(abstracts)
    abstract_annotations = chemdner.read_annotations(annotations)

    # align pairs, flatten and remove texts with no annotations
    aligned = list(chemdner.align_abstracts_and_annotations(abstracts,
                                                            abstract_annotations))
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

    targets = [pp.sample_targets(POSITIVE_CLASSES, annotations, nonpositive)
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


@click.group("chemdpred", help=__doc__)
@click.option("-m", "--models",
              help="Path to a directory with ChemPred models.")
# @click.option("-a", "--abstracts", default=None)
# @click.option("-t", "--annotations", default=None)
@click.pass_context
def chempred(ctx, models):
    pass


@chempred.group("train")
@click.option("-t", "--tagger",
              help="Configurations for the chemical entity tagging model; "
                   "the model will not be trained without them.")
@click.option("-d", "--detector",
              help="Configurations for the chemical entity detection model; "
                   "the model will not be trained without them.")
def train(ctx, segmentation, classifier):
    pass


@chempred.group("annotate")
def annotate(ctx):
    pass


if __name__ == "__main__":
    chempred()
