"""

ChemPred prototype

"""

from typing import List, Tuple
import json

import click

from chempred import training

# configuration keys
MODELS = "models"

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


@click.group("chemdpred", help=__doc__)
@click.option("-m", "--models",
              help="Path to a directory with ChemPred models.")
@click.pass_context
def chempred(ctx, models):
    ctx[MODELS] = models


@chempred.command("train")
@click.option("-t", "--tagger",
              help="Configurations for the chemical entity tagging model; "
                   "the model will not be trained without them.")
@click.option("-d", "--detector",
              help="Configurations for the chemical entity detection model; "
                   "the model will not be trained without them.")
def train(ctx, tagger, detector):
    pass


@chempred.group("annotate")
def annotate(ctx):
    pass


if __name__ == "__main__":
    chempred()
