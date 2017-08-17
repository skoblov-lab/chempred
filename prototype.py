"""

ChemPred prototype

"""

import click
import os

from chempred import training, model


MODELS = "models"
ABSTRACTS = "abstracts"
ANNOTATIONS = "annotations"


@click.group("chemdpred", help=__doc__)
@click.option("-m", "--models",
              help="Path to a directory with ChemPred models.")
@click.pass_context
def chempred(ctx, models):
    ctx[MODELS] = os.path.abspath(models)


@chempred.command("train")
@click.option("-t", "--tagger", type=str, default=None,
              help="Configurations for the chemical entity tagging model; "
                   "the model will not be trained without them.")
@click.option("-d", "--detector", type=str, default=None,
              help="Configurations for the chemical entity detection model; "
                   "the model will not be trained without them.")
def train(ctx, tagger, detector):
    pass


@chempred.group("annotate")
def annotate(ctx):
    pass


if __name__ == "__main__":
    chempred()
