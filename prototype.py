"""

ChemPred prototype

"""

from typing import List, Tuple

import click



@click.group("chemdpred", help=__doc__)
@click.option("-m", "--models",
              help="Path to a directory with ChemPred models.")
# @click.option("-a", "--abstracts", default=None)
# @click.option("-t", "--annotations", default=None)
@click.pass_context
def chempred(ctx, models):
    pass


@chempred.command("train")
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
