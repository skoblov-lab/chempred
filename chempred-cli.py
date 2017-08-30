"""

ChemPred prototype

"""

import click
import os
from keras import callbacks
from keras import layers
from keras import models

import chempred.util
from chempred import chemdner
from chempred import encoding
from chempred import model
from chempred import training
from chempred import util


NCHAR = encoding.MAXCHAR + 1
MODEL = "chempred"
DIRECTORY = "directory"

# CONFIGURATION KEYS

EMBEDDING = "embed"  # the size of character embeddings
# convolutional block
FILTER_WIDTH = "filter_width"  # 1D convolutional filter width
NFILTERS = "the number of filters to use at each layer"
# recurrent block
NSTEPS = "nsteps"  # the number of steps to use at each rnn layer
INPUT_DROPOUT = "input_dropout"  # rnn layer input dropout
REC_DROPOUT = "recurrent_dropout"  # rnn layer recurrent dropout
BIDIRECTIONAL = "bidirectional"  # bidirectional rnn flags
STATEFUL = "stateful"  # use stateful rnn layers
# sampling block
WINDOW = "window"  # sampling window width
MINLEN = "maxlen"  # sample size limit
MAPPING = "class_mapping"  # string to integer class mapping

# TODO extensive doc update
# TODO return conv-layers
# TODO check that data contains > 1 cls
# Note! Any class not in mapping is mapped into 0

@click.group("chemdpred", chain=True, help=__doc__)
@click.option("-d", "--directory",
              help="Path to a directory with ChemPred models.")
@click.option("-c", "--config",
              help="Configurations.")
@click.pass_context
def chempred(ctx, directory, config):
    ctx.obj[DIRECTORY] = os.path.abspath(directory)
    ctx.obj.update(model.Config(open(os.path.abspath(config))))


@chempred.command("detect")
@click.option("-i", "--input", htype=str, help="Plain text input file")
@click.pass_context
def detect(ctx, input):
    pass


@chempred.command("train")
@click.option("--train_abstracts", type=str, required=True)
@click.option("--train_annotations", type=str, required=True)
@click.option("--test_abstracts", type=str, required=True)
@click.option("--test_annotations", type=str, required=True)
@click.option("--epochs", type=int, default=30)
@click.option("--batchsize", type=int, default=400)
@click.pass_context
def train(ctx, train_abstracts, train_annotations, test_abstracts,
          test_annotations, epochs, batchsize):
    config = ctx.obj
    directory = config[DIRECTORY]
    model_destination = os.path.join(directory, "{}.json".format(MODEL))

    # extract training_configs
    mapping = chempred.util.parse_mapping(config[MAPPING])
    maxlen = config[MINLEN]
    window = config[WINDOW]

    embedding = config[EMBEDDING]
    lstm_layers = config[NSTEPS]
    bidirectional = config[BIDIRECTIONAL]
    input_dropout = config[INPUT_DROPOUT]
    rec_dropout = config[REC_DROPOUT]

    # TODO report failures
    # read data




if __name__ == "__main__":
    chempred(obj=model.Config({}))
