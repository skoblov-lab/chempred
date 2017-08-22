"""

ChemPred prototype

"""

import click
import os
from keras import callbacks
from keras import layers
from keras import models

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
MAXLEN = "maxlen"  # sample size limit
N_NONPOS = "n_nonpositive"  # the number of non-pos targets to sample per text
MAPPING = "class_mapping"  # string to integer class mapping
POSITIVE = "positive_cls"  # a sequence of positive classes

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
    mapping = model.parse_mapping(config[MAPPING])
    positive = set(config[POSITIVE])
    maxlen = config[MAXLEN]
    window = config[WINDOW]
    n_nonpos = config[N_NONPOS]

    embedding = config[EMBEDDING]
    lstm_layers = config[NSTEPS]
    bidirectional = config[BIDIRECTIONAL]
    input_dropout = config[INPUT_DROPOUT]
    rec_dropout = config[REC_DROPOUT]

    # TODO report failures
    # read data
    train_abstracts = chemdner.read_abstracts(os.path.abspath(train_abstracts))
    train_anno = chemdner.read_annotations(os.path.abspath(train_annotations))
    test_abstracts = chemdner.read_abstracts(os.path.abspath(test_abstracts))
    test_anno = chemdner.read_annotations(os.path.abspath(test_annotations))

    # encode data
    train_ids, train_samples, train_fail, train_x, train_y, train_mask = (
        training.process_data(train_abstracts, train_anno, window,
                              maxlen, n_nonpos, mapping, positive)
    )
    train_y_onehot = util.one_hot(train_y)
    train_y_masked = util.maskfalse(train_y_onehot, train_mask)

    test_ids, test_samples, test_fail, test_x, test_y, test_mask = (
        training.process_data(test_abstracts, test_anno, window,
                              maxlen, n_nonpos, mapping, positive)
    )
    test_y_onehot = util.one_hot(test_y)
    test_y_masked = util.maskfalse(test_y_onehot, test_mask)



if __name__ == "__main__":
    chempred(obj=model.Config({}))
