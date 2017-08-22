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

# config keys
MAPPING = "string_to_int_class_mapping"
POSITIVE_CLASSES = "positive_classes"
WINDOW = "sample_window_length_in_tokens"
MAXLEN = "maximum_sample_length_in_characters"
NONPOS_TARGETS = "number_of_non-positive_targets_to_sample_per_text"

MODEL_DIR = "model_directory"

EMBEDDING = "character_embedding_size"
CONV_WIDTH = "1d_conv_width"
CONV_NFILT = "number_of_filters"
LSTM_STEPS = "number_of_lstm_steps_per_layer"
BIDIRECTIONAL = "use_bidirectional_lstm_layers"
INPUT_DROP = "input_dropout_in_lstm_layers"
REC_DROP = "recurrent_dropout_in_lstm_layers"

EPOCHS = "epochs"
BATCHSIZE = "batchsize"

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
    ctx.obj[MODEL_DIR] = os.path.abspath(directory)

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
    # extract training_configs
    mapping = ctx[MAPPING]
    positive = ctx[POSITIVE_CLASSES]
    maxlen = ctx[MAXLEN]
    window = ctx[WINDOW]
    n_nonpos = ctx[NONPOS_TARGETS]

    embedding = ctx[EMBEDDING]
    lstm_layers = ctx[LSTM_STEPS]
    bidirectional = ctx[BIDIRECTIONAL]
    input_dropout = ctx[INPUT_DROP]
    rec_dropout = ctx[REC_DROP]

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
    chempred(obj={})
