"""

ChemPred prototype

"""
import os
import random

import click
from keras import callbacks

from chempred import chemdner
from chempred import encoding
from chempred import model
from chempred import training
from chempred.training import process_data
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
# training block
WEIGHTED = "weighted"

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
@click.option("-a", "--abstracts", type=str, required=True)
@click.option("-n", "--annotations", type=str, required=True)
@click.option("-e", "--epochs", type=int, default=30)
@click.option("-b", "--batchsize", type=int, default=400)
@click.option("-v", "--val_split", type=float, default=0.1)
@click.pass_context
def train(ctx, abstracts, annotations, epochs, batchsize, val_split):
    config = ctx.obj
    directory = config[DIRECTORY]
    model_destination = os.path.join(directory, "{}.json".format(MODEL))

    # extract configurations
    mapping = util.parse_mapping(config[MAPPING])
    minlen = config[MINLEN]
    window_width = config[WINDOW]

    embedding = config[EMBEDDING]
    nfilters = config[NFILTERS]
    filter_width = config[FILTER_WIDTH]

    nsteps = config[NSTEPS]
    bidirectional = config[BIDIRECTIONAL]
    input_dropout = config[INPUT_DROPOUT]
    rec_dropout = config[REC_DROPOUT]
    stateful = config[STATEFUL]

    weighted = config[WEIGHTED]

    if set(mapping.values()) | {0} != {0, 1}:
        raise ValueError("The mapping must be binary")
    ncls = len(set(mapping.values()) | {0})

    # read data
    parsed_abstracts = chemdner.read_abstracts(
        os.path.abspath(abstracts))
    parsed_anno = chemdner.read_annotations(
        os.path.abspath(annotations), mapping)
    pairs = chemdner.align_abstracts_and_annotations(parsed_abstracts,
                                                     parsed_anno)
    nonempty_pairs = [(abstact, abstract_annotations) for
                      abstact, abstract_annotations in pairs
                      if any(abstract_annotations[1:])]
    random.shuffle(nonempty_pairs)
    # separate training and testing datasets
    n_val_pairs = int(len(nonempty_pairs) * val_split)
    pairs_train = nonempty_pairs[:-n_val_pairs]
    pairs_test = nonempty_pairs[-n_val_pairs:]

    # generate training samples
    tokeniser = util.tokenise
    ids, src, samples, x, y, mask = process_data(
        pairs_train, tokeniser, window_width, minlen)
    ids_test, src_test, samples_test, x_test, y_test, mask_test = process_data(
        pairs_test, tokeniser,  window_width, minlen)
    y_onehot = (util.maskfalse(util.one_hot(y)) if not nfilters else
                util.one_hot(y))
    y_test_onehot = (util.maskfalse(util.one_hot(y_test)) if not nfilters else
                     util.one_hot(y_test))

    # calculate sample weights
    class_weights = util.balance_class_weights(y, mask) if weighted else None
    sample_weights = util.sample_weights(y, class_weights) if weighted else None

    # build, serialise and train the model
    detector_model = model.build_nn(
        window_width, embedding, ncls, nfilters, filter_width, nsteps,
        input_dropout, rec_dropout, bidirectional, stateful
    )
    detector_model.compile(optimizer="Adam", loss="binary_crossentropy",
                           sample_weight_mode="temporal", metrics=["accuracy"])
    with open(model_destination, "w") as model_out:
        print(detector_model.to_json(), file=model_out)
    with training.training(directory, MODEL) as model_checkpoint_path:
        checkpoint = callbacks.ModelCheckpoint(model_checkpoint_path,
                                               monitor="val_acc",
                                               verbose=1, mode="max",
                                               save_best_only=True)
        detector_model.fit(x, y_onehot, validation_data=(x_test, y_test_onehot),
                           verbose=1, epochs=epochs, batch_size=batchsize,
                           sample_weight=sample_weights,  callbacks=[checkpoint])


if __name__ == "__main__":
    chempred(obj=model.Config({}))
