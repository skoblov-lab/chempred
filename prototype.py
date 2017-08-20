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
from chempred.util import parse_mapping

MODELS = "models"
TRAIN = "Train"
TEST = "Test"
ABSTRACTS = "Abstracts"
ANNO = "Annotations"
EPOCHS = "epochs"
BATCHSIZE = "batchsize"

DETECTOR = "detector"
TAGGER = "tagger"

NCHAR = encoding.MAXCHAR + 1
EMBED = 50
DEF_POSITIVE_CLS = ("ABBREVIATION", "FAMILY", "FORMULA", "IDENTIFIER",
                    "MULTIPLE", "NO CLASS", "SYSTEMATIC", "TRIVIAL")
# TODO extensive doc update
# TODO check that data contains > 1 cls
# Any class not in mapping is mapped into 0
DEF_MAPPING_DETECTOR = ("ABBREVIATION:1", "FAMILY:1", "FORMULA:1",
                        "IDENTIFIER:1", "MULTIPLE:1", "NO CLASS:1",
                        "SYSTEMATIC:1", "TRIVIAL:1")


@click.group("chemdpred", help=__doc__)
@click.option("-d", "--directory",
              help="Path to a directory with ChemPred models.")
@click.pass_context
def chempred(ctx, directory):
    ctx[MODELS] = os.path.abspath(directory)


@chempred.group("train")
@click.option("--train_abstracts", type=str)
@click.option("--train_annotations", type=str)
@click.option("--test_abstracts", type=str)
@click.option("--test_annotations", type=str)
@click.option("--epochs", type=int, default=30)
@click.option("--batchsize", type=int, default=400)
@click.pass_context
def train(ctx, train_abstracts, train_annotations, test_abstracts,
          test_annotations, epochs, batchsize):
    # read data
    train_abstracts = (None if train_abstracts is None else
                       chemdner.read_abstracts(os.path.abspath(train_abstracts)))
    train_anno = chemdner.read_annotations(os.path.abspath(train_annotations))

    test_abstracts = (None if test_abstracts is None else
                      chemdner.read_abstracts(os.path.abspath(test_abstracts)))
    test_anno = chemdner.read_annotations(os.path.abspath(test_annotations))

    ctx[TRAIN + ABSTRACTS] = train_abstracts
    ctx[TRAIN + ANNO] = train_anno
    ctx[TEST + ABSTRACTS] = test_abstracts
    ctx[TEST + ANNO] = test_anno
    ctx[EPOCHS] = epochs
    ctx[BATCHSIZE] = batchsize


@train.command("detector")
@click.option("-l", "--lstm_steps", type=int, multiple=True,
              default=(200, 200))
@click.option("-b", "--bidirectional", type=bool,
              default=(True,))
@click.option("-idrop", "--input_dropout", type=float, multiple=True,
              default=(0.1,))
@click.option("-rdrop", "--rec_dropout", type=float, multiple=True,
              default=(0.1,))
@click.option("-m", "--maxlen", type=int,
              default=400)
@click.option("-w", "--window", type=int,
              default=5)
@click.option("-n", "--n_nonpositive", type=int,
              default=3)
@click.option("-p", "--positive", type=str, multiple=True,
              default=DEF_POSITIVE_CLS)
@click.option("-c", "--mapping", type=str, multiple=True)
@click.pass_context
def detector(ctx, lstm_steps, bidirectional, input_dropout, rec_dropout, maxlen,
             window, n_nonpositive, positive, mapping):
    train_abstracts = ctx.obj[TRAIN + ABSTRACTS]
    train_anno = ctx.obj[TRAIN + ANNO]
    test_abstracts = ctx.obj[TEST + ABSTRACTS]
    test_anno = ctx.obj[TEST + ANNO]

    if not train_abstracts or not test_abstracts:
        raise ValueError("You must pass abstracts to train a detector")

    # process arguments
    bidirectional = (bidirectional[0] if len(bidirectional) == 1 else
                     bidirectional)
    input_dropout = (input_dropout[0] if len(input_dropout) == 1 else
                     input_dropout)
    rec_dropout = (rec_dropout[0] if len(rec_dropout) == 1 else
                   rec_dropout)
    positive = set(positive)
    mapping = parse_mapping(mapping)
    ncls = len(set(mapping.values()) | {0})

    if set(mapping.values()) | {0} != {0, 1}:
        raise ValueError("The detector's mapping must be binary")

    # encode data
    train_ids, train_samples, train_fail, train_x, train_y, train_mask = (
        training.process_data_detector(train_abstracts, train_anno, window,
                                       maxlen, n_nonpositive, mapping, positive)
    )
    test_ids, test_samples, test_fail, test_x, test_y, test_mask = (
        training.process_data_detector(test_abstracts, test_anno, window,
                                       maxlen, n_nonpositive, mapping, positive)
    )

    train_y_onehot = chempred.encoding.maskfalse(
        chempred.encoding.one_hot(train_y), train_mask)
    test_y_onehot = chempred.encoding.maskfalse(
        chempred.encoding.one_hot(test_y), test_mask)

    # build the model
    l_in = layers.Input(shape=(maxlen,), name="l_in")
    l_emb = layers.Embedding(NCHAR, EMBED, mask_zero=True,
                             input_length=maxlen)(l_in)
    l_rec = model.build_rec(lstm_steps, input_dropout, rec_dropout,
                            bidirectional)(l_emb)
    l_out = layers.TimeDistributed(
        layers.Dense(ncls, activation='softmax'), name="l_out")(l_rec)
    detector_model = models.Model(l_in, l_out)
    detector_model.compile(optimizer="Adam", loss="binary_crossentropy",
                           metrics=["accuracy"])

    # train the model
    with training.training(ctx.obj[MODELS], DETECTOR) as (destination, weights):
        # save architecture
        detector_json = detector_model.to_json()
        with open(destination, "w") as json_file:
            json_file.write(detector_json)
        checkpoint = callbacks.ModelCheckpoint(weights, monitor="val_acc",
                                               verbose=1, mode="max",
                                               save_best_only=True)
        detector_model.fit(train_x, train_y_onehot, callbacks=[checkpoint],
                           validation_data=(test_x, test_y_onehot),
                           verbose=1, epochs=ctx.obj[EPOCHS],
                           batch_size=ctx.obj[BATCHSIZE])


if __name__ == "__main__":
    chempred(obj={})
