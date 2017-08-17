"""

ChemPred prototype

"""

import click
import os

from keras import models
from keras import layers
from keras import callbacks

from chempred import preprocessing as pp
from chempred import training
from chempred import model
from chempred import chemdner


MODELS = "models"
ABSTRACTS = "abstracts"
ANNOTATIONS = "annotations"
DETECTOR = "detector"
TAGGER = "tagger"

NCHAR = pp.MAXCHAR + 1
EMBED = 50


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
    models_dir = ctx[MODELS]

    if detector:
        config = training.read_config(detector)
        if set(config.mapping.values()) != {0, 1}:
            raise ValueError("The detector's mapping must be binary")
        ncls = 2

        # read training data
        train_abstracts = chemdner.read_abstracts(config.train_data[ABSTRACTS])
        train_anno = chemdner.read_annotations(config.train_data[ANNOTATIONS])
        train_ids, train_samples, train_fail, train_x, train_y, train_mask = (
            training.process_data(train_abstracts, train_anno, config.window,
                                  config.maxlen, config.nonpositive,
                                  config.mapping, config.positive)
        )
        # read testing data
        test_abstracts = chemdner.read_abstracts(config.test_data[ABSTRACTS])
        test_anno = chemdner.read_annotations(config.test_data[ANNOTATIONS])
        test_ids, test_samples, test_fail, test_x, test_y, test_mask = (
            training.process_data(test_abstracts, test_anno, config.window,
                                  config.maxlen, config.nonpositive,
                                  config.mapping, config.positive)
        )
        # build the model
        l_in = layers.Input(shape=(config.maxlen,), name="l_in")
        l_emb = layers.Embedding(NCHAR, EMBED, mask_zero=True,
                                 input_length=config.maxlen)(l_in)
        l_rec = model.build_rec(config.nsteps, config.in_drop,
                                config.rec_drop)(l_emb)
        l_out = layers.TimeDistributed(
            layers.Dense(ncls, activation='softmax'), name="l_out")(l_rec)
        detector_model = models.Model(l_in, l_out)
        detector_model.compile(optimizer="Adam", loss="binary_crossentropy",
                               metrics=["accuracy"])

        # train the model
        with training.training(models_dir, DETECTOR) as (destination, weights):
            # save architecture
            detector_json = detector_model.to_json()
            with open(destination, "w") as json_file:
                json_file.write(detector_json)
            checkpoint = callbacks.ModelCheckpoint(weights, monitor="val_acc",
                                                   verbose=1, mode="max",
                                                   save_best_only=True)
            detector_model.fit(train_x, train_y, callbacks=[checkpoint],
                               validation_data=(test_x, test_y), verbose=1,
                               epochs=config.epochs, batch_size=config.batchsize)
    if tagger:
        raise NotImplemented


@chempred.group("annotate")
def annotate(ctx):
    raise NotImplemented


if __name__ == "__main__":
    chempred()
