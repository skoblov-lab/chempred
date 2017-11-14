"""

This is a simple example CLI for the ChemPred model submitted for publication in
the Journal of Cheminformatics. All the configurations are hard-coded for the
model we present in the publication, though it is not hard to customise the
CLI and the functions. If you ever need assistance, don't hesitate to contact
the developers (e.g. the corresponding author) via the issue tracker.

"""

import operator as op
import os
import re
import sys
from typing import Callable, List

import click
import joblib
import numpy as np
from fn import F

from sciner import util, intervals
from sciner.preprocessing import parsing

NSTEPS = 200  # max sentence length (in time-steps/tokens)
CHARLEN = 30  # max character length per token
TOKENISER = F(parsing.ptokenise, [re.compile("\w+|[^\s\w]")])
THRESHOLD = 0.5


def join_nested(arrays, nsteps, nfeatures, trim=True):
    joined_features = (
        F(map, F(util.join, length=nfeatures, trim=trim))
        >> (map, op.itemgetter(0)) >> list
    )(arrays)
    return util.join(joined_features, nsteps, trim=trim)


def readlines(path: str,
              tokeniser: Callable[[str], List[intervals.Interval[str]]],
              maxsteps: int) \
        -> List[List[intervals.Interval[str]]]:
    with open(path) as lines:
        cleanlines = (F(map, str.strip) >> (filter, bool))(lines)
        tokenised = map(tokeniser, cleanlines)
        return [tokens[:maxsteps] for tokens in tokenised]


@click.command("chempred-example", help=__doc__)
@click.option("-s", "--sentences", type=str, required=True,
              help="a plain-text file with one sentence per line")
@click.option("-b", "--beginnings", type=str, required=True,
              help="output file for detected entity-beginnings")
@click.option("-p", "--parts", type=str, required=True,
              help="output file for detected entitys parts")
@click.option("-m", "--model_weights", type=str, required=True,
              help="an hdf5 file with model weights")
@click.option("-w", "--wencoder", type=str, required=True,
              help="joblib-serialised word encoder path")
@click.option("-c", "--cencoder", type=str, required=True,
              help="joblib-serialised character encoder path")
@click.option("--batchsize", type=int, default=100,
              help="batch size; increse or decrease to suit your RAM/VRAM")
def main(sentences: str, beginnings: str, parts: str, model_weights: str,
         wencoder: str, cencoder: str, batchsize: str):

    # create word and character encoders
    word_encoder = joblib.load(os.path.abspath(wencoder))
    char_encoder = joblib.load(os.path.abspath(cencoder))
    encode_words = (F(map, F(word_encoder.encode, vectors=True)) >> list
                    >> F(util.join, length=NSTEPS, trim=True))
    encode_chars = (F(map, char_encoder.encode) >> list
                    >> F(join_nested, nsteps=NSTEPS, nfeatures=CHARLEN))
    # tokenise and encode sentences
    tokens = readlines(os.path.abspath(sentences), TOKENISER, NSTEPS)
    encoded_words, word_mask = encode_words(tokens)
    encoded_characters, char_mask = encode_chars(tokens)
    prob_masks = word_mask.astype(np.float32)

    # it's not quite right to import something anywhere else other than the
    # module's top-level, but we'll have to do it this way to avoid painfully
    # slow tensorflow loading in case a user only wants to get the help page
    from keras import layers, models, optimizers
    from sciner.models import build

    # the size of your GloVe word embeddings
    wordemb_dim = word_encoder.vectors.shape[-1]
    # the size of character embeddings your are going to use
    charemb_dim = 50
    # the number of recurrent units to use for character-level
    # embeddings (per direction)
    units = 30
    # the type of recurrent network to use
    layer = layers.GRU

    # zero-padding masks
    masks = layers.Input((NSTEPS, 1), name="masks", dtype="float32")

    # word-level computational graph
    wordemb = layers.Input((NSTEPS, wordemb_dim), name="wordemb")
    wordcnn = build.cnn([200, 250], 2, [0.3, None], name_template="wordcnn{}")(
        wordemb)
    wordcnn = layers.multiply([wordcnn, masks])
    wordcnn = layers.Masking(0.0)(wordcnn)

    # character-level computational graph
    characters = layers.Input((NSTEPS, CHARLEN), dtype="int32",
                              name="characters")
    charemb = build.char_embeddings(len(char_encoder), NSTEPS, charemb_dim,
                                    units, 0.3, 0.3, mask=True, layer=layer)(
        characters)
    charcnn = build.cnn([200, 250], 2, [0.3, None], name_template="charcnn{}")(
        charemb)
    charcnn = layers.multiply([charcnn, masks])
    charcnn = layers.Masking(0.0)(charcnn)

    # merge word-level and character-level features and run them
    # through a 2-layer bidirectional RNN
    merged = layers.concatenate([wordcnn, charcnn], axis=-1)
    rnn = build.rnn([150, 150], 0.1, 0.1, bidirectional="concat", layer=layer)(
        merged)

    # branch out a computational graph for entity-part detection
    rnn_runs = build.rnn([150], 0.1, 0.1, bidirectional="concat", layer=layer)(
        rnn)
    output_runs = layers.Dense(1, activation="sigmoid")(rnn_runs)

    # create an attention loop for entity beginning detection
    rnn_borders = layers.multiply([output_runs, rnn])
    rnn_borders = build.rnn([150], 0.1, 0.1, bidirectional="concat",
                            layer=layer)(rnn_borders)
    output_borders = layers.Dense(1, activation="sigmoid")(rnn_borders)

    # compile the model
    model = models.Model([wordemb, characters, masks],
                         [output_runs, output_borders])
    model.compile(optimizer=optimizers.Adam(clipvalue=1.0),
                  loss="binary_crossentropy",
                  sample_weight_mode="temporal")
    # load model weights
    model.load_weights(os.path.abspath(model_weights))
    # run predictions
    inputs = [encoded_words, encoded_characters, prob_masks[:, :, None]]
    print("Running predictions! This might take a while", file=sys.stderr)
    pred_parts, pred_beginnings = model.predict(inputs, batch_size=batchsize)
    # write outputs
    print("Writing outputs.", file=sys.stderr)
    with open(os.path.abspath(parts), "w") as parts_out:
        for sentence_tks, decisions in zip(tokens, pred_parts >= THRESHOLD):
            selected = [part for part, take in zip(sentence_tks, decisions) if take]
            formatted = ["{}:{}:{}".format(iv.start, iv.stop, iv.data) for iv in selected]
            print("\t".format(formatted), file=parts_out)
    with open(os.path.abspath(beginnings), "w") as beg_out:
        for sentence_tks, decisions in zip(tokens, pred_beginnings >= THRESHOLD):
            selected = [part for part, take in zip(sentence_tks, decisions) if take]
            formatted = ["{}:{}:{}".format(iv.start, iv.stop, iv.data) for iv in selected]
            print("\t".format(formatted), file=beg_out)
    print("Done!")


if __name__ == "__main__":
    main()
