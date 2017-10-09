"""

Utility functions for creating ChemPred deep learning models and working with
their predictions

"""
from functools import reduce
from typing import Sequence, Tuple, Optional, Union, Callable

from keras import layers, backend as K
import numpy as np


def build_cnn(nfilters: Sequence[int],
              filter_width: Union[int, Sequence[int]],
              dropout: Union[Optional[float], Sequence[Optional[float]]]=None,
              padding: Union[str, Sequence[str]]="same",
              name_template: str="conv{}") \
        -> Callable:
    # TODO extend documentation
    # TODO more tests
    """

    :param nfilters:
    :param filter_width:
    :return:
    """
    def stack_conv(prev, param: Tuple[str, int, int, float, str]):
        name, nfilt, kern_size, drop_p, pad = param
        l = layers.Convolution1D(
            nfilt, kern_size, activation="relu", name=name, padding=pad
        )(prev)
        return layers.Dropout(drop_p)(l) if drop_p else l

    filter_width = (filter_width if isinstance(filter_width, Sequence) else
                    [filter_width] * len(nfilters))
    dropout = (dropout if isinstance(dropout, Sequence) else
               [dropout] * len(nfilters))
    padding = (padding if isinstance(padding, Sequence) and not isinstance(padding, str)
               else [padding] * len(nfilters))

    if not len(nfilters) == len(filter_width) == len(dropout) == len(padding):
        raise ValueError("Parameter sequences have different lengths")

    def conv(incomming):
        conv_names = (name_template.format(i+1) for i in range(0, len(nfilters)))
        parameters = zip(conv_names, nfilters, filter_width, dropout, padding)
        cnn = reduce(stack_conv, parameters, incomming)
        return cnn

    return conv


def build_rnn(nsteps: Sequence[int],
              inp_drop: Optional[Union[float, Sequence[float]]]=None,
              rec_drop: Optional[Union[float, Sequence[float]]]=None,
              bidirectional: Union[Optional[str], Sequence[Optional[str]]]=None,
              stateful=False, layer=layers.LSTM) -> Callable:
    # TODO extend documentation
    # TODO add name template argument
    # TODO tests
    """
    :param nsteps:
    :param inp_drop:
    :param rec_drop:
    :param bidirectional:
    :param stateful: use stateful RNN-cells
    :param layer: a recurrent layer to use
    :return:
    """

    def stack_layers(prev, param: Tuple[str, int, float, float, str]):
        """
        :param prev: incomming keras layer
        :param param: [layer name, steps, input dropout, recurrent dropout,
        bidirectional]
        """
        name, steps, indrop, recdrop, bidir = param
        layer_ = layer(steps, dropout=indrop, recurrent_dropout=recdrop,
                       return_sequences=True, stateful=stateful)
        return (layers.Bidirectional(layer_, bidir) if bidir else layer_)(prev)

    bidir_is_seq = (isinstance(bidirectional, Sequence)
                    and not isinstance(bidirectional, str))
    bi = (bidirectional if bidir_is_seq else [bidirectional] * len(nsteps))
    inp_drop = (inp_drop if isinstance(inp_drop, Sequence) else
                [inp_drop or 0] * len(nsteps))
    rec_drop = (rec_drop if isinstance(rec_drop, Sequence) else
                [rec_drop or 0] * len(nsteps))

    if not len(nsteps) == len(rec_drop) == len(inp_drop) == len(bi):
        raise ValueError("Parameter sequences have different length")

    def rec(incomming):
        rec_names = ("rec{}".format(i) for i in range(1, len(nsteps)+1))
        parameters = zip(rec_names, nsteps, inp_drop, rec_drop, bi)
        rnn = reduce(stack_layers, parameters, incomming)
        return rnn

    return rec


def build_word_embeddings(embeddings: np.ndarray, maxsteps, mask: bool):
    # TODO docs
    words = layers.Input(shape=(None, maxsteps), dtype="int32")
    embeddings = layers.embeddings.Embedding(input_dim=embeddings.shape[0],
                                             output_dim=embeddings.shape[1],
                                             mask_zero=mask,
                                             weights=[embeddings])(words)
    return embeddings


def build_char_embeddings(nchar: int, embsize: int, units: int, maxsteps: int,
                          maxlen: bool, mask: bool, layer=layers.LSTM):
    # TODO docs
    chars = layers.Input(shape=(None, maxsteps, maxlen), dtype="int32")
    embeddings = layers.embeddings.Embedding(input_dim=nchar,
                                             output_dim=embsize,
                                             mask_zero=mask)(chars)
    shape = K.shape(embeddings)
    embeddings = layers.Lambda(
        lambda x: K.reshape(x, shape=(-1, shape[-2], embsize)))(embeddings)

    forward = layer(units, return_state=True)(embeddings)[-2]
    reverse = layer(units, return_state=True, go_backwards=True)(embeddings)[-2]
    embeddings = layers.concatenate([forward, reverse], axis=-1)
    # shape = (batch size, max sentence length, char hidden size)
    embeddings = layers.Lambda(
        lambda x: K.reshape(x, shape=[-1, shape[1], 2 * units]))(embeddings)
    return embeddings


if __name__ == "__main__":
    raise RuntimeError