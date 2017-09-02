"""

Utility functions for creating ChemPred deep learning models and working with
their predictions

"""
from functools import reduce
from typing import Sequence, Tuple, Optional, Union, Callable

from keras import layers, models

from chempred import encoding

NCHAR = encoding.MAXCHAR + 1


def build_conv(nfilters: Sequence[int],
               filter_width: Union[int, Sequence[int]]) \
        -> Callable:
    # TODO extend documentation
    # TODO more tests
    """

    :param nfilters:
    :param filter_width:
    :return:
    >>> conv = build_conv([30, 30], 5)
    """
    def stack_conv(prev, param: Tuple[str, int, int]):
        name, nfilt, kern_size = param
        return layers.Convolution1D(
            nfilt, kern_size, activation="relu", name=name,
        )(prev)

    filter_width = (filter_width if isinstance(filter_width, Sequence) else
                    [filter_width] * len(nfilters))

    if not len(nfilters) == len(filter_width):
        raise ValueError("Parameter sequences have different length")

    def conv(incomming):
        conv_names = ("conv_{}".format(i) for i in range(1, len(nfilters)+1))
        parameters = zip(conv_names, nfilters, filter_width)
        cnn = reduce(stack_conv, parameters, incomming)
        return cnn

    return conv


def build_rec(nsteps: Sequence[int],
              lstm_inp_drop: Optional[Union[float, Sequence[float]]]=None,
              lstm_rec_drop: Optional[Union[float, Sequence[float]]]=None,
              bidirectional: Union[bool, Sequence[bool]]=False,
              stateful=False) -> Callable:
    # TODO extend documentation
    """
    :param nsteps:
    :param lstm_inp_drop:
    :param lstm_rec_drop:
    :param bidirectional:
    :param stateful: use stateful LSTM-cells
    :return:
    >>> rec = build_rec([200, 200], 0.1, 0.1, True)
    """

    def stack_lstm(prev, param: Tuple[str, int, float, float, bool]):
        """
        :param prev: incomming keras layer
        :param param: [layer name, steps, input dropout, recurrent dropout,
        bidirectional]
        """
        name, steps, indrop, recdrop, bidir = param
        layer = layers.LSTM(steps, dropout=indrop, recurrent_dropout=recdrop,
                            return_sequences=True, stateful=stateful)
        return (layers.Bidirectional(layer) if bidir else layer)(prev)

    bi = (bidirectional if isinstance(bidirectional, Sequence) else
          [bidirectional] * len(nsteps))
    inp_drop = (lstm_inp_drop if isinstance(lstm_inp_drop, Sequence) else
                [lstm_inp_drop or 0] * len(nsteps))
    rec_drop = (lstm_rec_drop if isinstance(lstm_rec_drop, Sequence) else
                [lstm_rec_drop or 0] * len(nsteps))

    if not len(nsteps) == len(rec_drop) == len(inp_drop) == len(bi):
        raise ValueError("Parameter sequences have different length")

    def rec(incomming):
        rec_names = ("rec_{}".format(i) for i in range(1, len(nsteps)+1))
        parameters = zip(rec_names, nsteps, map(float, inp_drop),
                         map(float, rec_drop), bi)
        rnn = reduce(stack_lstm, parameters, incomming)
        return rnn

    return rec


def build_nn(sample_size: int,
             embed: int,
             ncls: int,
             nfilters: Optional[Sequence[int]],
             filter_width: Optional[Union[int, Sequence[int]]],
             nsteps: Sequence[int],
             in_drop: Union[float, Sequence[float]],
             rec_drop: Union[float, Sequence[float]],
             bidirectional: Union[bool, Sequence[bool]],
             stateful: bool=False):
    # TODO tests
    # TODO documentation
    no_cnn = not bool(nfilters)
    l_in = layers.Input(shape=(sample_size,), name="l_in")
    encoder = layers.Embedding(
        NCHAR, embed, input_length=sample_size, mask_zero=no_cnn)(l_in)
    if not no_cnn:
        encoder = build_conv(nfilters, filter_width)(encoder)
        encoder = layers.Flatten(name="flat")(encoder)
        encoder = layers.RepeatVector(sample_size, name="repeat")(encoder)
    decoder = build_rec(
        nsteps, in_drop, rec_drop, bidirectional, stateful)(encoder)
    l_out = layers.TimeDistributed(
        layers.Dense(ncls, activation='softmax'), name="l_out")(decoder)
    return models.Model(l_in, l_out)


if __name__ == "__main__":
    raise RuntimeError
