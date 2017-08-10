"""



"""
from typing import Sequence, Tuple, Optional, Union
from functools import reduce
from fn import F

from keras import backend as k
from keras import losses
from keras import models
from keras import layers
import numpy as np
import click


nchar = 256
maxlen = 200


def weighted_binary_crossentropy(y_true, y_pred):
    weights = 1
    return losses.binary_crossentropy(y_pred, y_true) * weights


def stack_conv(prev: layers.Layer, param: Tuple[str, int, int]):
    name, nfilt, kern_size = param
    return layers.Convolution1D(
        nfilt, kern_size, activation="relu", name=name,
    )(prev)


def build_conv(incomming,
               filters: Optional[Sequence[int]],
               kernels: Optional[Sequence[int]]):
    filters = filters or []
    kernels = kernels or []
    assert len(filters) == len(kernels)

    conv_names = ("conv_{}".format(i) for i in range(1, len(kernels)+1))
    conv = reduce(stack_conv, zip(conv_names, filters, kernels), incomming)
    return conv


def stack_lstm(prev: layers.Layer, param: Tuple[str, int, float, float],
               bidirectional: bool, stateful: bool):
    name, units, indrop, recdrop = param
    layer = layers.LSTM(units, dropout=indrop, recurrent_dropout=recdrop,
                        return_sequences=True, stateful=stateful)
    return (layers.Bidirectional(layer) if bidirectional else layer)(prev)


def build_rec(nsteps: Optional[Sequence[int]],
              lstm_inp_drop: Optional[Sequence[float]],
              lstm_rec_drop: Optional[Sequence[float]],
              bidirectional=True,
              stateful=False):

    nsteps = nsteps or []
    lstm_inp_drop = lstm_inp_drop or []
    lstm_rec_drop = lstm_rec_drop or []
    assert len(nsteps) == len(lstm_rec_drop) == len(lstm_inp_drop)

    def rec(incomming):
        rec_names = ("rec_{}".format(i) for i in range(1, len(nsteps)+1))
        recur = reduce(
            F(stack_lstm, bidirectional=bidirectional, stateful=stateful),
            zip(rec_names, nsteps, lstm_inp_drop, lstm_rec_drop),
            incomming
        )
        return layers.TimeDistributed(
            layers.Dense(nchar, activation="softmax"), name="decoded"
        )(recur)

    return rec
