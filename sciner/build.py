"""

Utility functions for creating ChemPred deep learning models and working with
their predictions

"""
from functools import reduce
from typing import Sequence, Tuple, Optional, Union, Callable

from keras import layers, models

from sciner import encoding

NCHAR = encoding.MAXCHAR + 1


def build_conv(nfilters: Sequence[int],
               filter_width: Union[int, Sequence[int]],
               dropout: Union[Optional[float], Sequence[Optional[float]]]=None,
               name_template: str="l_conv_{}") \
        -> Callable:
    # TODO extend documentation
    # TODO more tests
    """

    :param nfilters:
    :param filter_width:
    :return:
    >>> conv = build_conv([30, 30], 5)
    """
    def stack_conv(prev, param: Tuple[str, int, int, float]):
        name, nfilt, kern_size, drop_p = param
        l = layers.Convolution1D(
            nfilt, kern_size, activation="relu", name=name,
        )(prev)
        return layers.Dropout(drop_p)(l) if drop_p else l

    filter_width = (filter_width if isinstance(filter_width, Sequence) else
                    [filter_width] * len(nfilters))
    dropout = (dropout if isinstance(dropout, Sequence) else
               [dropout] * len(nfilters))

    if not len(nfilters) == len(filter_width) == len(dropout):
        raise ValueError("Parameter sequences have different lengths")

    def conv(incomming):
        conv_names = (name_template.format(i+1) for i in range(0, len(nfilters)))
        parameters = zip(conv_names, nfilters, filter_width, dropout)
        cnn = reduce(stack_conv, parameters, incomming)
        return cnn

    return conv


def build_rec(nsteps: Sequence[int],
              inp_drop: Optional[Union[float, Sequence[float]]]=None,
              rec_drop: Optional[Union[float, Sequence[float]]]=None,
              bidirectional: Union[bool, Sequence[bool]]=False,
              stateful=False, layer=layers.LSTM) -> Callable:
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

    def stack_layers(prev, param: Tuple[str, int, float, float, bool]):
        """
        :param prev: incomming keras layer
        :param param: [layer name, steps, input dropout, recurrent dropout,
        bidirectional]
        """
        name, steps, indrop, recdrop, bidir = param
        layer_ = layer(steps, dropout=indrop, recurrent_dropout=recdrop,
                       return_sequences=True, stateful=stateful)
        return (layers.Bidirectional(layer_) if bidir else layer_)(prev)

    bi = (bidirectional if isinstance(bidirectional, Sequence) else
          [bidirectional] * len(nsteps))
    inp_drop = (inp_drop if isinstance(inp_drop, Sequence) else
                [inp_drop or 0] * len(nsteps))
    rec_drop = (rec_drop if isinstance(rec_drop, Sequence) else
                [rec_drop or 0] * len(nsteps))

    if not len(nsteps) == len(rec_drop) == len(inp_drop) == len(bi):
        raise ValueError("Parameter sequences have different length")

    def rec(incomming):
        rec_names = ("rec_{}".format(i) for i in range(1, len(nsteps)+1))
        parameters = zip(rec_names, nsteps, map(float, inp_drop),
                         map(float, rec_drop), bi)
        rnn = reduce(stack_layers, parameters, incomming)
        return rnn

    return rec


if __name__ == "__main__":
    raise RuntimeError
