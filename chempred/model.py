"""

Utility functions for creating ChemPred deep learning models and working with
their predictions

"""
import json
from typing import Sequence, Tuple, Optional, List, Union, Callable, Mapping
from io import TextIOWrapper
from functools import reduce
from itertools import chain

from enforce import runtime_validation
from keras import layers, models
import numpy as np

from chempred import encoding
from chempred.chemdner import Interval


NCHAR = encoding.MAXCHAR + 1


class Config(dict):
    """
    Model configurations
    """
    def __init__(self, config: Union[TextIOWrapper, Mapping]):
        """
        :param config: an opened json file or a mapping
        """
        super(Config, self).__init__(config if isinstance(config, Mapping) else
                                     json.load(config))

    def __getitem__(self, item):
        retval = self.get(item)
        if not retval:
            raise KeyError("Not {} configuration".format(item))
        return retval

    def get(self, item, default=None):
        """
        :param item: item to search for
        :param default: default value
        :return:
        >>> config = Config(open("testdata/config-detector.json"))
        >>> config["bidirectional"]
        True
        >>> config.get("epochs")
        >>> from_dict = Config(json.load(open("testdata/config-detector.json")))
        >>> from_mapping = Config(from_dict)
        >>> from_mapping["nsteps"]
        [200, 200]
        >>> from_mapping.update({"lstm": {"nsteps": [300, 300]}})
        >>> isinstance(from_mapping, Config)
        True
        >>> from_mapping["nsteps"]
        [300, 300]
        """
        to_visit = list(self.items())
        while to_visit:
            key, value = to_visit.pop()
            if key == item:
                return value
            if isinstance(value, Mapping):
                to_visit.extend(value.items())
        return default


def parse_mapping(classmaps: Sequence[str]) -> Mapping[str, int]:
    """
    :param classmaps:
    :return:
    >>> classmaps = ["a:1", "b:1", "c:2"]
    >>> parse_mapping(classmaps) == dict(a=1, b=1, c=2)
    True
    """
    try:
        return {cls: int(val)
                for cls, val in [classmap.split(":") for classmap in classmaps]}
    except ValueError as err:
        raise ValueError("Badly formatted mapping: {}".format(err))


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
    @runtime_validation
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

    @runtime_validation
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


def merge_predictions(intervals: List[Interval], predictions: np.ndarray) \
        -> np.ndarray:
    """
    :param intervals: intervals (non-inclusive on the right side)
    :param predictions:
    :return:
    >>> randints = np.random.randint(0, 1000, size=20)
    >>> intervals = sorted([tuple(sorted(randints[i:i+2]))
    ...                     for i in range(0, len(randints), 2)])
    >>> maxlen = max(end - start for start, end in intervals)
    >>> predictions = np.zeros((len(intervals), maxlen), dtype=float)
    >>> for i, (start, end) in enumerate(intervals):
    ...     predictions[i, :end-start] = np.random.uniform(0, 1, size=end-start)
    >>> manual = [[] for _ in range(max(chain.from_iterable(intervals)))]
    >>> for (i, (start, end)), pred in zip(enumerate(intervals), predictions):
    ...     for j, value in zip(range(start, end), pred[:end-start]):
    ...         manual[j].append(value)
    >>> means_man =  np.array([np.mean(values) if values else np.nan
    ...                       for values in manual])
    >>> means_func = merge_predictions(intervals, predictions)
    >>> nan_man = np.isnan(means_man)
    >>> nan_func = np.isnan(means_func)
    >>> (nan_man == nan_func).all()
    True
    >>> (means_man[~nan_man].round(3) == means_func[~nan_func].round(3)).all()
    True
    """
    # the intervals are half-inclusive and zero-indexed
    length = max(chain.from_iterable(intervals))
    buckets = np.zeros(length, dtype=np.float64)
    nsamples = np.zeros(length, dtype=np.int32)
    for (start, end), pred in zip(intervals, predictions):
        # `predictions` are zero-padded – we must remove the padded tail
        sample_length = end - start
        buckets[start:end] += pred[:sample_length]
        nsamples[start:end] += np.ones(sample_length, dtype=np.int32)
    with np.errstate(divide='ignore', invalid="ignore"):
        return buckets / nsamples


def build_nn(maxlen: int,
             embed: int,
             nfilters: Optional[Sequence[int]],
             filter_width: Optional[Union[int, Sequence[int]]],
             nsteps: Sequence[int],
             in_drop: Union[float, Sequence[float]],
             rec_drop: Union[float, Sequence[float]],
             bidirectional: Union[bool, Sequence[bool]],
             stateful: bool):
    # TODO tests
    # TODO documentation
    without_cnn = not bool(nfilters)
    l_in = layers.Input(shape=(maxlen,), name="l_in")
    l_emb = layers.Embedding(NCHAR, embed, input_length=maxlen,
                             mask_zero=without_cnn)(l_in)

    if without_cnn:  # build a pure rnn-model
        rnn = build_rec(nsteps, in_drop, rec_drop, bidirectional, stateful)(l_emb)
        l_out = layers.TimeDistributed(
            layers.Dense(2, activation='softmax'), name="l_out")(rnn)
    # build a cnn-rnn model
    cnn = build_conv(nfilters, filter_width)(l_emb)
    l_flat = layers.Flatten(name="flat_1")(cnn)
    l_repeat = layers.RepeatVector(maxlen, name="repeat")(l_flat)
    rnn = build_rec(nsteps, in_drop, rec_drop, bidirectional, stateful)
    l_out = layers.TimeDistributed(
        layers.Dense(2, activation='softmax'), name="l_out")(rnn)
    return models.Model(l_in, l_out)




if __name__ == "__main__":
    raise RuntimeError
