import json
from io import TextIOWrapper
from itertools import chain
from functools import reduce
import operator as op
from typing import List, Tuple, Optional, Text, Pattern, Sequence, Mapping, \
    Callable, Union, Iterator, Sized, Iterable, overload, TypeVar, NamedTuple, \
    Container, Generic
from numbers import Integral
import csv
import sys
import re

import numpy as np
from sklearn.utils import class_weight
from fn import F
import pandas as pd


_slots_supported = (sys.version_info >= (3, 6, 2) or
                    (3, 5, 3) <= sys.version_info < (3, 6))


ClassMapping = Mapping[Text, Integral]
WS_PATT = re.compile("\S+")
PUNCT_PATT = re.compile(r"[\w]+|[^\s\w]")
PUNCT_WS_PATT = re.compile(r"[\w]+|[^\w]")

T = TypeVar("T")


class Interval(Container, Generic[T]):

    if _slots_supported:
        __slots__ = ("start", "stop", "data")

    def __init__(self, start: int, stop: int, data: Optional[T]=None):
        self.start = start
        self.stop = stop
        self.data = data

    def __contains__(self, item: T) -> bool:
        return False if self.data is None or item is None else self.data == item

    def __len__(self):
        return self.stop - self.start

    def __bool__(self):
        return bool(len(self))

    def __repr__(self):
        return "{}(start={}, stop={}, data={})".format(type(self).__name__,
                                                       self.start,
                                                       self.stop,
                                                       self.data)


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
        if retval is None:
            raise KeyError("No {} configuration".format(item))
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
        [200, 200, 200, 200]
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


class EmbeddingsWrapper(Mapping):
    def __init__(self, embeddings: Text, transform: Callable[[Text], Text],
                 oov="<unk>"):
        embeddings = pd.read_table(embeddings, sep=" ", index_col=0,
                                   header=None, quoting=csv.QUOTE_NONE)
        self.vectors = embeddings.as_matrix()
        self.token_index = {tk: i for i, tk in enumerate(embeddings.index)}
        self.transform = transform
        if oov not in self.token_index:
            raise ValueError("there is no `oov` among the embeddings")
        self.oov = oov

    def __getitem__(self, tokens: Union[Text, Iterable[Text]]):
        tokens_ = [tokens] if isinstance(tokens, Text) else tokens
        index = self.token_index
        oov_idx = self.token_index[self.oov]
        transform = self.transform
        return self.vectors[[index.get(transform(tk), oov_idx)
                             for tk in tokens_]]

    def __iter__(self) -> Iterator[Text]:
        return iter(self.token_index)

    def __len__(self) -> int:
        return len(self.token_index)

    def get(self, token: Text):
        tk = self.transform(token)
        return self.vectors[self.token_index.get(tk, self.token_index[self.oov])]


def parse(text: Text, pattern: Pattern) -> np.ndarray:
    # TODO tests
    """
    Tokenise text
    :param text: text to parse
    :param pattern: token pattern
    :return: a sorted array of matches intervals
    """
    try:
        intervals = [m.span() for m in pattern.finditer(text)]
        return np.array([Interval(start, end) for start, end in intervals])
    except TypeError:
        raise TypeError("`{}` is not a valid unicode string".format(repr(text)))


def sample_windows(intervals: Sequence[Interval], window: int) \
        -> Iterator[Sequence[Interval]]:
    # TODO update docs
    # TODO test
    """
    Sample windows using a sliding window approach. Sampling windows start at
    the beginning of each interval in `intervals`
    :param intervals: a sequence (preferable a numpy array) of interval objects
    :param window: sampling window width in tokens
    """
    samples = (
        iter([intervals]) if len(intervals) <= window else
        (intervals[i:i+window] for i in range(len(intervals)-window+1))
    )
    return samples


def sample_length(sample: Sequence[Interval]) -> int:
    # TODO docs
    return 0 if not len(sample) else sample[-1].stop - sample[0].start


def sample_span(sample: Sequence[Interval]) -> Optional[Interval]:
    return Interval(sample[0].start, sample[-1].stop) if len(sample) else None


def extract_intervals(sequence: Sequence[T], intervals: Iterable[Interval]) \
        -> List[Sequence[T]]:
    return [sequence[iv.start:iv.stop] for iv in intervals]


flatmap = F(map) >> chain.from_iterable


def join(arrays: List[np.ndarray], length: int, padval=0) \
        -> Tuple[np.ndarray, np.ndarray]:
    """
    Join 1D or 2D arrays. The function uses zero-padding to bring all arrays to the
    same length. The dtypes will be coerced to `dtype`
    :param arrays: arrays to join
    :param length: final sample length
    :param padval: padding value
    :param dtype: output data type (must be a numpy integral type)
    :return: (joined and padded arrays, boolean array masks); masks are
    positive, i.e. padded regions are False
    >>> import random
    >>> length = 100
    >>> ntests = 10000
    >>> arrays = [np.random.randint(0, 127, size=random.randint(1, length))
    ...           for _ in range(ntests)]
    >>> joined, masks = join(arrays, length)
    >>> all((arr == j[m]).all() for arr, j, m in zip(arrays, joined, masks))
    True
    """
    if length < max(map(len, arrays)):
        raise ValueError("Some arrays are longer than `length`")
    ndim = set(arr.ndim for arr in arrays)
    if ndim not in ({1}, {2}):
        raise ValueError("`arrays` must be a nonempty list of 2D or 3D arrays ")
    masks = np.zeros((len(arrays), length), dtype=bool)
    shape = ((len(arrays), length) if ndim == {1} else
             (len(arrays), length, arrays[0].shape[1]))
    dtype = arrays[0].dtype
    joined = (
        np.repeat([padval], reduce(op.mul, shape)).reshape(shape).astype(dtype))
    for i, arr in enumerate(arrays):
        joined[i, :len(arr)] = arr
        masks[i, :len(arr)] = True
    return joined, masks


def one_hot(array: np.ndarray) -> np.ndarray:
    """
    One-hot encode an integer array; the output inherits the array's dtype.
    >>> nclasses = 10
    >>> permutations = np.vstack([np.random.permutation(nclasses)
    ...                           for _ in range(nclasses)])
    >>> (one_hot(permutations).argmax(permutations.ndim) == permutations).all()
    True
    """
    if not np.issubdtype(array.dtype, np.int):
        raise ValueError("`array.dtype` must be integral")
    if not len(array):
        return array
    vectors = np.eye(array.max()+1, dtype=array.dtype)
    return vectors[array]


def maskfalse(array: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """
    Replace False-masked items with zeros.
    >>> array = np.arange(10)
    >>> mask = np.random.binomial(1, 0.5, len(array)).astype(bool)
    >>> masked = maskfalse(array, mask)
    >>> (masked[mask] == array[mask]).all()
    True
    >>> (masked[~mask] == 0).all()
    True
    """
    if not np.issubdtype(mask.dtype, np.bool):
        raise ValueError("Masks are supposed to be boolean")
    copy = array.copy()
    copy[~mask] = 0
    return copy


def parse_mapping(classmaps: Iterable[str]) -> ClassMapping:
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


def balance_class_weights(y: np.ndarray, mask: Optional[np.ndarray]=None) \
        -> Optional[Mapping[int, float]]:
    """
    :param y: a numpy array encoding sample classes; samples are encoded along
    the 0-axis
    :param mask: a boolean array of shape compatible with `y`, wherein True
    shows that the corresponding value(s) in `y` should be used to calculate
    weights; if `None` the function will consider all values in `y`
    :return: class weights
    """
    if not len(y):
        raise ValueError("`y` is empty")
    y_flat = (y.flatten() if mask is None else
              np.concatenate([sample[mask] for sample, mask in zip(y, mask)]))
    classes = np.unique(y_flat)
    weights = class_weight.compute_class_weight("balanced", classes, y_flat)
    weights_scaled = weights / weights.min()
    return {cls: weight for cls, weight in zip(classes, weights_scaled)}


def sample_weights(y: np.ndarray, class_weights: Mapping[int, float]) \
        -> np.ndarray:
    """
    :param y: a 2D array encoding sample classes; each sample is a row of
    integers representing class code
    :param class_weights: a class to weight mapping
    :return: a 2D array of the same shape as `y`, wherein each position stores
    a weight for the corresponding position in `y`
    """
    weights_mask = np.zeros(shape=y.shape, dtype=np.float32)
    for cls, weight in class_weights.items():
        weights_mask[y == cls] = weight
    return weights_mask


# def merge_predictions(intervals: List[Interval], predictions: np.ndarray) \
#         -> np.ndarray:
#     """
#     :param intervals: intervals (non-inclusive on the right side)
#     :param predictions:
#     :return:
#     >>> randints = np.random.randint(0, 1000, size=20)
#     >>> intervals = sorted([tuple(sorted(randints[i:i+2]))
#     ...                     for i in range(0, len(randints), 2)])
#     >>> maxlen = max(end - start for start, end in intervals)
#     >>> predictions = np.zeros((len(intervals), maxlen), dtype=float)
#     >>> for i, (start, end) in enumerate(intervals):
#     ...     predictions[i, :end-start] = np.random.uniform(0, 1, size=end-start)
#     >>> manual = [[] for _ in range(max(chain.from_iterable(intervals)))]
#     >>> for (i, (start, end)), pred in zip(enumerate(intervals), predictions):
#     ...     for j, value in zip(range(start, end), pred[:end-start]):
#     ...         manual[j].append(value)
#     >>> means_man =  np.array([np.mean(values) if values else np.nan
#     ...                       for values in manual])
#     >>> means_func = merge_predictions(intervals, predictions)
#     >>> nan_man = np.isnan(means_man)
#     >>> nan_func = np.isnan(means_func)
#     >>> (nan_man == nan_func).all()
#     True
#     >>> (means_man[~nan_man].round(3) == means_func[~nan_func].round(3)).all()
#     True
#     """
#     # the intervals are half-inclusive and zero-indexed
#     length = max(chain.from_iterable(intervals))
#     buckets = np.zeros(length, dtype=np.float64)
#     nsamples = np.zeros(length, dtype=np.int32)
#     for (start, end), pred in zip(intervals, predictions):
#         # `predictions` are zero-padded – we must remove the padded tail
#         sample_length = end - start
#         buckets[start:end] += pred[:sample_length]
#         nsamples[start:end] += np.ones(sample_length, dtype=np.int32)
#     with np.errstate(divide='ignore', invalid="ignore"):
#         return buckets / nsamples


if __name__ == "__main__":
    raise RuntimeError
