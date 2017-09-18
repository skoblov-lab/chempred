import csv
import json
import operator as op
import re
import sys
from functools import reduce
from io import TextIOWrapper
from itertools import chain
from typing import List, Tuple, Optional, Text, Sequence, Mapping, \
    Callable, Union, Iterator, Iterable, TypeVar, Container, Generic

import numpy as np
import pandas as pd
from fn import F

_slots_supported = (sys.version_info >= (3, 6, 2) or
                    (3, 5, 3) <= sys.version_info < (3, 6))

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

    def __eq__(self, other: "Interval"):
        return (self.start, self.stop, self.data) == (other.start, other.stop, other.data)

    def __hash__(self):
        return hash((self.start, self.stop, self.data))

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


def one_hot(ncls: int, array: np.ndarray) -> np.ndarray:
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
    vectors = np.eye(ncls, dtype=array.dtype)
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


if __name__ == "__main__":
    raise RuntimeError
