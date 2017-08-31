from typing import List, Tuple, Optional, Text, Pattern, Sequence, Mapping, Callable, Iterable
from numbers import Integral
import re

import numpy as np
from enforce import runtime_validation
from sklearn.utils import class_weight

from chempred.intervals import Interval, Intervals


Tokeniser = Callable[[Text], Intervals[Interval[Text]]]
Token = Interval[Optional[Text]]
ClassMapping = Mapping[Text, Integral]

WS_PATT = re.compile("\S+")
PUNCT_PATT = re.compile(r"[\w]+|[^\s\w]")
PUNCT_WS_PATT = re.compile(r"[\w]+|[^\w]")


def tokenise(text: Text, pattern: Pattern=WS_PATT, inflate=False) \
        -> Intervals[Token]:
    # TODO tests
    """
    Tokenise text
    :param text: text to parse
    :param inflate: store token's text inside the tokens
    :param pattern: token pattern
    :return: a sorted list of tokens
    """
    intervals = [m.span() for m in pattern.finditer(text)]
    return Intervals(Interval(start, end, text[start:end] if inflate else None)
                     for start, end in intervals)


@runtime_validation
def join(arrays: List[np.ndarray], length: int, padval: int=0, dtype=np.int32) \
        -> Tuple[np.ndarray, np.ndarray]:
    """
    Join 1D arrays. The function uses zero-padding to bring all arrays to the
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
    if not np.issubdtype(dtype, np.int):
        raise ValueError("`dtype` must be integral")

    ndim = set(arr.ndim for arr in arrays)
    if ndim != {1}:
        raise ValueError("`arrays` must be a nonempty list of 1D numpy arrays")
    if length < max(map(len, arrays)):
        raise ValueError("Some arrays are longer than `length`")
    joined = np.zeros((len(arrays), length), dtype=dtype)
    joined[:] = padval
    masks = np.zeros((len(arrays), length), dtype=bool)
    for i, arr in enumerate(arrays):
        joined[i, :len(arr)] = arr
        masks[i, :len(arr)] = True
    return joined, masks


@runtime_validation
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
    vectors = np.eye(array.max()+1, dtype=array.dtype)
    return vectors[array]


@runtime_validation
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


def parse_mapping(classmaps: Sequence[str]) -> ClassMapping:
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
        -> Mapping[int, float]:
    """
    :param y: a 2D array encoding sample classes; each sample is a row of
    integers representing class codes
    :param mask: a boolean array of the same shape as `y`, wherein True shows
    that the corresponding value in `y` should be used to calculate weights;
    if `None` the function will consider all values in `y`
    :return: class weights
    """
    y_flat = (y.flat() if mask is None else
              np.concatenate([sample[mask] for sample, mask in zip(y, mask)]))
    classes = np.unique(y_flat)
    weights = class_weight.compute_class_weight("balanced", classes, y_flat)
    return {cls: weight for cls, weight in zip(classes, weights)}


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


if __name__ == "__main__":
    raise RuntimeError
