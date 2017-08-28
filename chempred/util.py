from typing import List, Tuple, Optional, Text, Pattern, Sequence, Mapping
import re

import numpy as np
from enforce import runtime_validation

from chempred.intervals import Interval, Intervals


Token = Interval[Optional[Text]]


def tokenise(text: Text, inflate=False, pattern: Pattern=re.compile("\S+")) \
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


if __name__ == "__main__":
    raise RuntimeError


