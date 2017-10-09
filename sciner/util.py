import operator as op
from functools import reduce
from itertools import chain, repeat
from typing import List, Tuple

import numpy as np
from fn import F

flatmap = F(map) >> chain.from_iterable


def flatzip(flat, nested):
    flatrep = map(F(map, repeat), flat)
    iterables = (*flatrep, *nested)
    return (F(zip) >> F(map, lambda x: zip(*x)) >> chain.from_iterable)(*iterables)


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
