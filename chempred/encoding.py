"""



"""


from typing import List, Mapping, Tuple

import numpy as np
from enforce import runtime_validation

from chempred.chemdner import Annotation


MAXCHAR = 127


def encode_sample_chars(text: str, sample: List[Annotation], dtype=np.int32) \
        -> np.ndarray:
    # TODO tests
    """
    Encode text cahracters at each position of the sample
    :param text: the complete text from which the sample was drawn
    :param sample: a list of annotations
    :param dtype: output data type; it must be an integral numpy dtype
    :return: an integer array
    """
    if not np.issubdtype(dtype, np.int):
        raise ValueError("`dtype` must be integral")
    start, end = sample[0].start, sample[-1].end
    length = end - start
    encoded = np.fromiter(map(ord, text[start:end]), dtype, length)
    encoded[encoded >= MAXCHAR] = MAXCHAR
    return encoded


def encode_sample_classes(mapping: Mapping[str, int], sample: List[Annotation],
                          dtype=np.int32) \
        -> np.array:
    # TODO tests
    """
    Encode classes at each position of the sample
    :param mapping: a mapping from string classes into integers
    :param sample: a list of annotations
    :param dtype: output data type; it must be an integral numpy dtype
    :return: an integer array
    """
    if not np.issubdtype(dtype, np.int):
        raise ValueError("`dtype` must be integral")
    offset = sample[0].start
    length = sample[-1].end - offset
    encoded = np.zeros(length, dtype=dtype)
    for _, start, end, _, cls in sample:
        encoded[start-offset:end-offset] = mapping.get(cls, 0)
    return encoded


def encode_annotation(mapping: Mapping[str, int], anno: Annotation) \
        -> Tuple[np.ndarray, np.ndarray]:
    # TODO docs
    # TODO tests
    txt = np.fromiter(map(ord, anno.text), dtype=np.int32, count=len(anno.text))
    txt[txt >= MAXCHAR] = MAXCHAR
    cls = np.array([mapping.get(anno.cls, 0)] * len(anno.text), dtype=np.int32)
    return txt, cls


@runtime_validation
def join(arrays: List[np.ndarray], length: int, dtype=np.int32) \
        -> Tuple[np.ndarray, np.ndarray]:
    """
    Join 1D arrays. The function uses zero-padding to bring all arrays to the
    same length. The dtypes will be coerced to `dtype`
    :return: (joined and padded arrays, boolean array masks); masks are
    positive, i.e. padded regions are False
    >>> import random
    >>> arrays = [np.random.randint(0, 127, size=random.randint(1, 101))
    ...           for _ in range(100)]
    >>> joined, masks = join(arrays)
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


if __name__ == "__main__":
    raise RuntimeError
