"""



"""

from typing import Mapping, Tuple, Text, Iterable, List
from itertools import chain

import numpy as np
from frozendict import frozendict
from fn.func import identity

from sciner.intervals import Interval
from sciner.util import oldmap, homogenous


MAXCLS = 255


class EmbeddingError(ValueError):
    pass


class EncodingError(ValueError):
    pass


class Vocabulary:
    """
    Zero is reserved for padding
    """

    def __init__(self, path: Text, oov: Text, transform=None):
        self.oov = oov
        self.transform = transform if transform else identity
        word_index, vectors = self._read_embeddings(path)
        self.vocab = word_index
        self.vectors = vectors

    def __str__(self):
        return "<Vocabulary> with {} entries".format(len(self.vocab))

    def encode(self, words: Iterable[Text]) -> List[int]:
        oov = self.vocab[self.oov]
        return [self.vocab.get(w, oov) for w in map(self.transform, words)]

    def _read_embeddings(self, path) -> Tuple[Mapping[str, int], np.ndarray]:
        with open(path) as lines:
            parsed = map(str.split, lines)
            words, vectors = zip(*((w, oldmap(float, v)) for w, *v in parsed))
        if not words:
            raise EmbeddingError("File {} is empty".format(path))
        if not homogenous(len, vectors):
            raise EmbeddingError("Word vectors must be homogeneous")
        ndim = len(vectors[0])
        padvec = [0.0] * ndim
        word_index = frozendict({word: i+1 for i, word in enumerate(words)})
        vectors_ = np.array(list(chain([padvec], vectors)))
        vectors_.flags["WRITEABLE"] = False
        return word_index, vectors_


class Alphabet:
    pass


def encode_annotation(annotations: Iterable[Interval], size: int) -> np.ndarray:
    # TODO update docs
    """
    Default class is 0.
    :param annotations:
    :param size:
    :return:
    """
    encoded_anno = np.zeros(size, dtype=np.uint8)
    for anno in annotations:
        if anno.stop > size:
            raise EncodingError("annotation `size` is insufficient")
        cls = anno.data
        if not 0 <= cls <= MAXCLS:
            raise EncodingError("class codes must be in [0, {}]".format(MAXCLS))
        encoded_anno[anno.start:anno.stop] = anno.data
    return encoded_anno


if __name__ == "__main__":
    pass

