"""



"""

from typing import Mapping, Tuple, Text, Iterable, List
from itertools import chain

import numpy as np
from frozendict import frozendict
from fn.func import identity

from sciner.intervals import Interval
from sciner.util import oldmap, homogenous


MAXLABEL = 255


class EncodingError(ValueError):
    pass


class WordEncoder:
    """
    Zero is reserved for padding
    """

    def __init__(self, path: Text, oov: Text, transform=None):
        self._oov = oov
        self._transform = transform if transform else identity
        word_index, vectors = self._read_embeddings(path)
        self._vocab = word_index
        self._vectors = vectors

    def __str__(self):
        return "<Vocabulary> with {} entries".format(len(self._vocab))

    def __len__(self):
        return len(self._vectors)  # including the pad and oov words

    @property
    def vocab(self):
        return self._vocab

    @property
    def vectors(self):
        return self._vectors

    @property
    def transform(self):
        return self._transform

    @property
    def oov(self):
        return self._oov

    def encode(self, words: Iterable[Text], ids=True) -> np.ndarray:
        oov = self._vocab[self._oov]
        ids_ = [self._vocab.get(w, oov) for w in map(self._transform, words)]
        return np.array(ids_, dtype=np.int32) if ids else self._vectors[ids_]

    @staticmethod
    def _read_embeddings(path) -> Tuple[Mapping[str, int], np.ndarray]:
        with open(path) as lines:
            parsed = map(str.split, lines)
            words, vectors = zip(*((w, oldmap(float, v)) for w, *v in parsed))
        if not words:
            raise EncodingError("File {} is empty".format(path))
        if not homogenous(len, vectors):
            raise EncodingError("Word vectors must be homogeneous")
        ndim = len(vectors[0])
        padvec = [0.0] * ndim
        word_index = frozendict({word: i+1 for i, word in enumerate(words)})
        vectors_ = np.array(list(chain([padvec], vectors)))
        vectors_.setflags(write=False)
        return word_index, vectors_


class CharEncoder:
    """
    Zero is reserved for the padding value
    """

    def __init__(self, path):
        self._chars = self._read_characters(path)
        self._oov = len(self._chars) + 1

    def __len__(self):
        return len(self._chars) + 2  # including the pad and oov characters

    @property
    def oov(self):
        return self._oov

    @property
    def characters(self):
        return self._chars

    @staticmethod
    def _read_characters(path):
        with open(path) as lines:
            characters = frozenset(chain.from_iterable(map(str.strip, lines)))
            char_index = frozendict(
                {char: i+1 for i, char in enumerate(characters)})
        if not characters:
            raise EncodingError("file {} is empty".format(path))
        return char_index

    def encode(self, words: Iterable[Text]) -> List[np.ndarray]:
        oov = self._oov
        return [np.array([self._chars.get(c, oov) for c in w], dtype=np.int32)
                for w in words]


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
        if not 0 <= cls <= MAXLABEL:
            raise EncodingError("class codes must be in [0, {}]".format(MAXLABEL))
        encoded_anno[anno.start:anno.stop] = anno.data
    return encoded_anno


if __name__ == "__main__":
    pass
