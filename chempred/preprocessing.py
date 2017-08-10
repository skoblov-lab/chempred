from itertools import chain

import numpy as np


SPACE = ord(" ")


def join_lists(sep, lists):
    """
    :type sep: object
    :type lists: List[List]
    :return: List
    >>> join_lists(3, [[1, 2], [4, 5]])
    [1, 2, 3, 4, 5]
    >>> join_lists(0, [[1], [2], [3]])
    [1, 0, 2, 0, 3]
    """
    lst_it = iter(lists)
    it = (next(lst_it) if not i % 2 else [sep] for i in range(len(lists)*2 - 1))
    return list(chain.from_iterable(it))


def enumerate_positive_classes(annotated_tokens, positive_classes):
    """
    :type annotated_tokens: List[Tuple[str, int, int, str, str]]
    :param annotated_tokens:
    :type positive_classes: Set[str]
    :rtype: List[int]
    """
    return [i for i, token in enumerate(annotated_tokens) if token[-1] in positive_classes]


def slide_through(tokens, central_idx, width, flanking=False):
    """
    :type tokens: List[Any]
    :param central_idx: central points to slide around
    :type window: int
    :type minlen: int
    :rtype: List[List[Any]]
    >>> from pprint import pprint
    >>> tokens = list(range(10))
    >>> pprint(slide_through(tokens, [3], 3, True))
    [[0, 1, 2], [1, 2, 3], [2, 3, 4], [3, 4, 5], [4, 5, 6]]
    >>> pprint(slide_through(tokens, [3], 5, True))
    [[0, 1, 2, 3, 4],
     [1, 2, 3, 4, 5],
     [2, 3, 4, 5, 6],
     [3, 4, 5, 6, 7],
     [4, 5, 6, 7, 8]]
    >>> pprint(slide_through(tokens, [5], 3, True))
    [[2, 3, 4], [3, 4, 5], [4, 5, 6], [5, 6, 7], [6, 7, 8]]
    >>> pprint(slide_through(tokens, [5], 3, False))
    [[3, 4, 5], [4, 5, 6], [5, 6, 7]]
    >>> pprint(slide_through(tokens, [], 3, True))
    []
    >>> pprint(slide_through(tokens, [10], 3, True))
    [[7, 8, 9]]

    """
    width_corr, end_corr = (width, 2) if flanking else (width - 1, 1)
    maxidx = len(tokens) - width + 1
    windows = [[(i, i + width) for i in range(max(0, idx-width_corr),
                                              min(idx+end_corr, maxidx))]
               for idx in central_idx]
    return [tokens[start:end] for start, end in chain.from_iterable(windows)]


def generate_training_samples(tokenised_abstracts, positive_classes, window=5,
                              flanking=False):
    """
    :type tokenised_abstracts: List[List[Tuple[str, int, int, str, str]]]
    :param tokenised_abstracts: abstracts tokenised by chempred.chemdner.annotate_abstract
    :type positive_classes: Set[str]
    :param positive_classes: a list of token classes to regard as positive
    :rtype: List[List[str]], List[List[bool]]
    :return: sampled windows, token is positive
    >>> from pprint import pprint
    >>> positive_classes = {"a", "b"}
    >>> tokens = [("src", 0, 1, "text1", "c"),
    ...           ("src", 1, 2, "text2", "a"),
    ...           ("src", 3, 4, "text3", "c"),
    ...           ("src", 5, 6, "text4", "b"),
    ...           ("src", 7, 8, "text5", "c")]
    >>> abstracts = [tokens, tokens[:-2]]
    >>> pprint(generate_training_samples(abstracts, positive_classes, 3, True))
    ([['text1', 'text2', 'text3'],
      ['text2', 'text3', 'text4'],
      ['text3', 'text4', 'text5'],
      ['text1', 'text2', 'text3'],
      ['text2', 'text3', 'text4'],
      ['text3', 'text4', 'text5'],
      ['text1', 'text2', 'text3']],
     [[False, True, False],
      [True, False, True],
      [False, True, False],
      [False, True, False],
      [True, False, True],
      [False, True, False],
      [False, True, False]])
    """
    positive = [enumerate_positive_classes(tokens, positive_classes)
                for tokens in tokenised_abstracts]
    windows = [slide_through(tokens, pos, window, flanking)
               for tokens, pos in zip(tokenised_abstracts, positive)]
    # flatten structures (i.e. merge abstracts)
    windows_flat = list(chain.from_iterable(windows))
    # split tokens into text and annotation
    text = [[token[3]for token in window] for window in windows_flat]
    classes = [[token[-1] in positive_classes for token in window] for window in windows_flat]
    return text, classes


def join_tokens_in_samples(samples, classes):
    """
    :type text: List[List[str]]
    :type classes: List[List[bool]]
    :rtype: List[List[int]], List[List[bool]]
    >>> from pprint import pprint
    >>> samples = [['a', 'b', 'c'],
    ...            ['d', 'e', 'Å']]
    >>> classes = [[False, True, False],
    ...            [False, True, True]]
    >>> pprint(join_tokens_in_samples(samples, classes))
    ([[97, 32, 98, 32, 99], [100, 32, 101, 32, 195, 133]],
     [[False, False, True, False, False], [False, False, True, False, True, True]])
    """
    encoded = [[list(t.encode("utf-8")) for t in sample] for sample in samples]
    classes_joined = [join_lists(False, [[t_cls] * len(t) for t_cls, t in zip(s_cls, s_encoded)])
                      for s_cls, s_encoded in zip(classes, encoded)]
    samples_joined = [join_lists(SPACE, s_encoded) for s_encoded in encoded]
    return samples_joined, classes_joined


def pad(samples, classes):
    """
    :type samples: List[List[int]]
    :type classes: List[List[bool]]
    :rtype: np.array[int], np.array[bool], np.array[bool]
    :return: padded samples, padded classes, mask
    >>> samples = [[97, 32, 98, 32, 99], [100, 32, 101, 32, 195, 133]]
    >>> classes = [[False, False, True, False, False],
    ...            [False, False, True, False, True, True]]
    >>> psamp, pcls, mask = pad(samples, classes)
    >>> (psamp == np.array([[97, 32, 98, 32, 99, 0], [100, 32, 101, 32, 195, 133]])).all()
    True
    >>> (pcls == np.array([[0, 0, 1, 0, 0, 0], [0, 0, 1, 0, 1, 1]])).all()
    True
    >>> (mask == np.array([[1, 1, 1, 1, 1, 0], [1, 1, 1, 1, 1, 1]])).all()
    True
    """
    if len(samples) != len(classes):
        raise ValueError
    maxlen = max(map(len, samples))
    padded_samples = np.zeros(shape=(len(samples), maxlen), dtype=np.int32)
    padded_classes = np.zeros(shape=(len(samples), maxlen), dtype=np.int32)
    masks = np.zeros(shape=(len(samples), maxlen), dtype=bool)
    for i, (sample, cls) in enumerate(zip(samples, classes)):
        if len(sample) != len(cls):
            raise ValueError
        padded_samples[i, :len(sample)] = sample
        padded_classes[i, :len(sample)] = cls
        masks[i, :len(sample)] = True
    return padded_samples, padded_classes, masks


def encode_one_hot(arr, ncls=None):
    """
    :type arr: np.ndarray[int]
    :param arr: an integer 1D/2D-numpy array
    :rtype: np.ndarray[int]
    :return: a one-hot encoded array
    """
    return np.eye(ncls or arr.max()+1, dtype=np.int32)[arr]


def mask_array(arr, mask, value=0):
    """
    Mask `False` positions
    :type arr: np.ndarray
    :param arr:
    :param mask: a boolean 1D array
    :param value:
    :return:
    """
    arr_cp = arr.copy()
    arr_cp[~mask] = value
    return arr_cp


if __name__ == "__main__":
    raise RuntimeError
