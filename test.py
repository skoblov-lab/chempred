import re
import unittest
from typing import Sequence, Iterable, cast

import numpy as np
from hypothesis import given, note
from hypothesis import strategies as st

from sciner import intervals, text, genia, sampling


# strategies

texts = st.text(st.characters(min_codepoint=32, max_codepoint=255), 0, 500, 1000)


@st.composite
def annotations_and_samples(draw, ncls, l, nintervals):
    ncls = draw(ncls)
    length = draw(l)
    n = draw(nintervals) + 1

    anno = np.random.choice(ncls, length)
    split_points = sorted(np.random.choice(length, n, False))
    ivs = [range(arr[0], arr[-1] + 1) for arr in
           np.split(np.arange(length), split_points)[1:-1]]

    return ncls, anno, ivs


# test cases

class TestText(unittest.TestCase):

    @staticmethod
    def unparse(txt, intervals_: Sequence[intervals.Interval]):
        if not len(intervals_):
            return ""
        codes = np.repeat([ord(" ")], intervals_[-1].stop)
        for iv in intervals_:
            token = intervals.extract(txt, [iv])[0]
            codes[iv.start:iv.stop] = list(map(ord, token))
        return "".join(map(chr, codes))

    @given(texts)
    def test_parse_text(self, txt):
        parsed = text.tointervals(text.spacy_tokeniser, txt)
        mod_text = re.sub("\s", " ", txt)
        self.assertEqual(self.unparse(txt, parsed), mod_text.rstrip())


class TestGenia(unittest.TestCase):
    @given(st.lists(st.text()))
    def test_text_boundaries(self, texts: list):
        """
        Test of text_boundaries() function.
        :return:
        """
        boundaries = genia.text_boundaries(texts)
        note(boundaries)

        self.assertTrue(all([boundaries[i][1] == boundaries[i + 1][0] for i in
                             range(len(boundaries) - 1)]))
        self.assertTrue(all([boundaries[i][0] <= boundaries[i][1] for i in
                            range(len(boundaries) - 1)]))
        if boundaries:
            self.assertTrue(boundaries[0][0] == 0)


class TestSampling(unittest.TestCase):

    @given(annotations_and_samples(st.integers(1, 10),
                                   st.integers(100, 500),
                                   st.integers(1, 100)))
    def test_annotate_sample(self, annotation_and_intervals):
        ncls, anno, sample = annotation_and_intervals
        sample_anno = sampling.annotate_sample(anno, ncls, sample)
        sample_anno_cls = [set(s_anno.nonzero()[-1])
                           for s_anno in cast(Iterable[np.ndarray], sample_anno)]
        self.assertSequenceEqual([set(anno[iv.start:iv.stop]) for iv in sample],
                                 sample_anno_cls)


if __name__ == "__main__":
    unittest.main()
