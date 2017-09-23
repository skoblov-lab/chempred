import re
import unittest
from typing import Sequence

import numpy as np
from hypothesis import given, note
from hypothesis import strategies as st

from sciner import intervals, text, genia


class TestText(unittest.TestCase):

    text_strategy = st.text(
        st.characters(min_codepoint=32, max_codepoint=126), 0, 500, 1000)

    @staticmethod
    def unparse(txt, intervals_: Sequence[intervals.Interval]):
        if not len(intervals_):
            return ""
        codes = np.repeat([ord(" ")], intervals_[-1].stop)
        for iv in intervals_:
            token = intervals.extract(txt, [iv])[0]
            codes[iv.start:iv.stop] = list(map(ord, token))
        return "".join(map(chr, codes))

    @given(text_strategy)
    def test_parse_text(self, txt):
        parsed = text.parser(text.spacy_tokeniser, txt)
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


if __name__ == "__main__":
    unittest.main()
