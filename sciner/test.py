from hypothesis import given, note, example, strategies as st

from sciner.genia import text_boundaries


@given(st.lists(st.text()))
def test_text_boundaries(texts: list):
    """
    Test of text_boundaries() function.
    :return:
    """
    boundaries = text_boundaries(texts)
    note(boundaries)

    assert all([boundaries[i][1] == boundaries[i + 1][0] for i in range(len(boundaries) - 1)])
    assert all([boundaries[i][0] <= boundaries[i][1] for i in range(len(boundaries) - 1)])
    if boundaries:
        assert boundaries[0][0] == 0


def run_tests():
    test_text_boundaries()

if __name__ == '__main__':
    run_tests()