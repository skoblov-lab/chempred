
from typing import Text, Tuple, Pattern, List
from functools import reduce
import re

from sciner.intervals import Interval, Intervals

# patterns
numeric = re.compile("[0-9]*\.?[0-9]+")
wordlike = re.compile("[\w]+")
misc = re.compile("[^\s\w]")


def tokenise(patterns: List[Pattern], string: Text, mask=" ") -> Intervals:
    """
    Return intervals matched by `patterns`. The patterns are applied
    in iteration order. Before applying pattern `i+1`, the function replaces
    each region `r` matched by pattern `i` with `mask * len(r)`. This means
    the output might be sensitive to pattern order.
    :param patterns: a list of patterns to search for
    :param string: a unicode string
    :param mask: the masking value
    :return: a list of intervals storing the corresponding string
    """

    def repl(match) -> Text:
        return mask * (match.end() - match.start())

    def reducer(acc: Tuple[List, Text], patt: Pattern):
        spans, s = acc
        spans.extend(m.span for m in patt.finditer(s))
        return spans, patt.sub(repl, s)

    return [Interval(start, stop, string[start:stop]) for start, stop in
            sorted(reduce(reducer, patterns, ([], string))[0])]


if __name__ == "__main__":
    raise RuntimeError
