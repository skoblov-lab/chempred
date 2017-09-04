"""


"""

from typing import Sequence, Container, Generic, TypeVar, Iterable

from intervaltree import Interval as _Interval, IntervalTree as _IntervalTree

T = TypeVar("T")


class Interval(_Interval, Container, Generic[T]):
    def __contains__(self, item: T) -> bool:
        return False if self.data is None or item is None else self.data == item

    def __len__(self):
        return self.length()

    def __bool__(self):
        return bool(self.length())

    def __repr__(self):
        return "{}(start={}, stop={}, data={})".format(type(self).__name__,
                                                       self.start,
                                                       self.stop,
                                                       self.data)

    # These two properties were added for compatibility with Python's
    # `range` naming convention
    @property
    def start(self) -> int:
        return self.begin

    @property
    def stop(self) -> int:
        return self.end

    def crop(self, begin=None, end=None):
        begin = self.begin if begin is None else begin
        end = self.end - 1 if end is None else end
        if end < begin:
            raise ValueError("Can't crop an interval to negative length")
        if not self.contains_point(begin):
            raise ValueError("New `begin` lies outside the interval")
        if not self.contains_point(end-1):
            raise ValueError("New `end` lies outdise the interval")
        return type(self)(begin, end, self.data)


IntervalT = TypeVar("IntervalT", bound=Interval)


class Intervals(Sequence, Generic[IntervalT]):

    def __init__(self, intervals: Iterable[Interval[T]]):
        self._intervals = sorted(intervals)
        self._tree = _IntervalTree(self._intervals)

    def __len__(self) -> int:
        return len(self._intervals)

    def __bool__(self):
        return bool(len(self))

    def __iter__(self):
        return iter(self._intervals)

    def __getitem__(self, i: int) -> Interval[T]:
        return self._intervals[i]

    def __repr__(self):
        return "{}({})".format(type(self).__name__, repr(self._intervals))

    @property
    def span(self) -> Interval:
        return Interval(self[0].start, self[-1].stop)

    def within(self, start, stop, partial=False, crop=True) \
            -> "Intervals[Interval[T]]":
        intervals = list(self._tree.search(start, stop, not partial))
        if partial and crop:
            intervals[0] = intervals[0].crop(begin=max(start,
                                                       intervals[0].begin))
            intervals[-1] = intervals[-1].crop(end=min(stop,
                                                       intervals[-1].end))
        return type(self)(intervals)


if __name__ == "__main__":
    raise RuntimeError
