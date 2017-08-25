from typing import TypeVar, Generic, Hashable, Tuple, Iterable, Sequence, \
    Sized, Container, Optional, overload, cast
from itertools import chain, islice
from numbers import Integral
from math import ceil

from fn.recur import tco


T = TypeVar("T")


class Interval(Hashable, Container, Sized, Generic[T]):

    """
    The intervals are non-inclusive on the right side, like Python's `range`.
    """

    # __slots__ = ("_start", "_stop", "_data")
    # note: a bug in `typing` makes it not fully compatible with __slots__;
    #       because of this you can't bind a real type to the container using
    #       stock `typing` from Python 3.5. This was patched in typing 3.6.2

    def __init__(self, start: Integral, stop: Integral, data: Optional[T]=None):
        if not isinstance(start, Integral) or not isinstance(stop, Integral):
            raise TypeError("`start` and `stop` are not Integral")
        if stop < start:
            raise ValueError("Negative-length intervals are not allowed")
        self._start = start
        self._stop = stop
        self._data = data

    def __repr__(self) -> str:
        return "{}(start={}, stop={})".format(
            type(self).__name__, self._start, self._stop
        )

    def __contains__(self, item: T) -> bool:
        return (self._data == item if self._data is not None and item is not None
                else False)

    def __lt__(self, other: "Interval") -> bool:
        return self._start < other._start

    def __len__(self) -> int:
        return self._stop - self._start

    def __eq__(self, other: "Interval[T]") -> bool:
        """
        It doesn't compare data
        :param other:
        :return:
        """
        return self.start == other.start and self.stop == other.stop

    def __hash__(self):
        return hash((self._start, self._stop))

    @property
    def start(self) -> Integral:
        return self._start

    @property
    def stop(self) -> Integral:
        return self._stop

    @property
    def data(self) -> T:
        return self._data

    def crop(self, start: Integral=None, stop: Integral=None,
             cropdata: bool=False) -> "Interval[T]":
        """
        :param start:
        :param stop:
        :param cropdata: crop content, too
        :return:
        >>> s = "a" * 10
        >>> i = Interval(0, 10, data=s)
        >>> i.crop(2, 10, False)
        Interval(start=2, stop=10)
        >>> i.crop(2, 10, False).data == s
        True
        >>> i.crop(2, 10, True).data == s[2:10]
        True
        >>> i.crop(-1, 10)
        Traceback (most recent call last):
        ...
        ValueError: New start lies outside interval's borders
        >>> i.crop(0, 12)
        Traceback (most recent call last):
        ...
        ValueError: New stop lies outside interval's borders
        >>> i.crop(1, 1, True).data == ""
        True
        """
        if start is not None and not self.contains(start):
            raise ValueError("New start lies outside interval's borders")
        if stop is not None and not self.contains(stop-1):
            raise ValueError("New stop lies outside interval's borders")

        new_start = self._start if start is None else start
        new_stop = self._stop if stop is None else stop

        # crop data if necessary
        try:
            offset = new_start - self._start
            l = new_stop - new_start
            new_data = self._data[offset:offset+l] if cropdata else self._data
            return Interval(new_start, new_stop, new_data)
        except (TypeError, ValueError):
            raise TypeError("Failed to slice the data")

    def before(self, point: Integral) -> bool:
        return self._stop <= point

    def after(self, point: Integral) -> bool:
        return self._start > point

    def contains(self, point: Integral) -> bool:
        return self._start <= point < self._stop


IntervalT = TypeVar("IntervalT", bound=Interval)


class Intervals(Generic[IntervalT], Sequence):

    # TODO report overlapping regions (raise an error)
    def __init__(self, regions: Iterable[Interval[T]]):
        self._intervals = sorted(regions)

    def __repr__(self):
        return "{}({})".format(type(self).__name__, repr(self._intervals))

    def __len__(self) -> int:
        return self[-1].stop - self[0].start

    def __iter__(self):
        return iter(self._intervals)

    @overload
    def __getitem__(self, item: slice) -> "Intervals[Interval[T]]":
        pass

    @overload
    def __getitem__(self, item: int) -> Optional[Interval[T]]:
        pass

    def __eq__(self, other: "Intervals[Interval[T]]") -> bool:
        return self._intervals == other._intervals

    def __getitem__(self, item):
        # Todo more tests
        """
        :param item:
        :return:
        >>> from itertools import starmap
        >>> ranges = [(2, 3), (5, 6), (7, 9), (11, 15), (19, 30)]
        >>> regions = list(starmap(Interval, ranges))
        >>> intervals = Intervals(regions)
        >>> intervals[0]
        Interval(start=2, stop=3)
        >>> intervals[-1]
        Interval(start=19, stop=30)
        >>> intervals[:] == intervals
        True
        >>> intervals["a"]
        Traceback (most recent call last):
        ...
        TypeError: Can't use objects of type str for indexing/slicing
        """
        try:
            return (self._intervals[item] if isinstance(item, int) else
                    Intervals(self._intervals[item]))
        except IndexError:
            raise IndexError("{} is out of bounds".format(item))
        except TypeError:
            raise TypeError("Can't use objects of type {} for "
                            "indexing/slicing".format(type(item).__name__))

    def __bool__(self) -> bool:
        return bool(self.size)

    @property
    def size(self) -> int:
        return len(self._intervals)

    def within(self, start: Integral, stop: Integral, dropcropped=True, crop=True,
               cropdata=False) -> "Intervals[Interval[T]]":
        # TODO docs
        # TODO more tests
        """
        :param start:
        :param stop:
        :return:
        >>> from itertools import starmap
        >>> ranges = [(2, 3), (5, 6), (7, 9), (11, 15), (19, 30)]
        >>> regions = list(starmap(Interval, ranges))
        >>> intervals = Intervals(regions)
        >>> [(i.start, i.stop) for i in intervals.within(4, 17)] == ranges[1:4]
        True
        >>> not intervals.within(0, 2)
        True
        >>> not intervals.within(12, 20)
        True
        >>> ([(i.start, i.stop) for i in intervals.within(12, 20, False, False)]
        ... == [(11, 15), (19, 30)])
        True
        >>> ([(i.start, i.stop) for i in intervals.within(12, 20, False)] ==
        ...  [(12, 15), (19, 20)])
        True
        >>> ([(i.start, i.stop) for i in intervals.within(12, 40, False)] ==
        ...  [(12, 15), (19, 30)])
        True
        >>> [(i.start, i.stop) for i in intervals.within(-1, 40)] == ranges
        True
        """
        first_idx, last_idx = self.borders(start, stop)
        if first_idx is None or last_idx is None or not last_idx - first_idx:
            return Intervals([])
        intervals = self._intervals[first_idx:last_idx+1]
        if not crop or not intervals:
            return Intervals(intervals)
        if len(intervals) == 1:
            return Intervals([intervals[0].crop(start, stop, cropdata)])
        if len(intervals) > 1:
            first, last = intervals[0], intervals[-1]
            fst_c = first.crop(start=max(start, first.start), cropdata=cropdata)
            lst_c = last.crop(stop=min(stop, last.stop), cropdata=cropdata)
            middle = islice(intervals, 1, len(intervals)-1)
            drop_first = dropcropped and (fst_c != first)
            drop_last = dropcropped and (lst_c != last)
            intervals_c = chain([fst_c] if not drop_first else [],
                                middle,
                                [lst_c] if not drop_last else [])
            return Intervals(intervals_c)
        assert False

    def borders(self, start: Integral, stop: Integral) \
            -> Tuple[Optional[int], Optional[int]]:
        # TODO docs
        """
        :param start: inclusive left-border
        :param stop: non-inclusive right-border
        :return: both borders are inclusive
        >>> from itertools import starmap
        >>> ranges = [(2, 3), (5, 6), (7, 9), (11, 15), (19, 30)]
        >>> regions = list(starmap(Interval, ranges))
        >>> annotation = Intervals(regions)
        >>> annotation.borders(4, 17)
        (1, 3)
        >>> annotation.borders(2, 12)
        (0, 3)
        >>> annotation.borders(0, 17)
        (0, 3)
        >>> annotation.borders(0, 25)
        (0, 4)
        >>> annotation.borders(0, 40)
        (0, 4)
        >>> annotation.borders(0, 2)
        (0, 0)
        >>> annotation.borders(3, 4)
        (1, 1)
        >>> annotation.borders(40, 50)
        (None, None)
        """
        final = self.size - 1

        @tco
        def left_border(l, idx):
            if self[idx].contains(l) or (not idx and self[idx].after(l)):
                return False, idx
            if idx >= final and self[idx].before(l):
                return False, None
            if self[idx].before(l) and self[idx+1].after(l):
                return False, idx + 1
            return True, (l, (idx // 2 if self[idx].after(l) else
                              ceil((idx + final) / 2)))

        @tco
        def right_border(r, idx):
            if self[idx].contains(r) or (idx == final and self[idx].before(r)):
                return False, idx
            if not idx and self[idx].after(r):
                return False, None
            if self[idx].before(r) and self[idx+1].after(r):
                return False, idx
            return True, (r, (idx // 2 if self[idx].after(r) else
                              ceil((idx + final) / 2)))

        first = None if not self else left_border(start, final // 2)
        last = None if first is None else right_border(stop-1, cast(int, first))
        return first, last or first


if __name__ == "__main__":
    raise RuntimeError
