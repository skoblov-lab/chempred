# TODO add module-level documentation
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

    # not supported by older `typing` versions
    # __slots__ = ("_start", "_stop", "_data")

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

    def __bool__(self):
        return bool(len(self))

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
        if not self:
            raise ValueError("Can't crop a zero-length interval")
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
        """
        Test whether the interval comes before the point
        :param point:
        :return:
        """
        return self._stop <= point

    def after(self, point: Integral) -> bool:
        """
        Test whether the interval comes after the point
        :param point:
        :return:
        """
        return self._start > point

    def contains(self, point: Integral) -> bool:
        """
        Test whether the interval contains the point
        :param point:
        :return:
        """
        return self._start <= point < self._stop


IntervalT = TypeVar("IntervalT", bound=Interval)


class Intervals(Generic[IntervalT], Sequence):

    # TODO report overlapping regions (raise an error)
    # TODO docs
    def __init__(self, regions: Iterable[Interval[T]]):
        """
        :param regions:
        >>> from itertools import starmap
        >>> ranges = [(2, 3), (5, 6), (7, 9), (11, 15), (19, 30)]
        >>> regions = list(starmap(Interval, ranges))
        >>> intervals = Intervals(regions)
        >>> all(Interval(start, stop) in intervals for start, stop in ranges)
        True
        >>> Interval(0, 2) not in intervals
        True
        >>> Interval(7, 8) not in intervals
        True
        """
        self._intervals = sorted(regions)

    def __repr__(self):
        return "{}({})".format(type(self).__name__, repr(self._intervals))

    def __len__(self) -> int:
        return len(self._intervals)

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
        return bool(len(self))

    @property
    def span(self) -> Optional[Interval]:
        """
        Return the spanning interval
        :return:
        """
        return Interval(self[0].start, self[-1].stop) if self else None

    def contains(self, interval: Interval) -> bool:
        """
        Test whether an `interval` lies within the span these `Intervals`
        :param interval:
        :return:
        >>> Intervals([Interval(0, 10)]).contains(Interval(2, 10))
        True
        >>> Intervals([Interval(1, 10)]).contains(Interval(0, 10))
        False
        """
        if not self:
            return False
        span_start, span_stop = self[0].start, self[-1].stop
        return span_start <= interval.start and interval.stop <= span_stop

    def covers(self, span: Interval) -> bool:
        # TODO tests
        """
        Check whether any interval in self covers (at least partially) the span
        :param span: an interval
        """
        start, stop = self._borders(span.start, span.stop)
        if any(border is None for border in [start, stop]) or not stop - start:
            return False
        return True

    def within(self, start: Integral, stop: Integral, partial: bool=False,
               crop: bool=True, cropdata: bool=False) \
            -> "Intervals[Interval[T]]":
        # TODO docs
        # TODO more autotests
        """
        :param start:
        :param stop:
        :param partial: keep partially covered intervals
        :param crop: crop partially covered intervals
        :param cropdata: crop data in partially covered intervals if `crop is True`
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
        >>> not intervals.within(2, 2)
        True
        >>> len(intervals.within(12, 20, True))
        2
        >>> ([(i.start, i.stop) for i in intervals.within(12, 20, True, False)]
        ... == [(11, 15), (19, 30)])
        True
        >>> ([(i.start, i.stop) for i in intervals.within(12, 20, True)] ==
        ...  [(12, 15), (19, 20)])
        True
        >>> ([(i.start, i.stop) for i in intervals.within(12, 40, True)] ==
        ...  [(12, 15), (19, 30)])
        True
        >>> [(i.start, i.stop) for i in intervals.within(-1, 40)] == ranges
        True
        >>> from random import randint
        >>> ranges = [[30, 35]]
        >>> for _ in range(1000):
        ...     start = ranges[-1][1] + randint(0, 100)
        ...     stop = start + randint(1, 100)
        ...     ranges.append([start, stop])
        >>> intervals = Intervals(list(starmap(Interval, ranges)))
        >>> max_ = max(chain.from_iterable(ranges))
        >>> spans = [sorted([randint(0, max_+10000), randint(0, max_+10000)])
        ...          for _ in range(1000)]
        >>> for start, stop in spans:
        ...     _ = intervals.within(start, stop)
        """
        left, right = self._borders(start, stop)
        if left is None or right is None or not right - left:
            return Intervals([])
        intervals = self._intervals[left:right]
        if not crop or not intervals:
            return Intervals(intervals)
        if len(intervals) == 1:
            interval = intervals[0]
            interval_c = interval.crop(max(start, interval.start),
                                       min(stop, interval.stop),
                                       cropdata)
            return (Intervals([]) if not partial and interval != interval_c else
                    Intervals([interval_c]))

        if len(intervals) > 1:
            first, last = intervals[0], intervals[-1]
            fst_c = first.crop(start=max(start, first.start), cropdata=cropdata)
            lst_c = last.crop(stop=min(stop, last.stop), cropdata=cropdata)
            middle = islice(intervals, 1, len(intervals)-1)
            drop_first = not partial and (fst_c != first)
            drop_last = not partial and (lst_c != last)
            intervals_c = chain([fst_c] if not drop_first else [],
                                middle,
                                [lst_c] if not drop_last else [])
            return Intervals(intervals_c)
        assert False

    def _borders(self, start: Integral, stop: Integral) \
            -> Tuple[Optional[int], Optional[int]]:
        # TODO docs
        """
        :param start: inclusive left-border
        :param stop: non-inclusive right-border
        :return: returns indices of the first (inclusive) and the last
        (non-inclusive) interval within [start:stop)
        >>> from itertools import starmap
        >>> ranges = [(2, 3), (5, 6), (7, 9), (11, 15), (19, 30)]
        >>> regions = list(starmap(Interval, ranges))
        >>> annotation = Intervals(regions)
        >>> annotation._borders(4, 17)
        (1, 4)
        >>> annotation._borders(2, 12)
        (0, 4)
        >>> annotation._borders(0, 17)
        (0, 4)
        >>> annotation._borders(0, 25)
        (0, 5)
        >>> annotation._borders(0, 40)
        (0, 5)
        >>> annotation._borders(0, 2)
        (0, 0)
        >>> annotation._borders(0, 0)
        (0, 0)
        >>> annotation._borders(3, 4)
        (1, 1)
        >>> annotation._borders(40, 50)
        (None, None)
        """
        # TODO use an Interval tree

        def left_border(range_, left) -> Optional[int]:
            return next((i for i in range_
                         if self[i].contains(left) or self[i].after(left)), None)

        def right_border(range_, right) -> Optional[int]:
            idx = next((i for i in range_ if self[i].after(right)), len(self)-1)
            return idx if self[idx].after(right) else idx + 1

        first = left_border(range(len(self)), start)
        last = (None if first is None else
                right_border(range(first, len(self)), stop-1))
        return first, last


if __name__ == "__main__":
    raise RuntimeError
