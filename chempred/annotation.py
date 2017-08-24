from typing import TypeVar, Generic, List, Tuple, Iterable, Sequence, Optional,\
    overload, cast, Any, Container
from numbers import Integral
from math import ceil

from fn.recur import tco


T = TypeVar("T")


class Interval(Container, Generic[T]):

    """
    The intervals are non-inclusive on the right side, like Python's `range`.
    """

    __slots__ = ("_start", "_stop", "_data")

    def __init__(self, start: Integral, stop: Integral, data: Optional[T]=None):
        if not isinstance(start, Integral) or not isinstance(stop, Integral):
            raise TypeError("`start` and `stop` are not Integral")
        self._start = start
        self._stop = stop
        self._data = data

    def __repr__(self) -> str:
        return "{}(start={}, stop={})".format(
            type(self).__name__, self._start, self._stop
        )

    def __contains__(self, item: T) -> bool:
        return self._data == item if self._data is not None else None

    def __lt__(self, other: "Interval") -> bool:
        return self._start < other._start

    @property
    def start(self) -> Integral:
        return self._start

    @property
    def stop(self) -> Integral:
        return self._stop

    @property
    def data(self) -> T:
        return self._data

    def before(self, point: Integral) -> bool:
        return self._stop <= point

    def after(self, point: Integral) -> bool:
        return self._start > point

    def contains(self, point: Integral) -> bool:
        return self._start <= point < self._stop


IntervalT = TypeVar("IntervalT", bound=Interval)


class Annotation(Generic[IntervalT], Sequence):

    # TODO report overlapping regions (raise an error)
    def __init__(self, regions: Iterable[Interval]):
        self._intervals = sorted(regions)

    def __len__(self) -> int:
        return len(self._intervals)

    def __iter__(self):
        return iter(self._intervals)

    @overload
    def __getitem__(self, item: slice) -> List[Interval]:
        pass

    @overload
    def __getitem__(self, item: Integral) -> Optional[Interval]:
        pass

    def __getitem__(self, item):
        """
        :param item:
        :return:
        >>> from itertools import starmap
        >>> ranges = [(2, 3), (5, 6), (7, 9), (11, 15), (19, 30)]
        >>> regions = list(starmap(Interval, ranges))
        >>> annotation = Annotation(regions)
        >>> annotation[0]
        Interval(start=2, stop=3)
        >>> annotation[-1]
        Interval(start=19, stop=30)
        >>> annotation["a"]
        Traceback (most recent call last):
        ...
        TypeError: Can't use objects of type str for indexing/slicing
        """
        try:
            return self._intervals[item]
        except IndexError:
            raise IndexError("{} is out of bounds".format(item))
        except TypeError:
            raise TypeError("Can't use objects of type {} for "
                            "indexing/slicing".format(type(item).__name__))

    def __bool__(self) -> bool:
        return bool(self._intervals)

    def lookup(self, start: Integral, stop: Integral) -> List[Interval]:
        # TODO docs
        """
        :param start:
        :param stop:
        :return:
        >>> from itertools import starmap
        >>> ranges = [(2, 3), (5, 6), (7, 9), (11, 15), (19, 30)]
        >>> regions = list(starmap(Interval, ranges))
        >>> annotation = Annotation(regions)
        >>> [(i.start, i._stop) for i in annotation.lookup(4, 17)] == ranges[1:4]
        True
        >>> annotation.lookup(0, 2)
        []
        """
        first, last = self.borders(start, stop)
        if first is None or last is None or not last - first:
            return []
        return self._intervals[first:last + 1]

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
        >>> annotation = Annotation(regions)
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
        final = len(self) - 1

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
