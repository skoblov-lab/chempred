from typing import overload, List, Tuple, Sequence, Optional, NamedTuple
from math import ceil

from fn.recur import tco


Interval = NamedTuple("Interval", [("start", int), ("end", int)])


class ClassifiedRegion:
    """
    The regions are non-inclusive on the right side, like Python's `range`.
    """
    __slots__ = ("start", "end", "cls")

    def __init__(self, start: int, end: int, cls: int):
        self.start = start
        self.end = end
        self.cls = cls

    def __repr__(self) -> str:
        return "AnnotatedRegion(start={}, end={}, cls={})".format(
            self.start, self.end, self.cls
        )

    def __contains__(self, point: int) -> bool:
        return self.contains(point)

    def __lt__(self, other: "ClassifiedRegion"):
        return self.start < other.start

    def before(self, point: int) -> bool:
        return self.end <= point

    def after(self, point: int) -> bool:
        return self.start > point

    def contains(self, point: int) -> bool:
        return self.start <= point < self.end


class Annotation:
    # TODO report overlapping regions (raise an error)
    def __init__(self, regions: Sequence[ClassifiedRegion], default=0):
        self._regions = sorted(regions)
        self.default = default

    @overload
    def __getitem__(self, item: slice) -> Sequence[int]:
        pass

    @overload
    def __getitem__(self, item: int) -> Optional[ClassifiedRegion]:
        pass

    def __getitem__(self, item):
        if isinstance(item, slice):
            pass
        elif isinstance(item, int):
            return self._regions[item] if item < self.size else None
        raise ValueError

    def __bool__(self):
        return bool(self._regions)

    def __len__(self):
        raise NotImplemented

    @property
    def regions(self) -> List[ClassifiedRegion]:
        return list(self._regions)

    @property
    def size(self) -> int:
        return len(self._regions)

    def borders(self, interval: slice) \
            -> Tuple[Optional[int], Optional[int]]:
        # TODO docs
        """
        !!!Both borders are inclusive!!!
        :param interval:
        :return:
        >>> regions = [ClassifiedRegion(s, e, 1) for s, e in
        ...            [(2, 3), (5, 6), (7, 9), (11, 15), (19, 30)]]
        >>> annotations = Annotation(regions)
        >>> annotations.borders(slice(4, 17))
        (1, 3)
        >>> annotations.borders(slice(2, 12))
        (0, 3)
        >>> annotations.borders(slice(0, 17))
        (0, 3)
        >>> annotations.borders(slice(0, 25))
        (0, 4)
        >>> annotations.borders(slice(0, 40))
        (0, 4)
        >>> annotations.borders(slice(0, 2))
        (0, 0)
        >>> annotations.borders(slice(3, 4))
        (1, 1)
        >>> annotations.borders(slice(40, 50))
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

        first = None if not self else left_border(interval.start, final // 2)
        last = None if first is None else right_border(interval.stop-1, first)
        return first, last or first
