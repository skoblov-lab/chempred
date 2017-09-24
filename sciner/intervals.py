import sys
from typing import TypeVar, Container, Generic, Optional, Sequence, Iterable, \
    List

_slots_supported = (sys.version_info >= (3, 6, 2) or
                    (3, 5, 3) <= sys.version_info < (3, 6))
T = TypeVar("T")


class Interval(Container, Generic[T]):

    if _slots_supported:
        __slots__ = ("start", "stop", "data")

    def __init__(self, start: int, stop: int, data: Optional[T]=None):
        self.start = start
        self.stop = stop
        self.data = data

    def __contains__(self, item: T) -> bool:
        return False if self.data is None or item is None else self.data == item

    def __eq__(self, other: "Interval"):
        return (self.start, self.stop, self.data) == (other.start, other.stop, other.data)

    def __hash__(self):
        return hash((self.start, self.stop, self.data))

    def __len__(self):
        return self.stop - self.start

    def __bool__(self):
        return bool(len(self))

    def __repr__(self):
        return "{}(start={}, stop={}, data={})".format(type(self).__name__,
                                                       self.start,
                                                       self.stop,
                                                       self.data)

    def reload(self, value: T):
        return type(self)(self.start, self.stop, value)


def extract(sequence: Sequence[T], intervals: Iterable[Interval]) \
        -> List[Sequence[T]]:
    return [sequence[iv.start:iv.stop] for iv in intervals]


def length(sample: Sequence[Interval]) -> int:
    # TODO docs
    return 0 if not len(sample) else sample[-1].stop - sample[0].start


def span(ivs: Sequence[Interval]) -> Optional[Interval]:
    return Interval(ivs[0].start, ivs[-1].stop) if len(ivs) else None


if __name__ == "__main__":
    raise ValueError
