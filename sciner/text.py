from numbers import Integral
from typing import Sequence, NamedTuple, Text, Iterable, Iterator, Tuple, List, \
    Mapping

from sciner.util import Interval

OTHER = "OTHER"
TITLE = "T"
BODY = "A"
ClassMapping = Mapping[Text, Integral]
ClassifiedInterval = Interval[Integral]
Annotation = Sequence[ClassifiedInterval]
AbstractAnnotation = NamedTuple("AbstractAnnotation", [("id", int),
                                                       ("title", Annotation),
                                                       ("body", Annotation)])
Abstract = NamedTuple("Abstract",
                      [("id", int), ("title", Text), ("body", Text)])


class AnnotationError(ValueError):
    pass


def align_abstracts_and_annotations(abstracts: Iterable[Abstract],
                                    annotations: Iterable[AbstractAnnotation]) \
        -> Iterator[Tuple[Abstract, AbstractAnnotation]]:
    # TODO tests
    """
    Align abstracts and annotations (i.e. match abstract ids)
    :param abstracts: parsed abstracts (e.g. produces by `read_abstracts`)
    :param annotations: parsed annotations (e.g. produces by `read_annotations`)
    :return: Iterator[(parsed abstract, parsed annotation)]
    """
    def empty(id_: int) -> AbstractAnnotation:
        return AbstractAnnotation(id_, [], [])

    anno_mapping = {anno.id: anno for anno in annotations}
    return ((abstract, anno_mapping.get(abstract.id, empty(abstract.id)))
            for abstract in abstracts)


def flatten_aligned_pair(pair: Tuple[Abstract, AbstractAnnotation]) \
        -> List[Tuple[int, Text, Text, Sequence[Interval]]]:
    # TODO tests
    """
    :return: list[(abstract id, source, text, annotation)]
    """
    (abstract_id, title, body), (anno_id, title_anno, body_anno) = pair
    if abstract_id != anno_id:
        raise AnnotationError("Abstract ids do not match")
    return [(abstract_id, TITLE, title, title_anno),
            (abstract_id, BODY, body, body_anno)]


def parse_mapping(classmaps: Iterable[str]) -> ClassMapping:
    """
    :param classmaps:
    :return:
    >>> classmaps = ["a:1", "b:1", "c:2"]
    >>> parse_mapping(classmaps) == dict(a=1, b=1, c=2)
    True
    """
    try:
        return {cls: int(val)
                for cls, val in [classmap.split(":") for classmap in classmaps]}
    except ValueError as err:
        raise AnnotationError("Badly formatted mapping: {}".format(err))


if __name__ == "__main__":
    raise RuntimeError
