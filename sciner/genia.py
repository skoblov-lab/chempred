from typing import Sequence, NamedTuple, Tuple, Iterable, Text, Optional, List, \
    Iterator
from itertools import starmap
from numbers import Integral
from functools import reduce
from xml.etree.ElementTree import Element, parse
import operator as op
import re

from pyrsistent import v, pvector
from fn import F

from sciner.text import AbstractAnnotation, Abstract, ClassMapping, \
    AnnotationError, ClassifiedInterval, TITLE, BODY
from sciner.util import Interval


LevelAnnotation = NamedTuple("Annotation", [("level", int),
                                            ("anno", Sequence[Optional[Text]]),
                                            ("terminal", bool)])
SENTENCE_TAG = "sentence"
ANNO_TAG = "sem"
CODE_PREF = "G#"
ARTICLE = "article"


def flatten_sentence(sentence: Element) \
        -> List[Tuple[Text, Sequence[LevelAnnotation]]]:
    # TODO docs
    def isterminal(element: Element):
        return next(iter(element), None) is None

    def getanno(element: Element):
        return element.get(ANNO_TAG, None)

    stack = [(sentence, iter(sentence), v())]
    texts = [sentence.text]
    annotations = [stack[0][2]]
    while stack:
        node, children, anno = stack[-1]
        child = next(children, None)
        if child is None:
            stack.pop()
            texts.append(node.tail)
            annotations.append(anno[:-1])
            continue
        child_anno = anno.append(
            LevelAnnotation(len(anno), getanno(child), isterminal(child)))
        texts.append(child.text)
        annotations.append(child_anno)
        stack.append((child, iter(child), child_anno))
    return list(zip(texts, annotations))


def text_boundaries(texts: Iterable[Text]) -> List[Tuple[int, int]]:
    # TODO docs
    def aggregate_boundaries(boundaries: pvector, text):
        return (
        boundaries + [(boundaries[-1][1], boundaries[-1][1] + len(text))]
        if boundaries else v((0, len(text))))

    return list(reduce(aggregate_boundaries, texts, v()))


ANNO_PATT = re.compile("G#(\w+)")


def parse_sentences(root: Element, mapping: ClassMapping,
                    default: Integral = None) \
        -> Tuple[Text, List[ClassifiedInterval]]:
    # TODO docs
    def wrap_interval(start: int, stop: int, levels: Sequence[LevelAnnotation]):
        # get the first nonempty annotation bottom to top
        anno = next(filter(bool, (l.anno for l in reversed(levels))), "")
        codes = set(ANNO_PATT.findall(anno))
        if not len(codes) == 1:
            raise AnnotationError(
                "The annotation is either ambiguous or empty: {}".format(codes))
        encoded = mapping.get(codes.pop(), default)
        return Interval(start, stop, encoded) if encoded is not None else None

    sentences = root.findall(SENTENCE_TAG)
    flattened = reduce(op.iadd, map(flatten_sentence, sentences), [])
    texts, annotations = zip(*((txt, anno) for txt, anno in flattened
                               if txt is not None))
    boundaries = text_boundaries(texts)
    intervals = [wrap_interval(start, stop, levels)
                 for (start, stop), levels in zip(boundaries, annotations)
                 if levels and levels[-1].terminal]
    return "".join(texts), list(filter(bool, intervals))


def parse_corpus(path: Text, mapping: ClassMapping, default: Integral = None) \
        -> List[Tuple[Abstract, AbstractAnnotation]]:
    parser = F(parse_sentences, mapping=mapping, default=default)

    def getid(article: Element) -> int:
        raw = article.find("articleinfo").find("bibliomisc").text
        return int(raw.replace("MEDLINE:", ""))

    def accumulate_articles(root) -> Iterator[Tuple[int, Element, Element]]:
        articles = root.findall(ARTICLE)
        ids = map(getid, articles)
        title_roots = [article.find("title") for article in articles]
        body_roots = [article.find("abstract") for article in articles]
        return zip(ids, title_roots, body_roots)

    def parse_article(id_: int, title_root: Element, body_root: Element) \
            -> Tuple[Abstract, AbstractAnnotation]:
        title_text, title_anno = parser(title_root)
        body_text, body_anno = parser(body_root)
        abstract = Abstract(id_, title_text, body_text)
        annotation = AbstractAnnotation(id_, title_anno, body_anno)
        return abstract, annotation

    corpus = parse(path)
    articles = accumulate_articles(corpus)
    return list(starmap(parse_article, articles))


if __name__ == "__main__":
    raise RuntimeError
