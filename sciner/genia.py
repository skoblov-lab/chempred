from xml.etree import ElementTree
from numbers import Integral
from typing import List, Tuple, Text, NamedTuple

import numpy as np
from fn import F

from sciner.chemdner import ClassMapping, AbstractAnnotation, Abstract, \
    TITLE, BODY
from sciner.util import Interval

Record = NamedTuple("Record", [("source", Text),
                               ("start", int),
                               ("end", int),
                               ("text", Text),
                               ("cls", Text)])


# TODO tests
def read_sentence(sent, src: str, shift: int = 0) \
        -> Tuple[str, List[Record]]:
    # TODO read test data from files (i.e. make a testdata directory)
    """
    Strip tags from sentence and add info to list of Annotations
    :param sent: sentence with tags to be parsed
    :type sent: xml.etree.ElementTree.Element
    :param src: text source type (i.e. title or body)
    :param shift: length of preceding text; needed to correctly tell start and
    end position in read_abstract
    :return: Tuple[Text, List[Annotaion]]

    >>> sentence = ElementTree.fromstring('<sentence><cons lex="IL-2_gene_expression" sem="G#other_name"><cons lex="IL-2_gene" sem="G#DNA_domain_or_region">IL-2 gene</cons> expression</cons> and <cons lex="NF-kappa_B_activation" sem="G#other_name"><cons lex="NF-kappa_B" sem="G#protein_molecule">NF-kappa B</cons> activation</cons> through <cons lex="CD28" sem="G#protein_molecule">CD28</cons> requires reactive oxygen production by <cons lex="5-lipoxygenase" sem="G#protein_molecule">5-lipoxygenase</cons>.</sentence>')
    >>> s = read_sentence(sentence, 'Title')
    >>> text = s[0]
    >>> print([text[ann.start:ann.end] for ann in s[1]])
    ['IL-2 gene', 'IL-2 gene expression', 'NF-kappa B', 'NF-kappa B activation', 'CD28', '5-lipoxygenase']
    >>> ann = {text[ann.start:ann.end]: ann.cls for ann in s[1]}
    >>> manual_ann = {'IL-2 gene expression': 'G#other_name', 'IL-2 gene': 'G#DNA_domain_or_region', 'NF-kappa B activation': 'G#other_name', 'NF-kappa B': 'G#protein_molecule', 'CD28': 'G#protein_molecule', '5-lipoxygenase': 'G#protein_molecule'}
    >>> ann == manual_ann
    True
    """

    def check_coordinated_clause(element) -> bool:
        """
        Identifies coordinated clauses in 'sem' attributes
        (e.g. (AS_WELL_AS G#other_name G#other_name))
        :param element: xml.etree.ElementTree.Element
        :return: whether or not the element includes a coordinated clauses
        """
        return len(element.get('sem').split()) > 1 if element.get('sem') else False

    def annotate_elements(element, src: str,
                          start: int = 0, cls: str = None) -> List[Record]:
        """
        Recursively annotate embedded terms
        >>> el = ElementTree.fromstring('<el>Demping<cons sem="a"><cons sem="b">B</cons>and<cons sem="c">C</cons></cons></el>')
        >>> x = list(annotate_elements(el, "A"))
        >>> text = "".join(el.itertext())
        >>> for i in x:
        ...     print(text[i.start:i.end], i.cls)
        B b
        C c
        BandC a
        DempingBandC None
        """
        # TODO embedded coordinated clauses
        start = start
        text = "".join(element.itertext())

        # Cycle looks unnecessary, but can be modified for multilevel annotation in future
        if cls:
            if element.get('sem'):
                cls = element.get('sem')
                # raise ValueError("Annotated class inside coordinated clause")
        else:
            cls = element.get('sem')

        end = start + len(text)
        new_start = start + len(element.text) if element.text else start
        if check_coordinated_clause(element):
            cls = element.get('sem').split()[1]

            for el in list(element):
                for ann in list(annotate_elements(el, src,
                                                  new_start, cls)):
                    yield ann
                    new_start += len(ann.text)

                    if el.tail:
                        new_start += len(el.tail)

        else:
            for el in list(element):
                for ann in list(annotate_elements(el, src,
                                                  new_start)):

                    yield ann
                    new_start += len(ann.text)

                    if el.tail:
                        new_start += len(el.tail)

            if not list(element):
                yield Record(src, start, end, text, cls)

    l = []
    text = "".join(sent.itertext())
    start = 0 + shift
    if sent.text:
        start += len(sent.text)
    for cons in list(sent):
        for annotation in list(annotate_elements(cons, src,
                                                 start)):
            l.append(annotation)
        start += len("".join(cons.itertext()))
        if cons.tail:
            start += len(cons.tail)
    return text, l


# TODO tests
def read_abstract(abstract: ElementTree.Element, src: str) -> Tuple[str, List[Record]]:
    """
    Strip tags from abstract and add info to list of Annotations
    :param abstract: abstract consisting of 'sentence' xml elements
    :type abstract: xml.etree.ElementTree.Element
    :param src: text source type (i.e. title or body)
    :return: Tuple[str, List[Record]]
    >>> abstract = ElementTree.fromstring('<abstract><sentence>Activation of the <cons lex="CD28_surface_receptor" sem="G#protein_family_or_group"><cons lex="CD28" sem="G#protein_molecule">CD28</cons> surface receptor</cons> provides a major costimulatory signal for <cons lex="T_cell_activation" sem="G#other_name">T cell activation</cons> resulting in enhanced production of <cons lex="interleukin-2" sem="G#protein_molecule">interleukin-2</cons> (<cons lex="IL-2" sem="G#protein_molecule">IL-2</cons>) and <cons lex="cell_proliferation" sem="G#other_name">cell proliferation</cons>.</sentence><sentence>In <cons lex="primary_T_lymphocyte" sem="G#cell_type">primary T lymphocytes</cons> we show that <cons lex="CD28" sem="G#protein_molecule">CD28</cons> ligation leads to the rapid intracellular formation of <cons lex="reactive_oxygen_intermediate" sem="G#inorganic">reactive oxygen intermediates</cons> (<cons lex="ROI" sem="G#inorganic">ROIs</cons>) which are required for <cons lex="CD28-mediated_activation" sem="G#other_name"><cons lex="CD28" sem="G#protein_molecule">CD28</cons>-mediated activation</cons> of the <cons lex="NF-kappa_B" sem="G#protein_molecule">NF-kappa B</cons>/<cons lex="CD28-responsive_complex" sem="G#protein_complex"><cons lex="CD28" sem="G#protein_molecule">CD28</cons>-responsive complex</cons> and <cons lex="IL-2_expression" sem="G#other_name"><cons lex="IL-2" sem="G#protein_molecule">IL-2</cons> expression</cons>.</sentence><sentence>Delineation of the <cons lex="CD28_signaling_cascade" sem="G#other_name"><cons lex="CD28" sem="G#protein_molecule">CD28</cons> signaling cascade</cons> was found to involve <cons lex="protein_tyrosine_kinase_activity" sem="G#other_name"><cons lex="protein_tyrosine_kinase" sem="G#protein_family_or_group">protein tyrosine kinase</cons> activity</cons>, followed by the activation of <cons lex="phospholipase_A2" sem="G#protein_molecule">phospholipase A2</cons> and <cons lex="5-lipoxygenase" sem="G#protein_molecule">5-lipoxygenase</cons>.</sentence><sentence>Our data suggest that <cons lex="lipoxygenase_metabolite" sem="G#protein_family_or_group"><cons lex="lipoxygenase" sem="G#protein_molecule">lipoxygenase</cons> metabolites</cons> activate <cons lex="ROI_formation" sem="G#other_name"><cons lex="ROI" sem="G#inorganic">ROI</cons> formation</cons> which then induce <cons lex="IL-2" sem="G#protein_molecule">IL-2</cons> expression via <cons lex="NF-kappa_B_activation" sem="G#other_name"><cons lex="NF-kappa_B" sem="G#protein_molecule">NF-kappa B</cons> activation</cons>.</sentence><sentence>These findings should be useful for <cons lex="therapeutic_strategies" sem="G#other_name">therapeutic strategies</cons> and the development of <cons lex="immunosuppressants" sem="G#other_name">immunosuppressants</cons> targeting the <cons lex="CD28_costimulatory_pathway" sem="G#other_name"><cons lex="CD28" sem="G#protein_molecule">CD28</cons> costimulatory pathway</cons>.</sentence></abstract>')
    >>> abst = read_abstract(abstract, 'A')
    >>> text = abst[0]
    >>> print([text[ann.start:ann.end] for ann in abst[1]])
    ['CD28', 'CD28 surface receptor', 'T cell activation', 'interleukin-2', 'IL-2', 'cell proliferation', 'primary T lymphocytes', 'CD28', 'reactive oxygen intermediates', 'ROIs', 'CD28', 'CD28-mediated activation', 'NF-kappa B', 'CD28', 'CD28-responsive complex', 'IL-2', 'IL-2 expression', 'CD28', 'CD28 signaling cascade', 'protein tyrosine kinase', 'protein tyrosine kinase activity', 'phospholipase A2', '5-lipoxygenase', 'lipoxygenase', 'lipoxygenase metabolites', 'ROI', 'ROI formation', 'IL-2', 'NF-kappa B', 'NF-kappa B activation', 'therapeutic strategies', 'immunosuppressants', 'CD28', 'CD28 costimulatory pathway']
    """
    sh = 0
    sentences = []
    records = []
    for sent in abstract.findall('sentence'):
        sent_text, sent_ann = read_sentence(sent, src, sh)
        sentences.append(sent_text)
        records += sent_ann
        sh += len(sent_text) + 1
    text = " ".join(sentences)
    return text, records


# TODO tests
def read_articles(path: str, mapping: ClassMapping, default: Integral=0) \
        -> List[Tuple[Abstract, AbstractAnnotation]]:
    """
    Parse the entire GENIA corpus
    :param path: path to GENIA xml file
    :param mapping: a class mapping
    :param default: default class integer value for out-of-mapping classes
    :return: Iterator[(parsed abstract, parsed annotation)]
    """

    def wrap_interval(rec: Record) -> Interval:
        value = mapping.get(rec.cls, default)
        return None if value is None else Interval(rec.start, rec.end, value)

    wrap_records = (F(map, wrap_interval)
                    >> (filter, bool)
                    >> list
                    >> np.array)

    tree = ElementTree.parse(path)
    root = tree.getroot()
    pairs = []
    for article in root.findall('article'):
        id_raw = article.find('articleinfo').find('bibliomisc').text
        id_ = int(id_raw.replace('MEDLINE:', ''))
        title_text, title_rec = read_abstract(article.find('title'), TITLE)
        title_ann = wrap_records(title_rec)
        abst_text, abst_rec = read_abstract(article.find('abstract'), BODY)
        abst_ann = wrap_records(abst_rec)
        pairs.append(((id_, title_text, abst_text), (id_, title_ann, abst_ann)))

    return pairs


if __name__ == "__main__":
    raise RuntimeError
