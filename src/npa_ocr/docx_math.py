"""Формулы и индексы в DOCX.

LaTeX переводится в редактируемую формулу Word по цепочке LaTeX → MathML
(latex2mathml) → OMML (mathml2omml). Разметка ``<sub>``/``<sup>`` становится
подстрочным и надстрочным шрифтом.

Формула, которую не удалось перевести, пишется в документ исходным LaTeX и
выделяется жёлтым: текст не теряется, а место видно при проверке.
"""

from __future__ import annotations

import re

from docx.enum.text import WD_ALIGN_PARAGRAPH, WD_COLOR_INDEX
from lxml import etree

_MATH_NS = "http://schemas.openxmlformats.org/officeDocument/2006/math"
_W_NS = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"
# слово DeepSeek по частям: \(LaTeX\), <sub>…</sub>, <sup>…</sup>, обычный текст
_PART_RE = re.compile(r"\\\((.*?)\\\)|<sub>(.*?)</sub>|<sup>(.*?)</sup>|([^\\<]+|[\\<])", re.S)


def latex_to_omml(latex: str) -> etree._Element | None:
    """Переводит LaTeX в элемент ``m:oMath``.

    Args:
        latex: Формула LaTeX без ограничителей ``\\[ \\]`` и ``\\( \\)``.

    Returns:
        Элемент ``m:oMath`` или None, если формулу не удалось перевести.
    """
    import latex2mathml.converter
    import mathml2omml

    try:
        omml = mathml2omml.convert(latex2mathml.converter.convert(latex))
        el = etree.fromstring(f'<root xmlns:m="{_MATH_NS}" xmlns:w="{_W_NS}">{omml}</root>')
    except Exception:  # noqa: BLE001 — любой сбой разбора: формула идёт текстом, документ не падает
        return None
    math = el.find(f"{{{_MATH_NS}}}oMath")
    return math


def add_formula(doc, latex: str) -> None:
    """Добавляет формулу в документ отдельным абзацем по центру.

    Args:
        doc: Документ python-docx.
        latex: Формула LaTeX без ограничителей.
    """
    para = doc.add_paragraph()
    para.alignment = WD_ALIGN_PARAGRAPH.CENTER
    math = latex_to_omml(latex)
    if math is None:
        run = para.add_run(latex)
        run.font.highlight_color = WD_COLOR_INDEX.YELLOW
        return
    wrap = etree.SubElement(para._p, f"{{{_MATH_NS}}}oMathPara")
    wrap.append(math)


def add_rich(para, markup: str, low: bool) -> None:
    """Добавляет в абзац слово с разметкой.

    Обычный текст и индексы пишутся прогонами (индексы — подстрочным и
    надстрочным шрифтом), ``\\(LaTeX\\)`` — формулой Word в строке.

    Args:
        para: Абзац python-docx.
        markup: Слово с разметкой ``<sub>``, ``<sup>`` и ``\\(…\\)``.
        low: Выделить слово жёлтым как неуверенно распознанное.
    """
    for m in _PART_RE.finditer(markup):
        latex, sub, sup, text = m.groups()
        if latex is not None:
            math = latex_to_omml(latex)
            if math is not None:
                para._p.append(math)
                continue
            text = latex
        run = para.add_run(sub if sub is not None else sup if sup is not None else text)
        if sub is not None:
            run.font.subscript = True
        if sup is not None:
            run.font.superscript = True
        if low:
            run.font.highlight_color = WD_COLOR_INDEX.YELLOW
