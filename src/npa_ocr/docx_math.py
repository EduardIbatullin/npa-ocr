"""Формулы и индексы в DOCX: LaTeX DeepSeek-OCR → MathML (latex2mathml) → OMML (mathml2omml) —
настоящая формула Word, редактируемая; <sub>/<sup> — подстрочный/надстрочный шрифт.

Формула, которую не удалось перевести, идёт в DOCX как есть (LaTeX текстом, жёлтым) — текст не
теряется, а место видно при проверке."""

from __future__ import annotations

import re

from docx.enum.text import WD_ALIGN_PARAGRAPH, WD_COLOR_INDEX
from lxml import etree

_MATH_NS = "http://schemas.openxmlformats.org/officeDocument/2006/math"
_W_NS = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"
# слово DeepSeek по частям: \(LaTeX\), <sub>…</sub>, <sup>…</sup>, обычный текст
_PART_RE = re.compile(r"\\\((.*?)\\\)|<sub>(.*?)</sub>|<sup>(.*?)</sup>|([^\\<]+|[\\<])", re.S)


def latex_to_omml(latex: str) -> etree._Element | None:
    """m:oMath для LaTeX; None — не удалось перевести."""
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
    """Формула отдельным абзацем по центру."""
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
    """Слово с разметкой DeepSeek в абзац: части текста — прогонами (индексы — подстрочным и
    надстрочным шрифтом), \\(LaTeX\\) — формулой Word в строке."""
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
