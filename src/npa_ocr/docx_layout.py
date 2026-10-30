"""Pagination and typography for editable OCR tables (no content corrections)."""
from __future__ import annotations

import re

from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Pt


def flag(parent, name: str, value: str = "1") -> None:
    node = parent.find(qn(name))
    if node is None:
        node = OxmlElement(name)
        parent.append(node)
    node.set(qn("w:val"), value)


def header_rows(block) -> int:
    """Repeat only actual leading headings, stopping before data/section rows."""
    for row in range(min(block.n_rows, 8)):
        cells = [c for c in block.cells if c.row == row]
        text = " ".join(c.text for c in cells)
        if not text.strip():
            continue
        first = next((c for c in cells if c.col == 0), None)
        if (first and re.fullmatch(r"\s*\d{1,4}[.)]?\s*", first.text)
                and not all(re.fullmatch(r"\s*\d{1,2}\s*", c.text) for c in cells)):
            return row
        if any(c.col_span == block.n_cols for c in cells):
            return row
        if row == 0 and not re.search(r"№|наименован|показател|норматив|размер|формул|веществ|оборудован", text, re.I):
            return 0
    return 0


def format_table(table, block) -> None:
    """Protect short rows and tokens; allow tall merged text to cross a page."""
    n_header = header_rows(block)
    for ri, row in enumerate(table.rows):
        props = row._tr.get_or_add_trPr()
        if ri < n_header:
            flag(props, "w:tblHeader")
        starting = [c for c in block.cells if c.row == ri]
        # No blanket cantSplit for a note spanning half a page or a tall group.
        if ri < n_header or (starting and all(len(c.text) <= 180 and c.row_span == 1 for c in starting)):
            flag(props, "w:cantSplit")
        seen = set()
        for cell in row.cells:
            if cell._tc in seen:
                continue
            seen.add(cell._tc)
            margins=cell._tc.get_or_add_tcPr().find(qn('w:tcMar'))
            if margins is None:
                margins=OxmlElement('w:tcMar');cell._tc.get_or_add_tcPr().append(margins)
            for side in ('left','right'):
                node=margins.find(qn('w:'+side))
                if node is None:
                    node=OxmlElement('w:'+side);margins.append(node)
                node.set(qn('w:w'),'30');node.set(qn('w:type'),'dxa')
            for para in cell.paragraphs:
                para.paragraph_format.space_before = Pt(0)
                para.paragraph_format.space_after = Pt(0)
                para.paragraph_format.line_spacing = 1
                # Word otherwise allows an English word to break mid-token.
                flag(para._p.get_or_add_pPr(), "w:wordWrap", "0")
                if ri < n_header:
                    para.paragraph_format.keep_with_next = True
                text = para.text.strip()
                compact = bool(re.fullmatch(r"[\w₀-₉⁰-⁹.,%°+−–—*/()\-]{1,24}", text))
                if compact:
                    flag(cell._tc.get_or_add_tcPr(), "w:noWrap")
                width_pt = (cell.width or Pt(55)) / 12700 - 7
                size = min(9.5, max(7, width_pt / max(1, len(text) * .57))) if compact else 9.5
                for run in para.runs:
                    run.font.size = Pt(size)


def set_document_language(doc, language: str = "ru-RU") -> None:
    """Set defaults and Normal so inherited template English does not win."""
    defaults = doc.styles.element.find(qn("w:docDefaults"))
    if defaults is not None:
        rpr = defaults.find(qn("w:rPrDefault"))
        if rpr is not None:
            child = rpr.find(qn("w:rPr"))
            if child is not None:
                flag(child, "w:lang", language)
    flag(doc.styles["Normal"].element.get_or_add_rPr(), "w:lang", language)


def keep_table_titles(doc) -> None:
    """Bind the final title paragraphs to their table, across a source caption."""
    pending = []
    previous_table=None
    for child in doc.element.body:
        if child.tag == qn("w:p"):
            text=''.join(child.itertext()).strip()
            if previous_table is not None and re.fullmatch(r'(?:»\s*\.?)+' ,text):
                from docx.text.paragraph import Paragraph
                paragraph=Paragraph(child,doc)
                paragraph.paragraph_format.space_before=Pt(0)
                paragraph.paragraph_format.space_after=Pt(0)
                paragraph.paragraph_format.line_spacing=1
                for run in paragraph.runs:run.font.size=Pt(10)
                last_row=previous_table.findall(qn('w:tr'))[-1]
                for p in last_row.iter(qn('w:p')):
                    flag(p.get_or_add_pPr(),'w:keepNext')
            previous_table=None
            pending.append(child)
        elif child.tag == qn("w:tbl"):
            # Only the trailing caption/title cluster, never the entire act.
            tail = pending[-6:]
            start = next((i for i, p in enumerate(tail)
                          if re.search(r"Таблица\s+\d", "".join(p.itertext()))), None)
            if start is not None:
                for p in tail[start:]:
                    flag(p.get_or_add_pPr(), "w:keepNext")
            pending = []
            previous_table=child


def format_requisites(paragraph, available) -> None:
    """Use actual tab stops for a recognized date / city / number row."""
    from docx.enum.text import WD_TAB_ALIGNMENT
    match=re.fullmatch(r"(\d{2}\.\d{2}\.\d{4})\s+([А-Яа-яЁё. -]+?)\s+(№\s*\d{1,5})",paragraph.text)
    if match is None:return
    highlight=next((r.font.highlight_color for r in paragraph.runs if r.font.highlight_color),None)
    paragraph.clear()
    paragraph.paragraph_format.first_line_indent=Pt(0)
    stops=paragraph.paragraph_format.tab_stops
    stops.add_tab_stop(int(available/2),WD_TAB_ALIGNMENT.CENTER)
    stops.add_tab_stop(available,WD_TAB_ALIGNMENT.RIGHT)
    for i,text in enumerate(match.groups()):
        if i:paragraph.add_run('\t')
        run=paragraph.add_run(text)
        if highlight is not None and i!=1:run.font.highlight_color=highlight
