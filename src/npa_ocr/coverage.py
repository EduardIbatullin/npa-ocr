"""Recover independently observed note lines and remove geometric duplicates."""
from __future__ import annotations

import re
from difflib import SequenceMatcher


def _norm(text: str) -> str:
    return re.sub(r"\W", "", text).casefold().replace("ё", "е")


def merge_note_readings(previous: str, first: str, second: str) -> str | None:
    """Insert a definition only when two crop readings agree on its full text.

    No glossary is used. Existing values remain verbatim; insertions require
    neighbouring definitions already present in the original OCR result.
    """
    old = [p.strip() for p in previous.split(";")]
    a = [" ".join(p.split()) for p in first.split(";")]
    b = [" ".join(p.split()) for p in second.split(";")]
    if len(old) < 2 or len(a) < 3 or len(a) > len(old) + 6:
        return None
    def similarity(x, y):
        return SequenceMatcher(None, _norm(x), _norm(y), autojunk=False).ratio()
    anchors = {}
    for i, entry in enumerate(a):
        j = max(range(len(old)), key=lambda j: similarity(entry, old[j]))
        if similarity(entry, old[j]) >= .82:
            anchors[i] = j
    if len(anchors) < 2 or list(anchors.values()) != sorted(set(anchors.values())):
        return None
    additions = {}
    for i, entry in enumerate(a):
        entry = re.sub(r"^[|_\s]+|[|_\s]+$", "", entry)
        if i in anchors or not re.match(r"^[A-Za-zА-Яа-яЁё()]{1,12}\s*[—–-]\s*\S", entry):
            continue
        if not any(_norm(entry) == _norm(other) for other in b):
            continue
        before = [k for k in anchors if k < i]
        after = [k for k in anchors if k > i]
        if not before or not after or anchors[min(after)] != anchors[max(before)] + 1:
            continue
        additions.setdefault(anchors[min(after)], []).append(entry)
    if not additions:
        return None
    result = []
    for j, entry in enumerate(old):
        result.extend(additions.get(j, []))
        result.append(entry)
    return ";\n".join(result)


def remove_table_duplicates(page):
    """Remove text already in a table only if geometry and content both agree."""
    from dataclasses import replace
    from npa_ocr.ocr import TextBlock, TableBlock

    tables = [b for b in page.blocks if isinstance(b, TableBlock) and b.column_edges]
    def duplicate(block):
        if not isinstance(block, TextBlock):
            return False
        cx = (block.x0 + block.x1) / 2
        cy = block.y + block.height / 2
        if (block.height<=6 and re.fullmatch(r"[|Г‚]",block.text)
                and any(t.bbox[0]-12<=cx<=t.bbox[2]+12 and t.bbox[1]<=cy<=t.bbox[3] for t in tables)):
            return True
        if len(_norm(block.text)) < 12:
            return False
        for table in tables:
            x0,y0,x1,y1 = table.bbox
            if not (x0 <= cx <= x1 and y0 <= cy <= y1):
                continue
            text = _norm(block.text)
            if any(text in _norm(c.text) or SequenceMatcher(None,text,_norm(c.text)).ratio() >= .8
                   for c in table.cells):
                return True
        return False
    return replace(page, blocks=[b for b in page.blocks if not duplicate(b)])


def absorb_ruled_notes(image, tables, text_blocks):
    """Return footer definitions to a wide cell, requiring its bottom rule."""
    from dataclasses import replace
    import numpy as np
    from npa_ocr.ocr import TextBlock
    from npa_ocr.ocr_table import TableCell
    from npa_ocr.vlm_tables import _line_masks

    horizontal, _ = _line_masks(np.array(image.convert('L')))
    remaining = list(text_blocks)
    result = []
    for table in tables:
        x0,y0,x1,y1=table.bbox
        after=sorted((b for b in remaining if isinstance(b,TextBlock)
                      and b.y >= y1-3 and x0-15 <= b.x0 and b.x1 <= x1+15),key=lambda b:b.y)
        if not after or after[0].y-y1 > .025*image.height:
            result.append(table)
            continue
        if not re.match(r"\s*(?:[А-ЯA-Z]{2,6}\s*[-–—]|Примечани|\*\s*[-–—])",after[0].text):
            result.append(table)
            continue
        note=[]
        bottom=y1
        for b in after:
            if b.y-bottom > .025*image.height or re.match(r"\s*\d+[.)]\s",b.text):
                break
            if re.fullmatch(r'[»”".]\s*[.]?',b.text):
                break
            note.append(b)
            bottom=b.y+b.height
        left,right=max(0,int(x0)),min(image.width,int(x1))
        start,end=max(0,int(bottom)),min(image.height,int(bottom+35))
        rows=(horizontal[start:end,left:right]>0).sum(axis=1)
        if not note or not len(rows) or max(rows) < .65*(right-left):
            result.append(table)
            continue
        rule=start+int(rows.argmax())
        cell=TableCell(table.n_rows,0,1,table.n_cols,'\n'.join(b.text for b in note),
                       tuple(any(b.low_words) for b in note))
        result.append(replace(table,cells=table.cells+[cell],n_rows=table.n_rows+1,
                              bbox=(x0,y0,x1,float(rule))))
        remaining=[b for b in remaining if not any(b is n for n in note)]
    return result,remaining


def review_items(pages) -> list[dict]:
    """List flagged output addresses without claiming unflagged text is correct."""
    from npa_ocr.ocr import TableBlock,TextBlock
    out=[]
    for page in pages:
        for bi,block in enumerate(page.blocks):
            if isinstance(block,TableBlock):
                for c in block.cells:
                    if any(c.low_lines) or '[не распознано]' in c.text:
                        out.append(dict(page=page.page_number,source_pages=block.source_pages,block=bi,
                                        row=c.row,col=c.col,text=c.text,
                                        reason='unreadable_cell' if '[не распознано]' in c.text else 'flagged_cell'))
            elif isinstance(block,TextBlock) and (any(block.low_words) or '[не распознано]' in block.text):
                out.append(dict(page=page.page_number,block=bi,text=block.text,
                                reason='unreadable_text' if '[не распознано]' in block.text else 'flagged_text'))
    return out


def review_report(pages, *, source_inventory=None) -> dict:
    """Describe unresolved output without rewriting it or supplying replacements."""
    items=review_items(pages)
    unreadable=sum('[не распознано]' in item['text'] for item in items)
    report = dict(schema_version=1,result_policy='automatic_only',
                complete_transcription_verified=False,
                summary=dict(flagged_fragments=len(items),unreadable_fragments=unreadable),items=items)
    if source_inventory is not None or any(page.review_issues for page in pages):
        source_inventory = source_inventory or {}
        from npa_ocr.ocr import TableBlock
        from npa_ocr.source_layout import source_review
        coverage = []
        table_pages = set()
        for page in pages:
            for block in page.blocks:
                if isinstance(block, TableBlock):
                    table_pages.update(block.source_pages or [page.page_number])
        for page in pages:
            coverage.extend(dict(issue, page=page.page_number) for issue in page.review_issues)
            source = source_inventory.get(page.page_number, {})
            coverage.extend(source_review(page.page_number, source,
                text_only=not page.ocr_applied and not page.blocks,
                output_tables=int(page.page_number in table_pages)))
            if not page.ocr_applied and page.blocks and 'native_characters' in source:
                from collections import Counter
                text = ''.join(''.join(c.text for c in b.cells) if isinstance(b, TableBlock)
                               else getattr(b, 'text', '') for b in page.blocks)
                actual = Counter(c for c in text if not c.isspace())
                expected = Counter(source['native_characters'])
                missing, extra = expected-actual, actual-expected
                if missing or extra:
                    coverage.append(dict(page=page.page_number,
                        reason='native_text_character_mismatch',
                        missing_characters=dict(missing), extra_characters=dict(extra)))
        report.update(schema_version=2, source_inventory=source_inventory,
                      coverage_issues=coverage)
    return report


def recover_table_closers(image, tables, text_blocks):
    """Read punctuation immediately below tables; never infer it from syntax."""
    import pytesseract
    from PIL import ImageOps
    from npa_ocr.ocr import TextBlock

    result=list(text_blocks)
    for table in tables:
        x0,y0,x1,y1=table.bbox
        after=[b for b in result if isinstance(b,TextBlock) and b.y>=y1]
        if any(b.y-y1<90 and re.fullmatch(r'\s*»\s*\.?\s*',b.text) for b in after):
            continue
        end=min([b.y-3 for b in after]+[image.height,y1+90])
        if end-y1<12:
            continue
        left=max(0,int(x1-200));right=min(image.width,int(x1+90))
        crop=image.crop((left,int(y1),right,int(end)))
        part=ImageOps.expand(crop.resize((crop.width*3,crop.height*3)),20,'white')
        a=pytesseract.image_to_string(part,lang='rus',config='--psm 6').strip()
        b=pytesseract.image_to_string(part,lang='rus',config='--psm 11').strip()
        if not re.fullmatch(r'»\s*\.?',a) or not re.fullmatch(r'»\s*\.?',b):
            continue
        data=pytesseract.image_to_data(part,lang='rus',config='--psm 6',output_type=pytesseract.Output.DICT)
        ids=[i for i,s in enumerate(data['text']) if '»' in s]
        if not ids:continue
        i=ids[0];x=left+(data['left'][i]-20)/3;y=y1+(data['top'][i]-20)/3
        result.append(TextBlock(y,a.replace(' ',''),x,x+data['width'][i]/3,data['height'][i]/3,
                                low_words=(True,)))
    return sorted(result,key=lambda b:b.y)
