"""Независимая проверка коротких ячеек по фрагменту исходного скана."""

from __future__ import annotations

import re
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from bisect import bisect_right
from difflib import SequenceMatcher

from PIL import Image, ImageOps

from npa_ocr.response_validation import matches_cell_type
from npa_ocr.scientific_text import ROMAN_CODE, chemical_indices, section_heading

_LOOK = str.maketrans({"1": "I", "l": "I", "|": "I", "Г": "I", "Т": "I", "х": "X",
                       "П": "II", "Ш": "III", "У": "V", "Х": "X", "H": "II", "T": "I"})


def _code(english: str, russian: str, previous: str, support: str = "") -> str | None:
    """Читает римскую часть, сохраняя кириллическую букву подразряда."""
    english = english.strip(" ‘’.:;\n")
    russian = russian.strip(" ‘’.:;\n")
    russian = re.sub(r"(?<=[IVXLC1l|ГТПШУХх])Вв?(?=\s|[.,—-]|$)","в",russian)
    ru = re.match(r"^([IVXLC1l|ГТПШУХх]+)([абвг])?(?=\s|\(|[.,]|$)", russian)
    old = re.match(r"^([IVXLC1l|ГТПШУХ]+)([абвг])?(?=\s|\(|[.,]|$)", previous)
    suffix = (ru[2] if ru else None) or (old[2] if old else None) or ""
    # Нижний регистр в английском чтении отделяет букву подразряда от I/V/X.
    en = re.match(r"^([IVXLC1l|HT]+)(?:[absrvB6])?(?=\s|\(|$)", english)
    if en is None:
        return None
    prefix = en[1].translate(_LOOK)
    strong = bool(ru and any(ch in ru[1] for ch in "ПШГ"))
    if strong:
        visual = ru[1].translate(_LOOK)
        if re.fullmatch(r"[IVXLC]+", visual):
            if visual!=prefix:
                evidence=re.match(r"^([IVXLC1l|ГТПШУХх!x]+)",support.strip())
                observed=evidence[1].translate(_LOOK).replace('x','X') if evidence else ''
                previous_prefix=old[1].translate(_LOOK) if old else ''
                if observed in (visual,prefix):
                    prefix=observed
                elif previous_prefix==visual:
                    prefix=visual
                elif previous_prefix!=prefix:
                    return None
            else:
                prefix = visual
    if old and ROMAN_CODE.fullmatch(previous) and old[1] != prefix and not strong:
        evidence = re.match(r"^([IVXLC1l|ГТПШУХх!x]+)", support.strip())
        corroborated = evidence and evidence[1].translate(_LOOK).replace("x", "X") == prefix
        if not corroborated:
            return None
    tail = previous[previous.find("("):] if "(" in previous else ""
    result = prefix + suffix + (" " + tail if tail else "")
    return result if ROMAN_CODE.fullmatch(result) else None


def _numeric(text: str, kind: str) -> str | None:
    """Удаляет только отдельные следы рамки вокруг короткого значения."""
    text = text.strip().strip("| ")
    if kind == "index":
        return text.rstrip(".") if re.fullmatch(r"\d{1,4}[.]?", text) else None
    if re.fullmatch(r"[-–—−]", text):
        return "—"
    m = re.fullmatch(r"(\d+(?:[.,]\d+)?\*{0,2})\s*[|.]?", text)
    return m[1] if m else None


def _mixed_labels(previous: str, local: str) -> str:
    """Переносит подтверждённые латинские обозначения без переписывания норм."""
    def norm(value):
        return re.sub(r"\W", "", value).lower()

    pieces = previous.split(";")
    for code, definition in re.findall(r"\b(W[hkd])\s*[-–—]\s*([^;]+)", " ".join(local.split())):
        candidates = []
        for i, piece in enumerate(pieces):
            match = re.match(r"\s*\S{1,8}\s*[-–—]\s*(.+)", piece, re.S)
            if match:
                candidates.append((SequenceMatcher(None, norm(match[1]), norm(definition)).ratio(), i, match[1]))
        if candidates:
            score, i, text = max(candidates)
            if score >= .9:
                pieces[i] = code + " — " + text
    value = ";".join(pieces)
    old = re.search(r"ГОСТ Р\s+([\d./-]+)", value)
    new = re.search(r"ГОСТ Р\s+([\d.-]+)", local)
    if old and new and re.sub(r"\D", "", old[1]) == re.sub(r"\D", "", new[1]):
        value = value[:old.start(1)] + new[1] + value[old.end(1):]
    return value


def _consensus_words(previous: str, first: str, second: str) -> str | None:
    """Исправляет короткие расхождения лишь при согласии двух чтений названия."""
    def words(text):
        return re.sub(r"(\S)-\s*\n\s*(\S)", r"\1-\2", text).split()

    def norm(text):
        return re.sub(r"[^\w]", "", text).lower().replace("ё", "е")

    old, a, b = words(previous), words(first), words(second)
    na, nb, no = list(map(norm, a)), list(map(norm, b)), list(map(norm, old))
    if na != nb or not a or SequenceMatcher(None, norm(previous), norm(first)).ratio() < .8:
        return None
    changed = False
    for op, x0, x1, y0, y1 in SequenceMatcher(None, no, na, autojunk=False).get_opcodes():
        if op != "replace" or x1 - x0 != y1 - y0 or x1 - x0 > 2:
            continue
        for x, y in zip(range(x0, x1), range(y0, y1)):
            if re.findall(r"\d+", old[x]) != re.findall(r"\d+", a[y]):
                continue
            if min(len(no[x]), len(na[y])) < 5 or SequenceMatcher(None, no[x], na[y]).ratio() < .75:
                continue
            old[x] = a[y]
            changed = True
    return " ".join(old) if changed else None


def split_index_cells(image: Image.Image, cells, row_edges: list[int]) -> int:
    """Разделяет ошибочно слитые номера строк по двум чтениям изображения.

    Потерянная короткая линия иногда объединяет соседние позиции в графе
    «№». Разделение разрешено лишь для двух или более последовательных
    номеров, которые действительно прочитаны обоими языковыми проходами.
    """
    import pytesseract

    replacements = []
    for cell in list(cells):
        if cell.c != 0 or cell.cs != 1 or cell.rs < 2:
            continue
        part = image.crop((cell.x0 + 5, cell.y0 + 5, cell.x1 - 5, cell.y1 - 5)).convert("RGB")
        part = ImageOps.expand(part.resize((part.width * 2, part.height * 2)), border=20, fill="white")
        tokens = []
        numbers = []
        for psm in (6, 11):
            data = pytesseract.image_to_data(part, lang="eng", config=f"--psm {psm}", output_type=pytesseract.Output.DICT)
            tokens = [(s.strip().rstrip("."), data["top"][i]) for i, s in enumerate(data["text"])
                      if re.fullmatch(r"\d{1,3}[.]?", s.strip())]
            russian = pytesseract.image_to_string(part, lang="rus", config=f"--psm {psm}")
            numbers = [int(s) for s, _ in tokens]
            if (len(numbers) >= 2 and all(b == a + 1 for a, b in zip(numbers, numbers[1:]))
                    and numbers == [int(s) for s in re.findall(r"\b\d{1,3}\b", russian)]):
                break
        else:
            continue
        starts = [cell.r] + [max(cell.r + 1, bisect_right(row_edges, cell.y0 + 5 + (y - 20) / 2) - 1)
                             for _, y in tokens[1:]]
        if starts != sorted(set(starts)) or starts[-1] >= cell.r + cell.rs:
            continue
        ends = starts[1:] + [cell.r + cell.rs]
        new = [replace(cell, r=r, rs=end - r, y0=row_edges[r] + 2, y1=row_edges[end] - 2,
                       tess=str(number) + ".", text="", ds_ids=[], literal=None)
               for number, r, end in zip(numbers, starts, ends)]
        replacements.append((cell, new))
    for old, new in replacements:
        cells.remove(old)
        cells.extend(new)
    cells.sort(key=lambda c: (c.r, c.c))
    return len(replacements)


def verify_cells(image: Image.Image, cells) -> None:
    """Перепроверяет типизированные короткие графы двумя чтениями фрагмента.

    Значения не выводятся из соседних строк или диапазонов. Новое чтение
    остаётся помеченным для человека, даже при согласии конфигураций OCR.
    """
    import pytesseract

    selected = [c for c in cells if c.literal is None
                and c.expected in ("number", "index", "code", "formula", "section", "mixed", "name")
                and (len(c.text) < 100 or c.expected in ("section", "mixed", "name"))
                and c.y1 - c.y0 < image.height * .6]

    def read(cell):
        margin = 5 if cell.expected=="code" else 3
        if (cell.expected in ("number", "index") and len(cell.text)<12
                and not re.search(r"[А-Яа-яA-Za-z]",cell.text)):
            margin = min(12,max(3,int(.08*(cell.x1-cell.x0))))
        vertical_margin = 5 if cell.expected=="code" else 3
        box = (cell.x0 + margin, cell.y0 + vertical_margin, cell.x1 - margin, cell.y1 - vertical_margin)
        if box[2] <= box[0] or box[3] <= box[1]:
            return None
        part = image.crop(box).convert("RGB")
        if cell.expected=='number' and re.fullmatch(r'\s*\d+(?:[.,]\d+)?\.{2,}\s*',cell.text):
            # A nearby broken rule can become an ellipsis after a number.
            # Remove only independently detected table rules, then read
            # the pixels again; never strip ellipses from the text itself.
            import cv2
            import numpy as np
            from npa_ocr.vlm_tables import _line_masks
            gray=np.array(image.convert('L'))
            horizontal,vertical=_line_masks(gray)
            gray[cv2.bitwise_or(horizontal,vertical)>0]=255
            part=Image.fromarray(gray).crop(box).convert('RGB')
            from npa_ocr.stroke_geometry import remove_broken_side_rule
            part=remove_broken_side_rule(part)
        scale = 3 if cell.expected in ("number", "index", "mixed") else 2
        part = ImageOps.expand(part.resize((part.width * scale, part.height * scale)), border=20, fill="white")
        if cell.expected == "name":
            a = pytesseract.image_to_string(part, lang="rus+eng", config="--psm 6")
            b = pytesseract.image_to_string(part, lang="rus", config="--psm 6")
            return _consensus_words(cell.text, a, b)
        if cell.expected == "section":
            ru = pytesseract.image_to_string(part, lang="rus+eng", config="--psm 7").strip()
            return section_heading(ru) or section_heading(cell.text)
        if cell.expected == "mixed":
            local = pytesseract.image_to_string(part, lang="rus+eng", config="--psm 6")
            ru = " ".join(local.split())
            value = cell.text
            if "Chemical Abstracts Service Registry Number" in ru and ";" in ru and ";" in cell.text:
                value = ru.split(";", 1)[0].lstrip("_‚| ") + ";" + cell.text.split(";", 1)[1]
            value = _mixed_labels(value, local)
            from npa_ocr.coverage import merge_note_readings
            second = pytesseract.image_to_string(part, lang="rus", config="--psm 6")
            value = merge_note_readings(value, local, second) or value
            if re.search(r"[CС][AА][$5]",value) and re.search(r"\bCAS\b",local):
                english=pytesseract.image_to_string(part,lang="eng",config="--psm 6")
                if re.search(r"\bCAS\b",english):
                    value=re.sub(r"(?<!\w)[CС][AА][$5](?!\w)","CAS",value)
            return value if value != cell.text else None
        if cell.expected == "code":
            en = pytesseract.image_to_string(part, lang="eng", config="--psm 7")
            ru = pytesseract.image_to_string(part, lang="rus", config="--psm 7")
            return _code(en, ru, cell.text or cell.tess, support=cell.tess)
        if cell.expected == "formula":
            a = pytesseract.image_to_string(part, lang="eng", config="--psm 7").strip()
            b = pytesseract.image_to_string(part, lang="eng", config="--psm 6").strip()
            a, b = chemical_indices(a.replace(" ", "")), chemical_indices(b.replace(" ", ""))
            previous = chemical_indices(cell.text.replace(" ", ""))
            if matches_cell_type(previous, "formula") and a != previous:
                # Согласие двух конфигураций Tesseract не даёт права менять
                # состав уже корректно записанной формулы.
                return None
            return a if a == b and matches_cell_type(a, "formula") else None
        a = _numeric(pytesseract.image_to_string(part, lang="eng", config="--psm 7"), cell.expected)
        b = _numeric(pytesseract.image_to_string(part, lang="rus", config="--psm 7"), cell.expected)
        if a is None and b is not None and b!="—":
            constrained = _numeric(pytesseract.image_to_string(part,lang="eng",
                config="--psm 7 -c tessedit_char_whitelist=0123456789.,-"),cell.expected)
            if constrained==b:
                a=constrained
        if a is not None and b is None and not re.search(r"\d", cell.text):
            # Пустой результат можно дополнить двумя совпавшими чтениями
            # фрагмента; явное противоречащее число русского прохода сохраняется.
            b = _numeric(pytesseract.image_to_string(part, lang="eng", config="--psm 6"), cell.expected)
        if a == "—" and re.search(r"\d", cell.text):
            # В повёрнутой узкой графе цифра 1 выглядит горизонтальным
            # штрихом. Без подтверждённой ориентации число не стирается.
            return None
        return a if a is not None and a == b else None

    with ThreadPoolExecutor(max_workers=4) as pool:
        for cell, value in zip(selected, pool.map(read, selected)):
            if value is not None:
                cell.text = value
                cell.tess = value
                cell.src = "local"


def orient_numbering(image: Image.Image, cells) -> None:
    """Confirm a sideways numbering column from several actual digit crops.

    Sequence structure selects candidates only; it never supplies missing
    values. Upright quantities (especially 1) cannot be rotated into dashes.
    """
    import pytesseract
    for col in {c.c for c in cells}:
        numbered=[c for c in cells if c.c==col and c.cs==1 and re.fullmatch(r'\d{1,3}',c.text.strip())]
        numbered.sort(key=lambda c:c.r)
        values=[int(c.text.strip()) for c in numbered]
        if len(values)<6 or min(values)!=1 or not all(0<b-a<=3 for a,b in zip(values,values[1:])):
            continue
        distinctive=[c for c in numbered if int(c.text.strip()) not in (1,8,11)]
        sample=distinctive[:3]
        if len(sample)<3:continue
        def score(angle):
            count=0
            for c in sample:
                part=image.crop((c.x0+4,c.y0+4,c.x1-4,c.y1-4)).rotate(angle,expand=True)
                part=ImageOps.expand(part.resize((part.width*3,part.height*3)),20,'white')
                text=pytesseract.image_to_string(part,lang='eng',config='--psm 7').strip().rstrip('.')
                count+=text==c.text.strip()
            return count
        upright=score(0)
        if upright>=2:continue
        scores={a:score(a) for a in (-90,90)}
        angle=max(scores,key=scores.get)
        if scores[angle]<3:continue
        for c in cells:
            if c.c==col and c.cs==1:
                c.rot=angle
                c.number_rotation=True
                c.literal=None
