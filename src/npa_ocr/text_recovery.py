"""Локальная перепроверка неполных заголовков по двум чтениям скана."""

from __future__ import annotations

import re
from difflib import SequenceMatcher

from PIL import Image

from npa_ocr.vlm_marks import Word, ds_blocks
from npa_ocr.response_validation import is_service_response


def recover_inline_codes(lines, image, ask):
    """Verify ambiguous chapter/section numerals using their source context."""
    from dataclasses import replace
    from npa_ocr.vlm_tables import VlmUnavailable
    from npa_ocr.vlm_reread import clean
    from PIL import ImageOps

    result=[]
    for line in lines:
        updated=list(line)
        for i,word in enumerate(line):
            if i==0 or not re.fullmatch(r"[ПШГУТХI|]+[.,;]?",word.text):
                continue
            if not re.fullmatch(r"глав[а-яё]*|раздел[а-яё]*",line[i-1].text,re.I):
                continue
            box=(max(0,line[i-1].box[0]-5),max(0,min(word.box[1],line[i-1].box[1])-5),
                 min(image.width,word.box[2]+5),min(image.height,max(word.box[3],line[i-1].box[3])+5))
            crop=image.crop(box);readings=[]
            for scale in (2,3):
                part=ImageOps.expand(crop.resize((crop.width*scale,crop.height*scale)),20,'white')
                try:raw,reason,lp=ask(part,'Free OCR.')
                except (VlmUnavailable,TimeoutError):break
                match=re.search(r"\b([IVXLC]+)[.,;]?\s*$",clean(raw))
                if reason=='length' or lp < -.6 or match is None:break
                readings.append(match[1])
            if len(readings)==2 and readings[0]==readings[1]:
                punctuation=re.search(r"[.,;]$",word.text)
                updated[i]=replace(word,text=readings[0]+(punctuation[0] if punctuation else ''),restored=True)
        result.append(updated)
    return result


def recover_titles(lines: list[list[Word]], raw: str | None, image: Image.Image) -> list[list[Word]]:
    """Возвращает потерянный текст заголовка только при подтверждении кропом.

    HTML модели не копируется в обычный текст. Нужны отдельная метка
    заголовка, близость нового OCR к тексту модели и те же числовые реквизиты.
    Восстановленный блок целиком помечается для сверки.
    """
    if not raw:
        return lines
    import pytesseract

    def norm(text):
        return re.sub(r"\W", "", text).lower().replace("ё", "е")

    result = [list(line) for line in lines]
    for label, box, expected in ds_blocks(raw, image.width, image.height):
        if label not in ("title", "table_caption") or len(expected) < 35 or is_service_response(expected):
            continue
        inside = lambda w: box[0] <= (w.box[0] + w.box[2]) / 2 <= box[2] and box[1] <= (w.box[1] + w.box[3]) / 2 <= box[3]
        old = [w for line in result for w in line if inside(w)]
        current = " ".join(w.text for w in old)
        if SequenceMatcher(None, norm(current), norm(expected)).ratio() > .97:
            continue
        x0, y0, x1, y1 = [int(v) for v in box]
        if x1 <= x0 or y1 <= y0:
            continue
        crop = image.crop((x0, y0, x1, y1))
        data = pytesseract.image_to_data(crop, lang="rus+eng", config="--psm 6", output_type=pytesseract.Output.DICT)
        words = [str(t).strip() for t in data["text"] if str(t).strip()]
        candidate = " ".join(words)
        if SequenceMatcher(None, norm(candidate), norm(expected)).ratio() < .9:
            continue
        if re.findall(r"\d+", candidate) != re.findall(r"\d+", expected):
            continue
        replacement = {}
        for i, text in enumerate(data["text"]):
            if not str(text).strip():
                continue
            key = (data["block_num"][i], data["par_num"][i], data["line_num"][i])
            x, y, width, height = (data[k][i] for k in ("left", "top", "width", "height"))
            replacement.setdefault(key, []).append(Word(str(text).strip(), float(data["conf"][i]),
                (x0 + x, y0 + y, x0 + x + width, y0 + y + height), restored=True))
        result = [[w for w in line if not inside(w)] for line in result]
        # A newly read Russian heading may turn II into П again. Apply the
        # same image-supported technical-token alignment used by the main
        # text pass, instead of undoing it during title recovery.
        from npa_ocr.vlm_text import reread_lines
        recovered,_=reread_lines(list(replacement.values()),raw,image.width,image.height)
        result = [line for line in result if line] + recovered
    return sorted(result, key=lambda line: min(w.box[1] for w in line))
