"""Повторная проверка номера в стандартной строке реквизитов акта."""

from __future__ import annotations

import re

from PIL import Image, ImageOps

from npa_ocr.vlm_marks import Word, ds_blocks
from npa_ocr.vlm_reread import AskFn, clean
from npa_ocr.vlm_tables import VlmUnavailable

_NUMBER = re.compile(r"(?:№|N[o°.]?)\s*(\d{1,5})\b", re.I)


def restore_inline_requisites(lines: list[list[Word]], image: Image.Image,
                              ask: AskFn, secondary_ask=None) -> list[list[Word]]:
    """Read a damaged inline date/number from its own crop at two scales."""
    from datetime import datetime
    result = []
    for line in lines:
        text = " ".join(w.text for w in line)
        damaged = (re.match(r"^от(?:\s|[0-9«\"'ГТЯ])", text, re.I) and not re.search(r"\d{2}\.\d{2}\.\d{4}",text)
                   and min(w.box[1] for w in line) < .4 * image.height
                   and len(re.findall(r"\d",text)) >= 4 and len(text) < 90)
        if not damaged:
            result.append(line)
            continue
        box = (max(0,min(w.box[0] for w in line)-12),max(0,min(w.box[1] for w in line)-12),
               min(image.width,max(w.box[2] for w in line)+20),min(image.height,max(w.box[3] for w in line)+12))
        crop=image.crop(box)
        readings=[]
        primary_candidates=[]
        for scale in (2,3):
            part=ImageOps.expand(crop.resize((crop.width*scale,crop.height*scale)),24,'white')
            try:
                raw,reason,lp=ask(part,'Free OCR.')
            except (VlmUnavailable,TimeoutError):
                break
            match=re.fullmatch(r"\s*от\s*(\d{2}\.\d{2}\.\d{4})\s*(?:№|N)\s*(\d{1,5})[.,]?\s*",clean(raw),re.I)
            loose=re.search(r"(\d{2}\.\d{2}\.\d{4}).*?(?:№|No\.?|N)\s*(\d{1,5})",clean(raw),re.I)
            if loose is not None and reason!='length':
                primary_candidates.append(loose.groups())
            if match is None or reason=='length' or lp < -.6:
                break
            try:
                datetime.strptime(match[1],'%d.%m.%Y')
            except ValueError:
                break
            readings.append(match.groups())
        if not (len(readings)==2 and readings[0]==readings[1]) and secondary_ask is not None:
            readings=[]
            for scale in (2,3):
                part=ImageOps.expand(crop.resize((crop.width*scale,crop.height*scale)),24,'white')
                try:
                    raw=secondary_ask(part,
                        'Перепиши только строку с датой и номером документа. '
                        'Не исправляй и не угадывай цифры. Формат: от ДД.ММ.ГГГГ № НОМЕР. '
                        'Если цифры неразборчивы, напиши «не распознано».')
                except (VlmUnavailable,TimeoutError,RuntimeError):
                    break
                match=re.fullmatch(r"\s*от\s*(\d{2}\.\d{2}\.\d{4})\s*№\s*(\d{1,5})[.,]?\s*",raw,re.I)
                if match is None:break
                try:datetime.strptime(match[1],'%d.%m.%Y')
                except ValueError:break
                readings.append(match.groups())
            # Agreement between scales of one model is insufficient when
            # the independent reader gave a contradictory date or number.
            if readings and readings[0] not in primary_candidates:
                readings=[]
        if len(readings)==2 and readings[0]==readings[1]:
            date,number=readings[0]
            result.append([Word(f'от {date} № {number}',100,box,restored=True)])
        else:
            from dataclasses import replace
            result.append([replace(w,conf=0,restored=True) for w in line])
    return result


def restore_header_number(lines: list[list[Word]], raw: str | None, image: Image.Image,
                          ask: AskFn) -> list[list[Word]]:
    """Восстанавливает номер только при согласии двух чтений фрагмента.

    Нужны заголовок вида акта и отдельная дата слева в верхней части листа.
    Номер читается из правой части той же строки. Имя PDF и внешние сведения
    не используются. При несогласии движков исходный текст сохраняется.
    """
    if not raw:
        return lines
    blocks = ds_blocks(raw, image.width, image.height)
    titles = [text for label, box, text in blocks if label == "title" and box[3] < .3 * image.height]
    if not any(re.sub(r"\s+", "", text).upper() in ("ПОСТАНОВЛЕНИЕ", "ПРИКАЗ", "РАСПОРЯЖЕНИЕ") for text in titles):
        return lines
    dates = [box for label, box, text in blocks if label == "text" and box[0] < .4 * image.width
             and box[3] < .35 * image.height and re.fullmatch(r"\d{2}\.\d{2}\.\d{4}", text.strip())]
    if len(dates) != 1:
        return lines
    date = dates[0]
    box = (int(.62 * image.width), max(0, int(date[1] - 8)), int(.94 * image.width), int(date[3] + 8))
    cy = (box[1] + box[3]) / 2
    if any(_NUMBER.search(" ".join(w.text for w in line if w.box[0] >= box[0]))
           for line in lines if abs(sum((w.box[1] + w.box[3]) / 2 for w in line) / len(line) - cy) < 35):
        return lines
    import pytesseract
    from npa_ocr.ocr import _resolve_tesseract_cmd

    cmd = _resolve_tesseract_cmd()
    if cmd:
        pytesseract.pytesseract.tesseract_cmd = cmd
    crop = image.crop(box)

    def read(part: Image.Image):
        """Не прерывает документ при недоступности необязательного дочитывания."""
        try:
            return ask(part, "Free OCR.")
        except (VlmUnavailable, TimeoutError):
            return "", "error", float("-inf")

    tess = pytesseract.image_to_string(crop, lang="rus+eng", config="--psm 7")
    a = _NUMBER.search(tess)
    if a is not None:
        response, reason, _ = read(crop)
        answer = clean(response)
        b = _NUMBER.search(answer) or re.fullmatch(r"\s*(\d{1,5})[.]?\s*", answer)
        if reason == "length" or b is None:
            return lines
        if a[1] != b[1]:
            return lines
        number = a[1]
    else:
        # У рукописного номера печатный знак № читается отдельно. Его
        # координаты задают фрагмент цифр; два масштаба должны дать одно число.
        sparse = pytesseract.image_to_data(crop, lang="rus+eng", config="--psm 11",
                                          output_type=pytesseract.Output.DICT)
        marks = [i for i, text in enumerate(sparse["text"]) if "№" in text]
        if not marks:
            return lines
        left = max(sparse["left"][i] + sparse["width"][i] for i in marks) + 8
        if left >= crop.width:
            return lines
        digits = crop.crop((left, 0, crop.width, crop.height))
        readings = []
        for scale in (2, 3):
            enlarged = ImageOps.expand(digits.resize((digits.width * scale, digits.height * scale)),
                                       border=24, fill="white")
            response, stop, confidence = read(enlarged)
            response = clean(response)
            got = _NUMBER.search(response) or re.fullmatch(r"\s*(\d{1,5})[.]?\s*", response)
            if stop == "length" or got is None or confidence < -.6:
                return lines
            readings.append(got[1])
        if readings[0] != readings[1]:
            return lines
        number = readings[0]
    x0, y0, x1, y1 = box
    restored = [Word("№", 100, (x0, y0, (x0 + x1) / 2, y1), restored=True),
                Word(number, 100, ((x0 + x1) / 2, y0, x1, y1), restored=True)]
    result = [[w for w in line if not (box[0] <= (w.box[0] + w.box[2]) / 2 <= box[2]
              and box[1] <= (w.box[1] + w.box[3]) / 2 <= box[3])] for line in lines]
    result = [line for line in result if line]
    result.append(restored)
    return sorted(result, key=lambda line: min(w.box[1] for w in line))
