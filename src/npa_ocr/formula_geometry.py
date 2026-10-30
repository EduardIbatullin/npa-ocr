"""Различение букв и подстрочных цифр по расположению глифов на скане."""
from __future__ import annotations

from statistics import median
from PIL import Image, ImageOps

from npa_ocr.scientific_text import chemical_indices, is_chemical_formula


def read_formula(image: Image.Image, scale: int = 4) -> tuple[str, bool] | None:
    """Возвращает формулу и признак исправления класса знака по геометрии.

    Состав вещества неизвестен функции. Маленький знак ниже базовой линии
    читается как цифра, а полноразмерный ноль — отдельно как буква.
    Неоднозначный отдельный знак не подменяется ожидаемым из справочника.
    """
    import cv2
    import numpy as np
    import pytesseract

    if image.convert('L').getextrema()[0] == image.convert('L').getextrema()[1]:
        return None
    part = ImageOps.expand(image.resize((image.width * scale, image.height * scale), Image.Resampling.LANCZOS).convert('L'), 32, fill=255)
    gray = np.array(part)
    rows = [s.split() for s in pytesseract.image_to_boxes(part, lang='eng', config='--psm 7').splitlines()]
    boxes = [(r[0], *map(int, r[1:5])) for r in rows if len(r) >= 5]
    initial_heights = [b[4]-b[2] for b in boxes if b[0].isupper()]
    if not initial_heights:
        return None
    tolerance = .15 * median(initial_heights)
    refined = []
    for ch, x0, y0, x1, y1 in boxes:
        region = (gray[part.height-y1:part.height-y0, x0:x1] < 180).astype(np.uint8)
        if not region.size:
            return None
        n, _, components, _ = cv2.connectedComponentsWithStats(region)
        if n < 2:
            return None
        anchor = max(components[1:], key=lambda s:s[cv2.CC_STAT_AREA])
        nearby = [s for s in components[1:] if s[cv2.CC_STAT_TOP] <= anchor[1]+anchor[3]+tolerance
                  and s[cv2.CC_STAT_TOP]+s[cv2.CC_STAT_HEIGHT] >= anchor[1]-tolerance]
        top = min(s[1] for s in nearby)
        bottom = max(s[1]+s[3] for s in nearby)
        refined.append((ch,x0,y1-bottom,x1,y1-top))
    boxes = refined
    caps = [b for b in boxes if b[0].isupper()]
    if len(caps) < 2:
        return None
    height = median(b[4] - b[2] for b in caps)
    baseline = median(b[2] for b in caps)
    has_subscripts = any(b[4]-b[2] < .78*height and b[2] < baseline-.03*height for b in boxes)
    output = []
    changed = False
    for ch, x0, y0, x1, y1 in boxes:
        width, h = x1-x0, y1-y0
        if (not ch.isupper() and not ch.isdigit() and h >= .8*height
                and sum(max(0,min(x1,b[3])-max(x0,b[1])) > .25*width
                        for b in boxes if b != (ch,x0,y0,x1,y1)) >= 2):
            changed = True
            continue
        subscript = h < .78 * height and y0 < baseline - .03 * height
        tall_zero = ch == '0' and h >= .9 * height and has_subscripts
        if tall_zero:
            # В записи с настоящими нижними индексами полноразмерный
            # овал на строке элементов — O, а не ещё одна цифра индекса.
            output.append('O')
            changed = True
            continue
        if subscript and (not ch.isdigit() or ch in '369'):
            original_char = ch
            glyph = ImageOps.expand(part.crop((x0-2,part.height-y1-2,x1+2,part.height-y0+2)), 24, fill=255)
            allowed = '0123456789' if subscript else 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
            contours, hierarchy = cv2.findContours((np.array(glyph) < 180).astype(np.uint8), cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
            holes = [i for i, c in enumerate(contours) if hierarchy is not None
                     and hierarchy[0][i][3] >= 0 and cv2.contourArea(c) >= 2 * scale]
            if subscript and len(holes) == 2:
                # Две замкнутые петли подтверждают 8 даже тогда, когда
                # языковая модель Tesseract предпочла похожую 3.
                output.append('8')
                changed = True
                continue
            readings = []
            for threshold in (None, 180):
                sample = glyph if threshold is None else glyph.point(lambda p: 255 if p >= threshold else 0)
                readings.append(pytesseract.image_to_string(sample, lang='eng',
                    config='--psm 10 -c tessedit_char_whitelist='+allowed).strip())
            if readings[0] != readings[1] or len(readings[0]) != 1:
                return None
            ch = readings[0]
            changed = changed or ch != original_char
        output.append(ch)
    text = chemical_indices(''.join(output))
    return (text, changed) if is_chemical_formula(text) else None


def formula_consensus(image: Image.Image, alternate: Image.Image | None = None) -> tuple[str, bool] | None:
    """Два масштаба должны подтвердить исправление геометрического типа."""
    from collections import Counter
    images = [image] if alternate is None else [image, alternate]
    readings = [read_formula(part, scale) for part in images for scale in (3, 4)]
    count = Counter(r[0] for r in readings if r)
    if count:
        text, votes = count.most_common(1)[0]
        required = 2 if alternate is None else 3
        if votes >= required:
            return text, sum(bool(r and r[0] == text and r[1]) for r in readings) >= 2
    return None


def confirmed_formula(image: Image.Image, alternate: Image.Image | None = None) -> str | None:
    got = formula_consensus(image, alternate)
    return got[0] if got and got[1] else None
