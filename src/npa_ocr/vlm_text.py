"""Второе прочтение текста страницы по ответу DeepSeek-OCR.

Tesseract с русским словарём не знает римских цифр, латинских переменных,
индексов и формул и пишет их похожими русскими буквами: «II» → «П»,
«IV» → «ГУ», «Z» → «7», «V» → «У», «16³» → «163», «K<sub>пр</sub>» → «Кир».
DeepSeek-OCR читает их верно: римские цифры — латиницей, индексы — разметкой
``<sub>``/``<sup>``, формулы — LaTeX (блок ``equation``, в тексте —
``\\(…\\)``). Русский текст при этом остаётся за Tesseract: DeepSeek иногда
пишет русские слова транслитом.

Правила :func:`reread_lines`:

- Слова DeepSeek заменяют слова Tesseract только в коротком расхождении,
  соседнем с совпавшим словом (выравнивание difflib по словам строк в рамке
  блока).
- Блок DeepSeek с транслитом (латинское слово из 4+ букв) не используется.
- Римская цифра принимается, только если слово Tesseract похоже на неё по
  начертанию («П» → II, «ГУ» → IV, «[» → I); диапазон «I–V» — если слово
  Tesseract состоит только из таких знаков.
- Латинская переменная (1–3 буквы, возможно с окончанием «-м», «-го»)
  принимается, если она не двойник русской буквы Tesseract (K/К, C/С), не её
  транслит («b» на месте «б») и не маркер пункта списка в начале блока.
- Индексы, степени и формулы в тексте хранятся разметкой (``Word.rich``); в
  разметку переводятся и индексы, записанные без неё: символы Юникода «Sᵢᵏ»,
  подчёркивание «C_t», заглавная латинская буква, слитая с русским индексом
  «Mфиатм» (только поверх неуверенного слова Tesseract).
- Цифра, прочитанная буквой или слитая со словом («Злет» → «3 лет»,
  «2013г.» → «2013 г.»), исправляется, если после замены букв-двойников цифр
  текст совпадает, а DeepSeek делит его только на стыке цифры и буквы или «№».
- Блок ``equation`` заменяет строки Tesseract в своей рамке формулой
  (:class:`Formula`).

Заменённые слова помечаются как восстановленные и выделяются в DOCX жёлтым.
"""

from __future__ import annotations

import difflib
import re
from dataclasses import dataclass

from npa_ocr.vlm_marks import Box, Word, ds_blocks

# Слово DeepSeek: \(LaTeX\) или слово с <sub>/<sup> целиком (внутри бывают пробелы:
# «P<sub>сд атм</sub>»), иначе — до пробела.
_TOKEN_RE = re.compile(r"\S*?\\\(.*?\\\)\S*|\S*?<su[bp]>.*?</su[bp]>(?:\S*?<su[bp]>.*?</su[bp]>)*\S*|\S+")
_TAG_RE = re.compile(r"</?su[bp]>")
_ROMAN_RE = re.compile(r"^[IVXLC]{1,6}$")
# диапазон: «I–V», «I-II»
_ROMAN_RANGE_RE = re.compile(r"^[IVXLC]{1,4}[-–—][IVXLC]{1,4}$")
_LATIN_VAR_RE = re.compile(r"^[A-Za-z]{1,3}(-[а-яё]{1,3})?$")
_LIST_MARKER_RE = re.compile(r"^[a-z]\)$")
_SUP_DIGITS = str.maketrans("0123456789", "⁰¹²³⁴⁵⁶⁷⁸⁹")
_SUBSUP_CHARS = "¹²³⁴⁵⁶⁷⁸⁹⁰₀₁₂₃₄₅₆₇₈₉"
# латинские буквы, неотличимые от русских на скане: на них в русском тексте вернее Tesseract
_HOMOGLYPHS = set("ABCEHKMOPTXaceopxy")
# русская буква Tesseract, которую DeepSeek пишет латинским транслитом: «б)» → «b)», «и» → «i».
# «У» на месте V и «7» на месте Z — не транслит, там прав DeepSeek
_TRANSLIT = {"a": "а", "b": "б", "v": "в", "g": "г", "d": "д", "e": "е", "z": "з", "i": "и", "k": "к",
             "l": "л", "m": "м", "n": "н", "o": "о", "p": "п", "r": "р", "s": "с", "t": "т", "u": "у",
             "f": "ф", "h": "х", "c": "ц"}
# как Tesseract с русским словарём пишет римские цифры
_ROMAN_LOOKALIKE = str.maketrans({"П": "II", "Ш": "III", "У": "V", "Г": "I", "Т": "I", "[": "I", "]": "I",
                                  "|": "I", "1": "I", "l": "I", "!": "I", "Н": "II", "Х": "X", "\\": ""})
# диапазон Tesseract пишет слитно и с потерей знаков: «I–IV» → «ГУ», «I–II» → «ЕП», «Г», «[-П», «1-П»
_ROMAN_RANGE_LOOKALIKE = "ПШУГТ[]|1l!НХЕ-–—"
_PUNCT = ".,;:()«»\"'’"
# индексы, записанные символами Юникода («Sᵢᵏ»), — в разметку <sub>/<sup>
_UNI_SUB = dict(zip("ᵢⱼₐₑₒₓₕₖₗₘₙₚₛₜᵣᵤᵥ₀₁₂₃₄₅₆₇₈₉", "ijaeoxhklmnpstruv0123456789"))
_UNI_SUP = dict(zip("ⁱʲᵏᵐⁿᵗˡʰᵃᵇᶜᵈᵉᶠᵍᵒᵖʳˢᵘᵛʷˣʸᶻ", "ijkmntlhabcdefgoprsuvwxyz"))
_UNDERSCORE_RE = re.compile(r"^(\(?)([A-Za-z]{1,3})_([0-9A-Za-zА-Яа-яЁё]+)([.,;:)]*)$")  # «C_t»
# латинская заглавная, слитая с русским индексом: «Mфиатм» (M с индексом «фиатм»)
_GLUED_RE = re.compile(r"^(\(?)([A-Z])([а-яё]{2,})([.,;:)]*)$")
# цифры, которые Tesseract читает буквами: «Злет» на месте «3 лет»
_DIGIT_LOOKALIKE = str.maketrans({"З": "3", "з": "3", "О": "0", "о": "0", "б": "6", "l": "1", "I": "1", "|": "1"})
_CYR_TWIN = str.maketrans("ABCEHKMOPTXaceopxy", "АВСЕНКМОРТХасеорху")
MAX_SPAN = 4  # длина расхождения (слов), в котором DeepSeek может заменить слова Tesseract
FORMULA_MARGIN = 10  # пикселей: строки Tesseract у рамки формулы — тоже её мусор


@dataclass(frozen=True)
class Formula:
    """Формула-блок страницы: LaTeX и рамка в пикселях."""

    latex: str  # без \[ \]
    box: Box


def plain(token: str) -> str:
    """Переводит разметку DeepSeek в простой текст.

    Например: «K<sub>пр</sub>» → «Kпр», «16<sup>3</sup>» → «16³»,
    ``\\(V_i\\)`` → «V_i».
    """
    token = re.sub(r"<sup>(\d+)</sup>", lambda m: m.group(1).translate(_SUP_DIGITS), token)
    token = _TAG_RE.sub("", token)
    token = re.sub(r"\\\((.*?)\\\)", r"\1", token)
    token = re.sub(r"\\text\{(.*?)\}", r"\1", token)
    return token.replace("{", "").replace("}", "").replace("\\", "")


def _norm(token: str) -> str:
    """Простой текст слова без знаков препинания по краям, в нижнем регистре."""
    return plain(token).strip(_PUNCT).lower()


def _core(token: str) -> str:
    """Простой текст слова без знаков препинания по краям."""
    return plain(token).strip(_PUNCT)


def _markup(token: str) -> str:
    """Переводит индексы, записанные без разметки, в ``<sub>``/``<sup>``.

    Например: «Sᵢᵏ» → «S<sub>i</sub><sup>k</sup>», «C_t» → «C<sub>t</sub>».
    """
    if any(c in _UNI_SUB or c in _UNI_SUP for c in token):
        out, mode = [], ""
        for c in token:
            m = "sub" if c in _UNI_SUB else "sup" if c in _UNI_SUP else ""
            if m != mode:
                out.append(f"</{mode}>" if mode else "")
                out.append(f"<{m}>" if m else "")
                mode = m
            out.append(_UNI_SUB.get(c) or _UNI_SUP.get(c) or c)
        out.append(f"</{mode}>" if mode else "")
        return "".join(out)
    m = _UNDERSCORE_RE.match(token)
    if m and "\\" not in token:
        return f"{m[1]}{m[2]}<sub>{m[3]}</sub>{m[4]}"
    return token


def _kind(token: str) -> str | None:
    """Вид слова DeepSeek, которому можно доверять больше, чем Tesseract.

    Returns:
        ``"latex"``, ``"index"``, ``"glued"``, ``"roman"``, ``"roman_range"``,
        ``"latin"`` или None для обычного слова.
    """
    if "\\(" in token:
        return "latex"
    if "<sub>" in token or "<sup>" in token or any(c in token for c in _SUBSUP_CHARS):
        return "index"
    if _GLUED_RE.match(token):
        return "glued"
    core = _core(token)
    if _ROMAN_RE.match(core):
        return "roman"
    if _ROMAN_RANGE_RE.match(core):
        return "roman_range"
    if _LATIN_VAR_RE.match(core):
        return "latin"
    return None


def _digit_split(ds: list[str], tess: list[Word]) -> bool:
    """Проверяет, что DeepSeek лишь отделил цифры от слов Tesseract.

    «3 лет» у DeepSeek и «Злет» у Tesseract совпадают после замены
    букв-двойников цифр; так же «2013г.» → «2013 г.». Разрешены только пробелы
    на стыке цифры и буквы или рядом с «№» («1316-р» не делится).
    """
    ds_joined = "".join(ds)
    if not any(c.isdigit() for c in ds_joined):
        return False
    if "".join(w.text for w in tess).translate(_DIGIT_LOOKALIKE) != ds_joined:
        return False
    for left, right in zip(ds, ds[1:]):
        a, b = left.rstrip(".")[-1:], right[:1]
        if not ((a.isdigit() and b.isalpha()) or (a.isalpha() and b.isdigit())
                or b == "№" or (a == "№" and b.isdigit())):  # «2019г.№89» → «2019 г. № 89»
            return False
    return True


def _translit_block(tokens: list[str]) -> bool:
    """Проверяет, что DeepSeek написал в блоке русские слова латиницей («oJJaCTH»)."""
    return any(len(re.sub(r"[^A-Za-z]", "", t)) >= 4 and not re.search(r"[А-Яа-яЁё\\<]", t) for t in tokens)


def _accept(ds: list[str], tess: list[Word]) -> bool:
    """Решает, можно ли заменить слова Tesseract словами DeepSeek в одном расхождении.

    Args:
        ds: Слова DeepSeek.
        tess: Слова Tesseract.

    Returns:
        True, если замена допустима по правилам модуля.
    """
    if not ds or not tess or len(ds) > MAX_SPAN or len(tess) > MAX_SPAN:
        return False
    kinds = [_kind(t) for t in ds]
    if not any(kinds):
        return _digit_split(ds, tess)
    # остальные слова расхождения должны совпасть (с точностью до знаков и тире) — русское не меняем
    rest_ds = [_norm(t).strip("-–—") for t, k in zip(ds, kinds) if not k]
    rest_tess = [_norm(w.text) for w in tess]
    if any(r and r not in rest_tess for r in rest_ds):
        return False
    tess_joined = "".join(w.text for w in tess)
    for t, k in zip(ds, kinds):
        core = _core(t)
        if k == "glued":
            # только поверх неуверенного слова и не двойником буквы («Bыбросы» поверх «выбросы»)
            if min(w.conf for w in tess) >= 60 or core.translate(_CYR_TWIN).lower() == _norm(tess_joined):
                return False
        elif k == "roman":
            looks = re.sub(r"[^IVXLC]", "", tess_joined.strip(_PUNCT).translate(_ROMAN_LOOKALIKE))
            if looks != core:
                return False
        elif k == "roman_range":
            # слово Tesseract — только из похожих на римские цифры знаков и не длиннее диапазона:
            # сколько I в «I–II» уцелело, по скану не проверить — верим DeepSeek
            body = tess_joined.strip(_PUNCT)
            if len(tess) > 2 or not body or any(c not in _ROMAN_RANGE_LOOKALIKE for c in body):
                return False
            if len(re.sub(r"[-–—]", "", body.translate(_ROMAN_LOOKALIKE))) > len(core) - 1:
                return False
        elif k == "latin":
            letters = re.sub(r"[^A-Za-z]", "", core)
            if len(letters) <= 1 and letters in _HOMOGLYPHS:
                return False
            if len(tess) > 2:
                return False
            tess_letters = re.sub(r"[^А-Яа-яЁё]", "", tess_joined).lower()
            if len(letters) == 1 and tess_letters == _TRANSLIT.get(letters.lower()):
                return False
    return True


def _span_words(tess: list[Word], ds: list[str]) -> list[Word]:
    """Создаёт слова DeepSeek на месте слов Tesseract, деля их общую рамку поровну."""
    x0, y0 = tess[0].box[0], min(w.box[1] for w in tess)
    x1, y1 = tess[-1].box[2], max(w.box[3] for w in tess)
    step = (x1 - x0) / len(ds)
    out = []
    for i, t in enumerate(ds):
        glued = _GLUED_RE.match(t)
        if glued:
            t = f"{glued[1]}{glued[2]}<sub>{glued[3]}</sub>{glued[4]}"
        rich = t if (_TAG_RE.search(t) or "\\(" in t) else ""
        # «P<sub>сд атм</sub>» — одно слово: пробел внутри неразрывный (слова строки — по пробелам)
        out.append(Word(plain(t).replace(" ", " "), 100.0, (x0 + i * step, y0, x0 + (i + 1) * step, y1),
                        restored=True, rich=rich))
    return out


def tokens(text: str) -> list[str]:
    """Делит текст блока DeepSeek на слова.

    Слово с разметкой берётся целиком, ``<br>`` считается пробелом, «##»
    заголовка Markdown словом не считается.
    """
    return [_markup(t) for t in _TOKEN_RE.findall(text.replace("<br>", " ")) if t.strip("#")]


def _join_fragmented_lines(lines: list[list[Word]], blocks) -> list[list[Word]]:
    """Соединяет фрагменты одной строки внутри подтверждённого текстового блока."""
    text_boxes = [box for label, box, _ in blocks if label not in ("table", "image", "equation")]
    groups: list[list[Word]] = []
    for line in lines:
        if not line:
            continue
        height = max(w.box[3] - w.box[1] for w in line)
        baseline = max(w.box[3] for w in line)
        joined = False
        for group in reversed(groups[-4:]):
            gh = max(w.box[3] - w.box[1] for w in group)
            gb = max(w.box[3] for w in group)
            left, right = sorted((group, line), key=lambda ws: min(w.box[0] for w in ws))
            gap = min(w.box[0] for w in right) - max(w.box[2] for w in left)
            if abs(baseline - gb) > 0.3 * max(height, gh) or not (0 <= gap <= 5 * max(height, gh)):
                continue
            all_words = group + line
            if not any(all(b[0] - 8 <= (w.box[0] + w.box[2]) / 2 <= b[2] + 8
                           and b[1] - 8 <= (w.box[1] + w.box[3]) / 2 <= b[3] + 8 for w in all_words)
                       for b in text_boxes):
                continue
            group.extend(line)
            group.sort(key=lambda w: w.box[0])
            joined = True
            break
        if not joined:
            groups.append(list(line))
    return groups


def reread_lines(lines: list[list[Word]], ds_raw: str | None, width: int, height: int
                 ) -> tuple[list[list[Word]], list[Formula]]:
    """Исправляет строки Tesseract по ответу DeepSeek-OCR на страницу.

    Args:
        lines: Строки слов Tesseract.
        ds_raw: Ответ DeepSeek-OCR на страницу или None.
        width: Ширина страницы, px.
        height: Высота страницы, px.

    Returns:
        Исправленные строки (строки на месте формул удалены) и формулы-блоки.
    """
    if not ds_raw:
        return lines, []
    blocks = ds_blocks(ds_raw, width, height)
    lines = _join_fragmented_lines(lines, blocks)
    formulas = []
    for label, box, text in blocks:
        if label == "equation":
            latex = text.strip().removeprefix("\\[").removesuffix("\\]").strip()
            if latex:
                formulas.append(Formula(latex, box))

    def in_formula(line: list[Word]) -> bool:
        """Центр строки лежит в рамке формулы (с запасом)."""
        cx = sum((w.box[0] + w.box[2]) / 2 for w in line) / len(line)
        cy = sum((w.box[1] + w.box[3]) / 2 for w in line) / len(line)
        return any(f.box[0] - FORMULA_MARGIN <= cx <= f.box[2] + FORMULA_MARGIN
                   and f.box[1] - FORMULA_MARGIN <= cy <= f.box[3] + FORMULA_MARGIN for f in formulas)

    lines = [line for line in lines if not in_formula(line)]
    replaced: dict[int, dict[int, tuple[int, list[Word]]]] = {}  # строка -> начало -> (длина, слова)
    for label, box, text in blocks:
        if label in ("table", "image", "equation"):
            continue
        ds = tokens(text)
        if not ds or _translit_block(ds):
            continue
        # слова строк Tesseract, лежащих в рамке блока, — по порядку
        refs = [(li, wi) for li, line in enumerate(lines)
                if box[1] - 8 <= (min(w.box[1] for w in line) + max(w.box[3] for w in line)) / 2 <= box[3] + 8
                and line[0].box[0] < box[2] and line[-1].box[2] > box[0]
                for wi in range(len(line))]
        if not refs:
            continue
        # Отдельный блок рукописной даты не имеет слова-якоря. Его можно
        # использовать только поверх повреждённой даты, в тех же координатах.
        date_text = text.strip()
        if re.fullmatch(r"\d{2}\.\d{2}\.\d{4}", date_text) and box[3] < 0.35 * height:
            local = [(li, wi) for li, wi in refs
                     if box[0] - 8 <= (lines[li][wi].box[0] + lines[li][wi].box[2]) / 2 <= box[2] + 8]
            if local and len({li for li, _ in local}) == 1:
                old = [lines[li][wi] for li, wi in local]
                raw_date = "".join(w.text for w in old)
                if not re.fullmatch(r"\d{2}\.\d{2}\.\d{4}", raw_date) and len(re.findall(r"\d", raw_date)) >= 4:
                    from datetime import datetime

                    try:
                        datetime.strptime(date_text, "%d.%m.%Y")
                    except ValueError:
                        pass
                    else:
                        li, wi = local[0]
                        replaced.setdefault(li, {})[wi] = (len(old), _span_words(old, [date_text]))
                        continue
        tess = [lines[li][wi] for li, wi in refs]
        sm = difflib.SequenceMatcher(None, [_norm(t) for t in ds], [_norm(w.text) for w in tess], autojunk=False)
        ops = sm.get_opcodes()
        for k, (op, a0, a1, b0, b1) in enumerate(ops):
            if op != "replace":
                continue
            anchored = (k > 0 and ops[k - 1][0] == "equal") or (k + 1 < len(ops) and ops[k + 1][0] == "equal")
            # «b)» в начале блока — пункт списка «б)»: маркеры в актах кириллические
            if a0 == 0 and _LIST_MARKER_RE.match(ds[0]):
                continue
            span = refs[b0:b1]
            # замена — в пределах одной строки Tesseract
            if not anchored or len({li for li, _ in span}) != 1 or not _accept(ds[a0:a1], tess[b0:b1]):
                continue
            li, wi = span[0]
            replaced.setdefault(li, {})[wi] = (len(span), _span_words(tess[b0:b1], ds[a0:a1]))
    out = []
    for li, line in enumerate(lines):
        if li not in replaced:
            out.append(line)
            continue
        new, wi = [], 0
        while wi < len(line):
            if wi in replaced[li]:
                n, words = replaced[li][wi]
                new.extend(words)
                wi += n
            else:
                new.append(line[wi])
                wi += 1
        out.append(new)
    return out, formulas
