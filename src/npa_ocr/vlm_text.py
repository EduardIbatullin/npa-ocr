"""Второе прочтение текста страницы ответом DeepSeek-OCR: римские цифры, латинские переменные,
индексы и степени, формулы. Tesseract с русским словарём их не знает и пишет похожими русскими
буквами: «II» → «П», «IV» → «ГУ», «Z» → «7», «V» → «У», «i-м» → «1-м», «16³» → «163»,
«K<sub>пр</sub>» → «Кир», формула «Z = S/V × 100%» → «7-5. х100%, у». DeepSeek-OCR (ответ на
страницу, уже полученный для таблиц) читает их верно: римские — латиницей, индексы — <sub>/<sup>,
формулы — LaTeX (блок equation, в тексте — \\(…\\)). Сверка по 179 страницам 14 документов —
npa-vlm-ocr formulas/: 79 римских, 24 индекса/степени, 12 формул; у Tesseract верных — ни одного.

Русский текст при этом остаётся за Tesseract: DeepSeek иногда сам портит русские слова
транслитом («окружающей» → «okpyxakouIeI»). Правила (`reread_lines`):
- слово DeepSeek заменяет слова Tesseract только внутри короткого расхождения, у которого с
  одной из сторон совпавшее слово (выравнивание по difflib на словах строк в рамке блока);
- блок DeepSeek с транслитом (латинское слово из 4+ букв на месте русского) не используется;
- римская цифра — только если слово Tesseract похоже на неё по начертанию («П» → II, «Ш» → III,
  «ГУ» → IV, «У» → V, «[», «|», «Г», «1» → I): «I. МОСКВА» у DeepSeek на месте «г.» не проходит;
- латинская буква-переменная — 1–3 буквы (с «-м», «-го» и т. п.), и не двойник русской буквы
  Tesseract (K/К, B/В, C/С — в русском тексте вернее Tesseract), не её транслит («b» на месте «б»)
  и не маркер пункта списка в начале блока («b)» — в актах пункты «а)», «б)»);
- индекс, степень, \\(формула\\) в тексте — разметкой DeepSeek (`Word.rich`, в DOCX — подстрочный
  и надстрочный шрифт, формула Word); индексы, записанные без разметки, переводятся в неё:
  символы Юникода «Sᵢᵏ», подчёркивание «C_t», латинская заглавная, слитая с русским индексом
  «Mфиатм» (последнее — только поверх неуверенного слова Tesseract и не двойником буквы);
- цифра, прочитанная Tesseract буквой или слитая со словом: «Злет» → «3 лет», «2013г.» →
  «2013 г.», «пункта3» → «пункта 3» — если у Tesseract то же после замены букв-двойников цифр, а
  DeepSeek делит только на стыке цифры и буквы или «№» («1316-р» не делится);
- блок equation заменяет строки Tesseract в своей рамке — формула Word (`Formula`).
Заменённое — жёлтое, как и прочие правки по второму прочтению."""

from __future__ import annotations

import difflib
import re
from dataclasses import dataclass

from npa_ocr.vlm_marks import Box, Word, ds_blocks

# Слово DeepSeek: \(LaTeX\) или слово с <sub>/<sup> целиком (внутри бывают пробелы:
# «P<sub>сд атм</sub>», «\(P_{\text{сд атм}}\)»), иначе — до пробела.
_TOKEN_RE = re.compile(r"\S*?\\\(.*?\\\)\S*|\S*?<su[bp]>.*?</su[bp]>(?:\S*?<su[bp]>.*?</su[bp]>)*\S*|\S+")
_TAG_RE = re.compile(r"</?su[bp]>")
_ROMAN_RE = re.compile(r"^[IVXLC]{1,6}$")
_LATIN_VAR_RE = re.compile(r"^[A-Za-z]{1,3}(-[а-яё]{1,3})?$")
_LIST_MARKER_RE = re.compile(r"^[a-z]\)$")
_SUP_DIGITS = str.maketrans("0123456789", "⁰¹²³⁴⁵⁶⁷⁸⁹")
_SUBSUP_CHARS = "¹²³⁴⁵⁶⁷⁸⁹⁰₀₁₂₃₄₅₆₇₈₉"
# латинские буквы, неотличимые от русских на скане: на них в русском тексте вернее Tesseract
_HOMOGLYPHS = set("ABCEHKMOPTXaceopxy")
# русская буква у Tesseract, транслитом которой DeepSeek пишет латинскую: «б)» → «b)» в списке
# (156 стр. 8), «и» → «i». «У» на месте V и «7» на месте Z — не транслит, там прав DeepSeek
_TRANSLIT = {"a": "а", "b": "б", "v": "в", "g": "г", "d": "д", "e": "е", "z": "з", "i": "и", "k": "к",
             "l": "л", "m": "м", "n": "н", "o": "о", "p": "п", "r": "р", "s": "с", "t": "т", "u": "у",
             "f": "ф", "h": "х", "c": "ц"}
# как Tesseract с русским словарём пишет римские цифры
_ROMAN_LOOKALIKE = str.maketrans({"П": "II", "Ш": "III", "У": "V", "Г": "I", "Т": "I", "[": "I", "]": "I",
                                  "|": "I", "1": "I", "l": "I", "!": "I", "Н": "II", "Х": "X", "\\": ""})
_PUNCT = ".,;:()«»\"'’"
# индексы, которые DeepSeek пишет символами Юникода («Sᵢᵏ», 710 стр. 12) — в разметку <sub>/<sup>
_UNI_SUB = dict(zip("ᵢⱼₐₑₒₓₕₖₗₘₙₚₛₜᵣᵤᵥ₀₁₂₃₄₅₆₇₈₉", "ijaeoxhklmnpstruv0123456789"))
_UNI_SUP = dict(zip("ⁱʲᵏᵐⁿᵗˡʰᵃᵇᶜᵈᵉᶠᵍᵒᵖʳˢᵘᵛʷˣʸᶻ", "ijkmntlhabcdefgoprsuvwxyz"))
_UNDERSCORE_RE = re.compile(r"^(\(?)([A-Za-z]{1,3})_([0-9A-Za-zА-Яа-яЁё]+)([.,;:)]*)$")  # «C_t» (710 стр. 13)
# латинская заглавная, слитая с русским индексом: «Mфиатм» (710 стр. 10, на скане M с индексом «фi атм»)
_GLUED_RE = re.compile(r"^(\(?)([A-Z])([а-яё]{2,})([.,;:)]*)$")
# цифры, которые Tesseract читает буквами: «Злет» на месте «3 лет» (710 стр. 3)
_DIGIT_LOOKALIKE = str.maketrans({"З": "3", "з": "3", "О": "0", "о": "0", "б": "6", "l": "1", "I": "1", "|": "1"})
_CYR_TWIN = str.maketrans("ABCEHKMOPTXaceopxy", "АВСЕНКМОРТХасеорху")
MAX_SPAN = 4  # длина расхождения (слов), в котором DeepSeek может заменить слова Tesseract
FORMULA_MARGIN = 10  # пикселей: строки Tesseract у рамки формулы — тоже её мусор


@dataclass(frozen=True)
class Formula:
    latex: str  # без \[ \]
    box: Box


def plain(token: str) -> str:
    """Разметка DeepSeek → простой текст (для text блока: колонтитулы, поиск): «K<sub>пр</sub>» →
    «Kпр», «16<sup>3</sup>» → «16³», «\\(V_i\\)» → «V_i»."""
    token = re.sub(r"<sup>(\d+)</sup>", lambda m: m.group(1).translate(_SUP_DIGITS), token)
    token = _TAG_RE.sub("", token)
    token = re.sub(r"\\\((.*?)\\\)", r"\1", token)
    token = re.sub(r"\\text\{(.*?)\}", r"\1", token)
    return token.replace("{", "").replace("}", "").replace("\\", "")


def _norm(token: str) -> str:
    return plain(token).strip(_PUNCT).lower()


def _core(token: str) -> str:
    return plain(token).strip(_PUNCT)


def _markup(token: str) -> str:
    """Индексы, записанные DeepSeek без разметки, — в <sub>/<sup>: «Sᵢᵏ» → «S<sub>i</sub><sup>k</sup>»,
    «C_t» → «C<sub>t</sub>»."""
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
    if "\\(" in token:
        return "latex"
    if "<sub>" in token or "<sup>" in token or any(c in token for c in _SUBSUP_CHARS):
        return "index"
    if _GLUED_RE.match(token):
        return "glued"
    core = _core(token)
    if _ROMAN_RE.match(core):
        return "roman"
    if _LATIN_VAR_RE.match(core):
        return "latin"
    return None


def _digit_split(ds: list[str], tess: list[Word]) -> bool:
    """«3 лет» у DeepSeek, «Злет» у Tesseract: то же самое, если у Tesseract буквы-двойники цифр
    заменить цифрами; так же «2013г.» → «2013 г.», «пункта3» → «пункта 3». Меняется только цифра
    или пробел на стыке цифры и буквы — русские слова те же («1316-р» не трогаем)."""
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
    """Блок, где DeepSeek написал русские слова латиницей («oJJaCTH»): ему не верим."""
    return any(len(re.sub(r"[^A-Za-z]", "", t)) >= 4 and not re.search(r"[А-Яа-яЁё\\<]", t) for t in tokens)


def _accept(ds: list[str], tess: list[Word]) -> bool:
    """Можно ли заменить слова Tesseract tess словами DeepSeek ds (одно расхождение)."""
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
            # «Mфиатм» поверх неуверенного «Мам», но не «Bыбросы» поверх «выбросы» (двойник буквы)
            if min(w.conf for w in tess) >= 60 or core.translate(_CYR_TWIN).lower() == _norm(tess_joined):
                return False
        elif k == "roman":
            looks = re.sub(r"[^IVXLC]", "", tess_joined.strip(_PUNCT).translate(_ROMAN_LOOKALIKE))
            if looks != core:
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
    """Слова DeepSeek на месте слов Tesseract: рамки — по рамкам заменённых слов."""
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
    return [_markup(t) for t in _TOKEN_RE.findall(text.replace("<br>", " "))]


def reread_lines(lines: list[list[Word]], ds_raw: str | None, width: int, height: int
                 ) -> tuple[list[list[Word]], list[Formula]]:
    """Строки Tesseract с правками по DeepSeek и формулы-блоки (строки на их месте выкинуты)."""
    if not ds_raw:
        return lines, []
    blocks = ds_blocks(ds_raw, width, height)
    formulas = []
    for label, box, text in blocks:
        if label == "equation":
            latex = text.strip().removeprefix("\\[").removesuffix("\\]").strip()
            if latex:
                formulas.append(Formula(latex, box))

    def in_formula(line: list[Word]) -> bool:
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
        tess = [lines[li][wi] for li, wi in refs]
        sm = difflib.SequenceMatcher(None, [_norm(t) for t in ds], [_norm(w.text) for w in tess], autojunk=False)
        ops = sm.get_opcodes()
        for k, (op, a0, a1, b0, b1) in enumerate(ops):
            if op != "replace":
                continue
            anchored = (k > 0 and ops[k - 1][0] == "equal") or (k + 1 < len(ops) and ops[k + 1][0] == "equal")
            # «b)» в начале блока — пункт списка «б)» (156 стр. 8): маркеры в актах кириллические
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
