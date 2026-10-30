"""Сохранение индексов в явно распознанных химических формулах."""

from __future__ import annotations

import re

_ELEMENTS = set("H He Li Be B C N O F Ne Na Mg Al Si P S Cl Ar K Ca Sc Ti V Cr Mn Fe Co Ni Cu Zn Ga Ge As Se Br Kr Rb Sr Y Zr Nb Mo Tc Ru Rh Pd Ag Cd In Sn Sb Te I Xe Cs Ba La Ce Pr Nd Pm Sm Eu Gd Tb Dy Ho Er Tm Yb Lu Hf Ta W Re Os Ir Pt Au Hg Tl Pb Bi Po At Rn Fr Ra Ac Th Pa U Np Pu Am Cm Bk Cf Es Fm Md No Lr Rf Db Sg Bh Hs Mt Ds Rg Cn Nh Fl Mc Lv Ts Og".split())
_FORMULA = re.compile(r"(?<![\w])(?:[A-Z][a-z]?\d*|[()])+\d*(?![\w])")
_SUB = str.maketrans("0123456789", "₀₁₂₃₄₅₆₇₈₉")
ROMAN_CODE = re.compile(r"[IVXLC]{1,6}[абвг]?(?:\s*\([^)]*\))?")
_LOOKALIKE_CODE = re.compile(r"[IVXLC1ПШУГТХЖl!|\[\]]{1,6}[абвгr6]?(?:\s*\([^)]*\))?")


def is_chemical_formula(text: str) -> bool:
    """Проверяет запись формулы по символам элементов без проверки её смысла."""
    token = text.translate(str.maketrans("₀₁₂₃₄₅₆₇₈₉", "0123456789")).replace(" ", "")
    symbols = re.findall(r"[A-Z][a-z]?", token)
    return bool(symbols and re.fullmatch(r"(?:[A-Z][a-z]?\d*|[()])+\d*", token)
                and all(s in _ELEMENTS for s in symbols)
                and not re.fullmatch(r"[IVXLC]+\d+", token)
                and token.count("(") == token.count(")"))


def doubtful_roman_code(text: str) -> bool:
    """Выделяет повреждённые обозначения категорий, не угадывая их значение."""
    text = text.strip()
    return bool(_LOOKALIKE_CODE.fullmatch(text) and not ROMAN_CODE.fullmatch(text)
                and not re.fullmatch(r"\d+", text))


def section_heading(text: str) -> str | None:
    """Нормализует похожие начертания римского номера перед названием раздела."""
    match = re.match(r"^([IVXLC1l|ГТПШУХ!]{1,6})[.:\s]+([А-ЯЁ].*)$", text.strip())
    if not match:
        return None
    if not re.match(r"[А-ЯЁ][а-яё]{2}",match[2]):
        return None
    roman = match[1].translate(str.maketrans({"1": "I", "l": "I", "|": "I", "!": "I",
        "Г": "I", "Т": "I", "П": "II", "Ш": "III", "У": "V", "Х": "X"}))
    if not re.fullmatch(r"X{0,3}(?:IX|IV|V?I{0,3})", roman) or not roman:
        return None
    return roman + ". " + match[2]


def chemical_indices(text: str) -> str:
    """Переводит цифры в индексы, не исправляя состав формулы по справочнику.

    Преобразуются только цельные записи с цифрами, составленные из символов
    химических элементов. Обозначения документов, CAS и обычный текст остаются
    неизменными. Необычная формула оригинала не заменяется более привычной.
    """
    compact = re.sub(r"\s+", "", text).translate(str.maketrans("₀₁₂₃₄₅₆₇₈₉", "0123456789"))
    if re.search(r"\d", compact) and is_chemical_formula(compact):
        return compact.translate(_SUB)

    def replace(match: re.Match) -> str:
        token = match.group()
        if not re.search(r"\d", token) or not is_chemical_formula(token):
            return token
        return token.translate(_SUB)

    return _FORMULA.sub(replace, text)
