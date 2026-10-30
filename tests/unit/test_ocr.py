"""Тесты точки входа OCR → DOCX (гибридная схема, см. docs/history/plan.md и
модуль npa_ocr.ocr). Движки — фейковые функции на реальных фикстурах/данных,
без загрузки тяжёлых моделей."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from PIL import Image

from npa_ocr.mark_detection import Mark
from npa_ocr.ocr import (
    UNRECOGNIZABLE_MARK,
    OcrDocumentResult,
    PageResult,
    TableBlock,
    TextBlock,
    build_docx,
    expected_signatory_title,
    ocr_document,
    ocr_page,
    run_paddle_tables,
    run_tesseract_text,
)
from npa_ocr.ocr import _recover_bottom_gap_text
from npa_ocr.ocr import (
    _best_img2table_match,
    _looks_like_fragment_continuation,
    _merge_img2table_fragment_rows,
    _table_cells_from_img2table,
)
from npa_ocr.ocr_table import TableCell

FIXTURES = Path(__file__).parent / "fixtures" / "ocr_tables"


def _load_simple_table_json() -> dict:
    """Фикстура в fixtures/ocr_tables/ хранится в "плоском" виде (для прямых тестов
    reconstruct_table в test_ocr_table.py) — rec_boxes/rec_texts не вложены в
    table_ocr_pred, как в реальном JSON движка. Здесь оборачиваем обратно в форму,
    которую реально отдаёт PPStructureV3 (см. _extract_table_blocks)."""
    with open(FIXTURES / "simple_table_0.json", encoding="utf-8") as f:
        flat = json.load(f)
    nested = {
        "cell_box_list": flat["cell_box_list"],
        "table_ocr_pred": {
            "rec_boxes": flat["rec_boxes"],
            "rec_texts": flat["rec_texts"],
        },
    }
    return {"table_res_list": [nested]}


def _tesseract_data(words: list[tuple[str, int, int, int, int, int]]) -> dict:
    """words — (text, conf, left, top, width, height); остальные поля Tesseract
    (block/par/line_num) проставляются по порядку — одна строка на все слова,
    если явно не указано иное, этого достаточно для тестов группировки."""
    data = {
        "text": [],
        "conf": [],
        "left": [],
        "top": [],
        "width": [],
        "height": [],
        "block_num": [],
        "par_num": [],
        "line_num": [],
    }
    for text, conf, left, top, width, height in words:
        data["text"].append(text)
        data["conf"].append(conf)
        data["left"].append(left)
        data["top"].append(top)
        data["width"].append(width)
        data["height"].append(height)
        data["block_num"].append(1)
        data["par_num"].append(1)
        data["line_num"].append(1)
    return data


def _tesseract_data_multiline(words: list[tuple[str, int, int, int, int, int, int]]) -> dict:
    """Как _tesseract_data, но с явным line_num (7-й элемент каждого тьюпла) —
    нужно воспроизвести реальный случай, когда Tesseract разбивает одну визуальную
    строку на несколько геометрических из-за помехи печати."""
    data = {
        "text": [],
        "conf": [],
        "left": [],
        "top": [],
        "width": [],
        "height": [],
        "block_num": [],
        "par_num": [],
        "line_num": [],
    }
    for text, conf, left, top, width, height, line_num in words:
        data["text"].append(text)
        data["conf"].append(conf)
        data["left"].append(left)
        data["top"].append(top)
        data["width"].append(width)
        data["height"].append(height)
        data["block_num"].append(1)
        data["par_num"].append(1)
        data["line_num"].append(line_num)
    return data


def test_run_paddle_tables_real_fixture():
    fixture = _load_simple_table_json()
    tables = run_paddle_tables(Path("unused.png"), lambda _p: fixture)
    assert len(tables) == 1
    t = tables[0]
    assert t.n_rows == 7
    assert t.n_cols == 3
    header = [c for c in t.cells if c.row == 0]
    assert [c.text for c in header] == ["Год", "Месяц", "Значения коэффициентов"]


def test_table_regions_for_page_skips_paddle_when_tatr_finds_nothing(monkeypatch):
    """Страница без таблиц не должна платить цену полной загрузки PPStructureV3
    (~11 тяжёлых моделей PaddleX) — см. npa_ocr.ocr.table_regions_for_page.
    Раньше PaddleOCR стоял первым в `or`, поэтому вызывался всегда; это и
    привело к OOM на страницах без единой таблицы (найдено на реальном
    прогоне, контейнер 12 ГБ падал ещё до начала распознавания текста)."""
    from PIL import Image

    import npa_ocr.ocr as ocr_module

    monkeypatch.setattr(ocr_module.table_structure, "detect_table_regions", lambda _img: [])

    def _should_not_be_called(_path):
        raise AssertionError("PaddleOCR (PPStructureV3) не должен вызываться — TATR не нашёл таблиц")

    img = Image.new("RGB", (10, 10), "white")
    assert ocr_module.table_regions_for_page(img, _should_not_be_called) == []


def test_table_regions_for_page_calls_paddle_when_tatr_finds_candidate(monkeypatch):
    """Когда TATR нашёл область-кандидат, PaddleOCR всё равно вызывается (даёт
    точные отдельные области на страницах с несколькими таблицами) — порядок
    поменялся, но итоговое поведение на реальных страницах с таблицами нет."""
    from PIL import Image

    import npa_ocr.ocr as ocr_module

    monkeypatch.setattr(ocr_module, "_paddle_runtime_usable", lambda: True)
    monkeypatch.setattr(
        ocr_module.table_structure, "detect_table_regions", lambda _img: [(0.0, 0.0, 10.0, 10.0)]
    )
    calls = {"n": 0}

    def fake_predict(_path):
        calls["n"] += 1
        return {"table_res_list": [{"cell_box_list": [(1, 1, 2, 2), (3, 3, 4, 4)]}]}

    img = Image.new("RGB", (10, 10), "white")
    found = ocr_module.table_regions_for_page(img, fake_predict)
    assert calls["n"] == 1
    assert found == [(1, 1, 4, 4)]


def test_table_regions_for_page_falls_back_to_tatr_when_paddle_disagrees(monkeypatch):
    """TATR нашёл кандидата, но PaddleOCR не подтвердил (вернул пусто) — область
    TATR используется как резерв, а не отбрасывается совсем."""
    from PIL import Image

    import npa_ocr.ocr as ocr_module

    monkeypatch.setattr(ocr_module, "_paddle_runtime_usable", lambda: True)
    tatr_box = (5.0, 5.0, 50.0, 50.0)
    monkeypatch.setattr(ocr_module.table_structure, "detect_table_regions", lambda _img: [tatr_box])

    def fake_predict_empty(_path):
        return {"table_res_list": []}

    img = Image.new("RGB", (10, 10), "white")
    assert ocr_module.table_regions_for_page(img, fake_predict_empty) == [tatr_box]


def test_table_regions_for_page_skips_paddle_when_runtime_unusable(monkeypatch):
    """На CPU без AVX2 import paddle даёт SIGILL — PaddleOCR не вызываем, берём TATR."""
    from PIL import Image

    import npa_ocr.ocr as ocr_module

    tatr_box = (1.0, 2.0, 30.0, 40.0)
    monkeypatch.setattr(ocr_module, "_paddle_runtime_usable", lambda: False)
    monkeypatch.setattr(ocr_module.table_structure, "detect_table_regions", lambda _img: [tatr_box])

    def _should_not_be_called(_path):
        raise AssertionError("PaddleOCR не должен вызываться — runtime недоступен")

    img = Image.new("RGB", (10, 10), "white")
    assert ocr_module.table_regions_for_page(img, _should_not_be_called) == [tatr_box]


def test_run_tesseract_text_marks_low_confidence_and_skips_table_area():
    # Взяты реальные порядки величин confidence из эксперимента со штампом
    # (докс.stage3-plan.md): чистый текст 71-97, повреждённый печатью 0-55.
    data = _tesseract_data(
        [
            ("Председатель", 96, 10, 10, 100, 20),
            ("Правительства", 1, 120, 10, 100, 20),  # под печатью
            ("Д.Медведев", 95, 230, 10, 100, 20),
            ("внутри_таблицы", 90, 10, 100, 50, 20),  # должно быть пропущено
        ]
    )
    table_bboxes = [(0.0, 90.0, 200.0, 130.0)]
    blocks = run_tesseract_text(
        Path("unused.png"), table_bboxes, image_data_fn=lambda _p: data, confidence_threshold=60
    )
    assert len(blocks) == 1
    assert blocks[0].text == "Председатель Правительства Д.Медведев"
    assert blocks[0].low_words == (False, True, False)


def test_expected_signatory_title_known_authorities():
    assert (
        expected_signatory_title("Правительство Российской Федерации")
        == "Председатель Правительства Российской Федерации"
    )
    assert (
        expected_signatory_title("Президент Российской Федерации")
        == "Президент Российской Федерации"
    )
    assert expected_signatory_title("Федеральная служба по аккредитации") == "Руководитель"
    # Министерства пока не поддержаны — склонение названия ведомства не выводится
    # надёжно без риска угадывания (см. docs/history/plan.md).
    assert expected_signatory_title("Министерство природных ресурсов") is None
    assert expected_signatory_title("Неизвестный орган") is None


def test_run_tesseract_text_restores_damaged_title_formula_without_touching_name():
    # Реальный кейс из эксперимента со штампом: формула должности и ФИО
    # подписанта стоят на одной строке ("Председатель Правительства Российской
    # Федерации Д.Медведев"), "Правительства" повреждено печатью — должна
    # восстановиться только формула, ФИО не трогается.
    data = _tesseract_data(
        [
            ("Председатель", 96, 10, 10, 100, 20),
            ("Правительства", 1, 120, 10, 100, 20),  # под печатью
            ("Российской", 90, 230, 10, 100, 20),
            ("Федерации", 93, 340, 10, 100, 20),
            ("Д.Медведев", 95, 450, 10, 100, 20),
        ]
    )
    blocks = run_tesseract_text(
        Path("unused.png"),
        [],
        image_data_fn=lambda _p: data,
        confidence_threshold=60,
        authority_name="Правительство Российской Федерации",
    )
    assert len(blocks) == 1
    assert blocks[0].text == "Председатель Правительства Российской Федерации Д.Медведев"


def test_run_tesseract_text_restores_title_split_across_tesseract_lines_near_stamp():
    # Реальный случай (живой тест, 12-стр. документ): печать сдвинула геометрию
    # слов так, что Tesseract разбил формулу должности на ДВЕ отдельные строки —
    # "Председатель [нрзб.]" / "Российской [нрзб.] Д.Медведев". Восстановление
    # должно сработать сквозным поиском по словам страницы, не только внутри
    # одной уже сгруппированной строки.
    data = _tesseract_data_multiline(
        [
            ("Председатель", 96, 10, 100, 100, 20, 1),
            ("Правительства", 1, 120, 100, 100, 20, 1),  # под печатью, строка 1
            ("Российской", 90, 10, 130, 100, 20, 2),
            ("Федерации", 1, 120, 130, 100, 20, 2),  # под печатью, строка 2
            ("Д.Медведев", 95, 230, 130, 100, 20, 2),
        ]
    )
    blocks = run_tesseract_text(
        Path("unused.png"),
        [],
        image_data_fn=lambda _p: data,
        confidence_threshold=60,
        authority_name="Правительство Российской Федерации",
    )
    assert len(blocks) == 2
    assert blocks[0].text == "Председатель Правительства"
    assert blocks[1].text == "Российской Федерации Д.Медведев"


def test_run_tesseract_text_fixes_high_confidence_misread_of_title_word():
    # Реальный случай (калибровка на 4 документах, 2026-10-02): печать на
    # «Федеральный закон №310-ФЗ» исказила «Президент» до «Ярезидент», но
    # Tesseract дал этому слову confidence=73 — выше порога 60, поэтому оно
    # НЕ было замаскировано как [нрзб.]. Восстановление должно сработать и на
    # уже "чистом" (по confidence), но текстово похожем на ожидаемое слове.
    data = _tesseract_data(
        [
            ("Ярезидент", 73, 10, 10, 150, 20),
            ("Российской", 95, 170, 10, 150, 20),
            ("Федерации", 96, 330, 10, 150, 20),
            ("В.Путин", 62, 490, 10, 150, 20),
        ]
    )
    blocks = run_tesseract_text(
        Path("unused.png"),
        [],
        image_data_fn=lambda _p: data,
        confidence_threshold=60,
        authority_name="Президент Российской Федерации",
    )
    assert len(blocks) == 1
    assert blocks[0].text == "Президент Российской Федерации В.Путин"


def test_run_tesseract_text_fixes_truncated_title_words_across_lines():
    # Реальный случай (Постановление №1430, 2026-10-02): печать обрезала слова
    # формулы ("Правительства"→"Прави", "Федерации"→"Федей"), конфиденс не
    # замаскировал их, полное нечёткое сравнение тоже не ловило (разная длина
    # режет ratio до 0.56/0.57) — добавлено отдельное префиксное сравнение.
    # Формула к тому же снова разорвана на две строки, как и в более раннем
    # реальном случае ("Председатель Прави" / "Российской Федей М.Мишустин").
    data = _tesseract_data_multiline(
        [
            ("Председатель", 95, 10, 100, 150, 20, 1),
            ("Прави", 65, 170, 100, 80, 20, 1),  # искажено печатью, но выше порога 60
            ("Российской", 90, 10, 130, 150, 20, 2),
            ("Федей", 68, 170, 130, 80, 20, 2),  # искажено печатью, но выше порога 60
            ("М.Мишустин", 93, 260, 130, 150, 20, 2),
        ]
    )
    blocks = run_tesseract_text(
        Path("unused.png"),
        [],
        image_data_fn=lambda _p: data,
        confidence_threshold=60,
        authority_name="Правительство Российской Федерации",
    )
    assert len(blocks) == 2
    assert blocks[0].text == "Председатель Правительства"
    assert blocks[1].text == "Российской Федерации М.Мишустин"


def test_run_tesseract_text_does_not_corrupt_unrelated_mentions_of_russian_federation():
    # "Российской Федерации" — обычная фраза, многократно встречающаяся в теле
    # любого НПА вне подписи. Слово перед ней не похоже на "Президент"/
    # "Председатель" — окно должно быть отклонено целиком, без порчи текста.
    data = _tesseract_data(
        [
            ("в", 95, 10, 10, 30, 20),
            ("соответствии", 95, 50, 10, 150, 20),
            ("с", 95, 210, 10, 30, 20),
            ("законодательством", 95, 250, 10, 220, 20),
            ("Российской", 95, 480, 10, 150, 20),
            ("Федерации", 95, 640, 10, 150, 20),
        ]
    )
    blocks = run_tesseract_text(
        Path("unused.png"),
        [],
        image_data_fn=lambda _p: data,
        confidence_threshold=60,
        authority_name="Президент Российской Федерации",
    )
    assert len(blocks) == 1
    assert blocks[0].text == "в соответствии с законодательством Российской Федерации"


def test_run_tesseract_text_does_not_touch_clean_lines_or_unrelated_damage():
    data = _tesseract_data(
        [
            ("Обычный", 95, 10, 10, 100, 20),
            ("абзац", 1, 120, 10, 100, 20),  # повреждено, но не формула должности
        ]
    )
    blocks = run_tesseract_text(
        Path("unused.png"),
        [],
        image_data_fn=lambda _p: data,
        confidence_threshold=60,
        authority_name="Правительство Российской Федерации",
    )
    assert blocks[0].text == "Обычный абзац"
    assert blocks[0].low_words == (False, True)


def test_run_tesseract_text_applies_known_ocr_corrections_like_table_text():
    # normalize_header_text (известные искажения OCR — смесь латиницы/кириллицы,
    # «Γ.» вместо «г.») раньше применялся только к тексту таблиц; те же классы
    # ошибок встречаются и в обычном тексте страницы (см. docs/history/plan.md, 2026-10-24).
    data = _tesseract_data(
        [
            ("Cpoк", 95, 10, 10, 100, 20),  # «Cpoк» -> «Срок», тот же словарь, что и у таблиц
            ("действия", 95, 120, 10, 100, 20),
            ("до", 95, 230, 10, 50, 20),
            ("2024", 95, 290, 10, 60, 20),
            ("Γ.", 95, 360, 10, 40, 20),  # «Γ.» (греческая гамма) -> «г.»
        ]
    )
    blocks = run_tesseract_text(Path("unused.png"), [], image_data_fn=lambda _p: data, confidence_threshold=60)
    assert blocks[0].text == "Срок действия до 2024 г."


def test_run_tesseract_text_without_authority_name_leaves_mark_as_is():
    data = _tesseract_data(
        [
            ("Председатель", 96, 10, 10, 100, 20),
            ("Правительства", 1, 120, 10, 100, 20),
        ]
    )
    blocks = run_tesseract_text(
        Path("unused.png"), [], image_data_fn=lambda _p: data, confidence_threshold=60
    )
    assert blocks[0].text == "Председатель Правительства"
    assert blocks[0].low_words == (False, True)


def _full_page_data(words: list[tuple[str, int, int, int, int, int]]) -> dict:
    """Как _tesseract_data, но с полным набором ключей pytesseract.image_to_data
    (level/page_num/word_num) — нужен для _recover_bottom_gap_text, которая
    дописывает новые слова сразу во все массивы словаря."""
    data: dict[str, list] = {
        "level": [],
        "page_num": [],
        "block_num": [],
        "par_num": [],
        "line_num": [],
        "word_num": [],
        "left": [],
        "top": [],
        "width": [],
        "height": [],
        "conf": [],
        "text": [],
    }
    for text, conf, left, top, width, height in words:
        data["level"].append(5)
        data["page_num"].append(1)
        data["block_num"].append(1)
        data["par_num"].append(1)
        data["line_num"].append(1)
        data["word_num"].append(1)
        data["left"].append(left)
        data["top"].append(top)
        data["width"].append(width)
        data["height"].append(height)
        data["conf"].append(conf)
        data["text"].append(text)
    return data


def test_recover_bottom_gap_text_adds_confident_word_from_bottom_gap(monkeypatch):
    # Основной проход нашёл текст только до y=100 из 1000 — разрыв 90% высоты,
    # явно больше порога _BOTTOM_GAP_RATIO. Повторный проход (замоканный)
    # находит уверенное слово — должно быть дописано со сдвигом Y на отступ кропа
    # и с block_num, не совпадающим с уже существующими (см. docs/history/plan.md,
    # «Министр ... Козлов», Приказ №215).
    import pytesseract

    data = _full_page_data([("Пункт", 96, 10, 80, 60, 20)])
    extra = _full_page_data([("Министр", 96, 10, 5, 90, 20), ("x", 10, 110, 5, 10, 20)])
    monkeypatch.setattr(pytesseract, "image_to_data", lambda *a, **k: extra)

    img = Image.new("L", (1000, 1000), color=255)
    _recover_bottom_gap_text(img, data)

    assert "Министр" in data["text"]
    idx = data["text"].index("Министр")
    assert data["conf"][idx] == 96
    assert data["top"][idx] > 100  # смещён на высоту уже обработанной части страницы
    assert data["block_num"][idx] != 1  # не склеится в чужую строку по совпадению ключа


def test_recover_bottom_gap_text_ignores_noise_without_confident_word(monkeypatch):
    # Повторный проход нашёл только мусор: короткий символ без буквенного
    # содержимого ("—", росчерк подписи) и слово с низкой уверенностью — ни то,
    # ни другое не проходит фильтр, данные не меняются (иначе почти на каждой
    # странице с таблицей у нижнего поля появлялись бы лишние [нрзб.]).
    import pytesseract

    data = _full_page_data([("Пункт", 96, 10, 80, 60, 20)])
    extra = _full_page_data([("—", 70, 10, 5, 20, 20), ("АБ", 40, 50, 5, 30, 20)])
    monkeypatch.setattr(pytesseract, "image_to_data", lambda *a, **k: extra)
    n_before = len(data["text"])

    img = Image.new("L", (1000, 1000), color=255)
    _recover_bottom_gap_text(img, data)

    assert len(data["text"]) == n_before


def test_recover_bottom_gap_text_skips_when_gap_is_small(monkeypatch):
    # Основной проход уже захватил текст почти до низа страницы — повторный
    # проход не должен даже запускаться.
    import pytesseract

    def _should_not_be_called(*_a, **_k):
        raise AssertionError("не должен вызываться при маленьком разрыве")

    monkeypatch.setattr(pytesseract, "image_to_data", _should_not_be_called)
    data = _full_page_data([("Пункт", 96, 10, 950, 60, 20)])
    img = Image.new("L", (1000, 1000), color=255)
    _recover_bottom_gap_text(img, data)
    assert data["text"] == ["Пункт"]


def test_run_tesseract_text_without_detect_marks_fn_is_unaffected():
    # Без detect_marks_fn (None, значение по умолчанию) поведение не должно
    # отличаться от версии до появления ИИ-эскалации — регрессионная проверка.
    data = _tesseract_data(
        [
            ("Председатель", 96, 10, 10, 100, 20),
            ("Правительства", 1, 120, 10, 100, 20),
        ]
    )
    blocks = run_tesseract_text(Path("unused.png"), [], image_data_fn=lambda _p: data, confidence_threshold=60)
    assert blocks[0].text == "Председатель Правительства"
    assert blocks[0].low_words == (False, True)


def test_run_tesseract_text_escalates_low_confidence_words_under_covering_mark(tmp_path: Path):
    # Сценарий из docs/history/plan.md: эвристика нашла кандидата (скопление
    # низкоуверенных слов), detect_marks_fn (фейк) подтверждает — это печать,
    # перекрывающая текст документа; recover_covered_text_fn (фейк) отдаёт
    # восстановленный текст. Результат остаётся low=True — проверять руками.
    image_path = tmp_path / "page.png"
    Image.new("RGB", (400, 100), "white").save(image_path)
    data = _tesseract_data(
        [
            ("Председатель", 96, 10, 10, 100, 20),
            ("Прав", 1, 120, 10, 40, 20),  # под печатью, обрезано
            ("Д.Медведев", 95, 230, 10, 100, 20),
        ]
    )

    def fake_detect(_image_path: Path, low_word_boxes: list) -> list[Mark]:
        assert low_word_boxes == [(120.0, 10.0, 160.0, 30.0)]
        return [Mark(bbox=(115.0, 5.0, 165.0, 35.0), mark_type="печать", covers_text=True)]

    def fake_recover(_crop_path: Path) -> str:
        return "Правительства"

    blocks = run_tesseract_text(
        image_path,
        [],
        image_data_fn=lambda _p: data,
        confidence_threshold=60,
        detect_marks_fn=fake_detect,
        recover_covered_text_fn=fake_recover,
    )
    assert blocks[0].text == "Председатель Правительства Д.Медведев"
    assert blocks[0].low_words == (False, True, False)


def test_run_tesseract_text_rejects_escalation_answer_shorter_than_low_word_count(tmp_path: Path):
    # Реальный дефект, Приказ №215, стр. 1 (docs/history/plan.md, «штампы просачиваются
    # в текст»): bbox метки, найденный по скоплению низкоуверенных слов,
    # захватил не только штамп, но и край настоящей строки шапки документа
    # («И ЭКОЛОГИИ» в «МИНИСТЕРСТВО ПРИРОДНЫХ РЕСУРСОВ И ЭКОЛОГИИ») — на эту
    # область (с двумя низкоуверенными словами) VLM вернула один короткий
    # неверный ответ «на», которым в итоге была бы заменена часть настоящего
    # текста. Признак неудачного распознавания — ответ короче, чем слов нужно
    # заполнить, — для такой метки замена не должна применяться вовсе, слова
    # остаются как были. Другая метка в том же вызове (с одним низкоуверенным
    # словом и ровно одним восстановленным) — обычный штатный случай, должна
    # продолжать работать как раньше (регрессия не должна сломаться вместе с
    # защитой).
    image_path = tmp_path / "page.png"
    Image.new("RGB", (400, 300), "white").save(image_path)
    data = _tesseract_data_multiline(
        [
            ("МИНИСТЕРСТВО", 95, 10, 10, 200, 20, 1),
            ("И", 10, 220, 10, 20, 20, 1),
            ("ЭКОЛОГИИ", 10, 250, 10, 100, 20, 1),
            ("ЗАРЕГИСТРИРОВАНО", 5, 10, 200, 150, 30, 2),
        ]
    )

    underfilled_mark = Mark(bbox=(215.0, 5.0, 360.0, 35.0), mark_type="штамп_регистрации", covers_text=True)
    ok_mark = Mark(bbox=(0.0, 190.0, 200.0, 240.0), mark_type="штамп_регистрации", covers_text=True)

    def fake_detect(_image_path: Path, _low_word_boxes: list) -> list[Mark]:
        return [underfilled_mark, ok_mark]

    recovered_answers = iter(["на", "ЗАРЕГИСТРИРОВАНО"])

    def fake_recover(_crop_path: Path) -> str:
        return next(recovered_answers)

    blocks = run_tesseract_text(
        image_path,
        [],
        image_data_fn=lambda _p: data,
        confidence_threshold=60,
        detect_marks_fn=fake_detect,
        recover_covered_text_fn=fake_recover,
    )
    assert blocks[0].text == "МИНИСТЕРСТВО И ЭКОЛОГИИ"
    assert blocks[0].low_words == (False, True, True)
    assert blocks[1].text == "ЗАРЕГИСТРИРОВАНО"
    assert blocks[1].low_words == (True,)


def test_run_tesseract_text_drops_words_inside_non_covering_e_signature_mark(tmp_path: Path):
    # Пример Постановление №710 (docs/history/plan.md): штамп электронной подписи внизу
    # страницы — чистый визуальный мусор, не текст документа, не перекрывает
    # ничего содержательного. Слова внутри такой метки исключаются целиком, а
    # не просто выделяются как низкая уверенность.
    data = _tesseract_data(
        [
            ("Обычный", 96, 10, 10, 80, 20),
            ("текст", 95, 100, 10, 60, 20),
            ("abcDEF123", 5, 10, 200, 150, 30),  # штамп э-подписи, отдельная строка
        ]
    )
    data["block_num"][-1] = 2
    data["par_num"][-1] = 2
    data["line_num"][-1] = 2

    def fake_detect(_image_path: Path, low_word_boxes: list) -> list[Mark]:
        return [Mark(bbox=(0.0, 190.0, 200.0, 240.0), mark_type="электронная_подпись", covers_text=False)]

    blocks = run_tesseract_text(
        Path("unused.png"),
        [],
        image_data_fn=lambda _p: data,
        confidence_threshold=60,
        detect_marks_fn=fake_detect,
        recover_covered_text_fn=None,
    )
    assert len(blocks) == 1
    assert blocks[0].text == "Обычный текст"


def test_run_tesseract_text_drops_words_inside_non_covering_registration_stamp(tmp_path: Path):
    # Решение пользователя (2026-10-21): штамп регистрации (Минюст и т.п.) —
    # тоже служебная отметка, не текст документа, выкидывается целиком, если
    # сам по себе ничего содержательного не перекрывает — та же логика, что
    # уже была у электронной подписи.
    data = _tesseract_data(
        [
            ("Обычный", 96, 10, 10, 80, 20),
            ("текст", 95, 100, 10, 60, 20),
            ("ЗАРЕГИСТРИРОВАНО", 5, 10, 200, 150, 30),
        ]
    )
    data["block_num"][-1] = 2
    data["par_num"][-1] = 2
    data["line_num"][-1] = 2

    def fake_detect(_image_path: Path, low_word_boxes: list) -> list[Mark]:
        return [Mark(bbox=(0.0, 190.0, 200.0, 240.0), mark_type="штамп_регистрации", covers_text=False)]

    blocks = run_tesseract_text(
        Path("unused.png"),
        [],
        image_data_fn=lambda _p: data,
        confidence_threshold=60,
        detect_marks_fn=fake_detect,
        recover_covered_text_fn=None,
    )
    assert len(blocks) == 1
    assert blocks[0].text == "Обычный текст"


def test_ocr_page_orders_tables_and_text_by_vertical_position():
    fixture = _load_simple_table_json()
    text_data = _tesseract_data([("Заголовок", 95, 10, 5, 100, 20)])

    blocks = ocr_page(
        Path("unused.png"),
        predict_tables_fn=lambda _p: fixture,
        image_data_fn=lambda _p: text_data,
    )
    assert len(blocks) == 2
    assert isinstance(blocks[0], TextBlock)  # y=5, выше таблицы (y~1412 в фикстуре)
    assert isinstance(blocks[1], TableBlock)


def test_build_docx_writes_text_and_merged_table(tmp_path: Path):
    from docx import Document

    cells = [
        TableCell(row=0, col=0, row_span=1, col_span=2, text="Заголовок на всю ширину"),
        TableCell(row=1, col=0, row_span=1, col_span=1, text="a"),
        TableCell(row=1, col=1, row_span=1, col_span=1, text="b"),
    ]
    pages = [
        PageResult(page_number=1, ocr_applied=False, plain_text="Обычный текстовый слой"),
        PageResult(
            page_number=2,
            ocr_applied=True,
            blocks=[
                TextBlock(y=0, text="Распознанный абзац"),
                TableBlock(y=10, bbox=(0, 10, 100, 100), cells=cells, n_rows=2, n_cols=2),
            ],
        ),
    ]
    out = tmp_path / "out.docx"
    build_docx(pages, out)
    assert out.is_file()

    doc = Document(str(out))
    texts = [p.text for p in doc.paragraphs]
    assert "Обычный текстовый слой" in texts
    assert "Распознанный абзац" in texts
    assert len(doc.tables) == 1
    table = doc.tables[0]
    assert table.cell(0, 0).text == "Заголовок на всю ширину"
    # Объединение по горизонтали — (0,0) и (0,1) должны указывать на одну физическую ячейку.
    assert table.cell(0, 0)._tc is table.cell(0, 1)._tc
    assert table.cell(1, 0).text == "a"
    assert table.cell(1, 1).text == "b"


def test_build_docx_survives_non_rectangular_merge_conflict(tmp_path: Path, monkeypatch):
    # Реальный случай (Постановление №1430, 2026-10-02): на сложной реальной
    # таблице геометрия двух ячеек даёт конфликтующие span'ы, которые python-docx
    # не может представить одним прямоугольным объединением
    # (InvalidSpanError: requested span not rectangular). Раньше это роняло всю
    # сборку DOCX, теряя OCR всех страниц документа. Теперь такая ячейка остаётся
    # неотъединённой (текст не теряется), а не валит всю сборку.
    import docx.table
    from docx.exceptions import InvalidSpanError

    original_merge = docx.table._Cell.merge
    call_count = {"n": 0}

    def _flaky_merge(self, other_cell):
        call_count["n"] += 1
        if call_count["n"] == 1:
            raise InvalidSpanError("requested span not rectangular")
        return original_merge(self, other_cell)

    monkeypatch.setattr(docx.table._Cell, "merge", _flaky_merge)

    cells = [
        TableCell(row=0, col=0, row_span=1, col_span=2, text="Конфликтующая ячейка"),
        TableCell(row=1, col=0, row_span=1, col_span=1, text="a"),
        TableCell(row=1, col=1, row_span=1, col_span=1, text="b"),
    ]
    pages = [
        PageResult(
            page_number=1,
            ocr_applied=True,
            blocks=[TableBlock(y=0, bbox=(0, 0, 100, 100), cells=cells, n_rows=2, n_cols=2)],
        ),
    ]
    out = tmp_path / "out.docx"
    build_docx(pages, out)  # не должно бросить InvalidSpanError
    assert out.is_file()

    from docx import Document

    doc = Document(str(out))
    table = doc.tables[0]
    # Текст не потерян, даже хотя объединение не удалось.
    assert table.cell(0, 0).text == "Конфликтующая ячейка"
    assert table.cell(1, 0).text == "a"
    assert table.cell(1, 1).text == "b"


def test_ocr_document_reuses_existing_text_layer_without_calling_engines(tmp_path: Path):
    import fitz

    pdf_path = tmp_path / "with_text.pdf"
    doc = fitz.open()
    page = doc.new_page()
    # Латиница намеренно — встроенный шрифт fitz.insert_text() по умолчанию не
    # поддерживает кириллицу (буквы превращаются в "." при извлечении текста);
    # для этого теста важна только сама логика пропуска OCR, не язык содержимого.
    page.insert_text((72, 72), "This page already has a real text layer, not a scan.")
    doc.save(str(pdf_path))
    doc.close()

    def _should_not_be_called(_path):
        raise AssertionError("движок не должен вызываться — у страницы есть текстовый слой")

    result = ocr_document(
        pdf_path,
        tmp_path / "out.docx",
        predict_tables_fn=_should_not_be_called,
        image_data_fn=_should_not_be_called,
    )
    assert isinstance(result, OcrDocumentResult)
    assert len(result.pages) == 1
    assert result.pages[0].ocr_applied is False
    assert "real text layer" in result.pages[0].plain_text
    assert result.docx_path.is_file()


def test_ocr_document_does_not_build_default_engines_when_no_scan_pages(
    tmp_path: Path, monkeypatch
):
    # PPStructureV3 грузит ~19 моделей (минуты, сотни МБ) — документ без единой
    # страницы-скана не должен платить эту цену; CLI smoke-тест на реальном
    # венве подтвердил это (0.3с вместо нескольких минут), здесь — то же самое
    # без реальных тяжёлых движков, через monkeypatch default-конструктора.
    import fitz

    import npa_ocr.ocr as ocr_module

    def _should_not_be_called():
        raise AssertionError(
            "_default_predict_tables_fn не должен вызываться — у документа нет страниц-сканов"
        )

    monkeypatch.setattr(ocr_module, "_default_predict_tables_fn", _should_not_be_called)

    pdf_path = tmp_path / "with_text.pdf"
    doc = fitz.open()
    page = doc.new_page()
    page.insert_text((72, 72), "This page already has a real text layer, not a scan.")
    doc.save(str(pdf_path))
    doc.close()

    result = ocr_document(pdf_path, tmp_path / "out.docx")
    assert result.pages[0].ocr_applied is False


def test_ocr_document_runs_hybrid_pipeline_when_no_text_layer(tmp_path: Path):
    import fitz

    pdf_path = tmp_path / "scan.pdf"
    doc = fitz.open()
    doc.new_page()  # пустая страница — имитация скана без текстового слоя
    doc.save(str(pdf_path))
    doc.close()

    fixture = _load_simple_table_json()
    text_data = _tesseract_data([("Распознано", 90, 10, 5, 100, 20)])
    calls = {"tables": 0, "text": 0}

    def fake_predict(_path):
        calls["tables"] += 1
        return fixture

    def fake_image_data(_path):
        calls["text"] += 1
        return text_data

    result = ocr_document(
        pdf_path,
        tmp_path / "out.docx",
        predict_tables_fn=fake_predict,
        image_data_fn=fake_image_data,
    )
    assert calls["tables"] == 1
    assert calls["text"] == 1
    assert result.pages[0].ocr_applied is True
    assert any(isinstance(b, TableBlock) for b in result.pages[0].blocks)
    assert result.docx_path.is_file()


def test_ensure_free_memory_raises_below_threshold(monkeypatch):
    import psutil

    from npa_ocr.ocr import InsufficientMemoryError, ensure_free_memory

    class _Mem:
        available = 500 * 1024 * 1024

    monkeypatch.setattr(psutil, "virtual_memory", lambda: _Mem())
    with pytest.raises(InsufficientMemoryError, match="перед страницей 3"):
        ensure_free_memory(2048, "перед страницей 3")
    ensure_free_memory(0, "отключено")  # 0 — проверка отключена, ничего не делает


def test_ocr_document_stops_before_engines_when_memory_is_low(tmp_path: Path, monkeypatch):
    import fitz
    import psutil

    from npa_ocr.ocr import InsufficientMemoryError

    class _Mem:
        available = 100 * 1024 * 1024

    monkeypatch.setattr(psutil, "virtual_memory", lambda: _Mem())
    pdf_path = tmp_path / "scan.pdf"
    doc = fitz.open()
    doc.new_page()
    doc.save(str(pdf_path))
    doc.close()

    def _should_not_load(*_a, **_k):
        raise AssertionError("движок не должен загружаться при нехватке памяти")

    import npa_ocr.ocr as ocr_module

    monkeypatch.setattr(ocr_module, "_default_predict_tables_fn", _should_not_load)
    with pytest.raises(InsufficientMemoryError):
        ocr_document(pdf_path, tmp_path / "out.docx", min_free_memory_mb=2048)


def test_text_paragraphs_restores_centered_lines_indents_and_breaks():
    from npa_ocr.ocr import _text_paragraphs

    # Геометрия реальной первой страницы Постановления №1430 (200 DPI).
    run = [
        TextBlock(y=435, text="ПРАВИТЕЛЬСТВО РОССИЙСКОЙ ФЕДЕРАЦИИ", x0=270, x1=1381, height=53),
        TextBlock(y=821, text="Об утверждении технологических показателей", x0=313, x1=1337, height=35),
        TextBlock(y=866, text="доступных технологий в сфере", x0=366, x1=1281, height=34),
        TextBlock(y=1088, text="В соответствии с пунктом 5 статьи 23", x0=300, x1=1447, height=44),
        TextBlock(y=1142, text="Федерального закона Правительство", x0=203, x1=1446, height=35),
        TextBlock(y=1201, text="постановляет:", x0=202, x1=563, height=19),
        TextBlock(y=1242, text="Утвердить прилагаемые показатели", x0=299, x1=1447, height=34),
        TextBlock(y=1290, text="доступных технологий в сфере", x0=201, x1=1446, height=36),
    ]
    paras = _text_paragraphs(run, dpi=200)
    # «постановляет:» — короткая последняя строка вводной части, абзац не начинает.
    assert [p[0] for p in paras] == [
        "ПРАВИТЕЛЬСТВО РОССИЙСКОЙ ФЕДЕРАЦИИ",
        "Об утверждении технологических показателей доступных технологий в сфере",
        "В соответствии с пунктом 5 статьи 23 Федерального закона Правительство постановляет:",
        "Утвердить прилагаемые показатели доступных технологий в сфере",
    ]
    assert [p[1] for p in paras][:2] == ["center", "center"]
    assert paras[2][1] == "left" and paras[2][2] > 0  # «В соответствии» начинается с отступа (x0=300)
    assert paras[3][2] > 0  # «Утвердить» с отступом первой строки


def test_text_paragraphs_without_geometry_keeps_one_line_per_paragraph():
    from npa_ocr.ocr import _text_paragraphs

    run = [TextBlock(y=0, text="a"), TextBlock(y=40, text="b")]
    assert [(t, a, f) for t, a, f, _ in _text_paragraphs(run, dpi=200)] == [("a", "left", 0.0), ("b", "left", 0.0)]


def test_remove_running_headers_and_footers_drops_page_numbers_and_repeats_only():
    from npa_ocr.ocr import remove_running_headers_and_footers

    def page(n: int, footer_text: str, body: list[str]) -> PageResult:
        blocks = [TextBlock(y=500 + 60 * i, text=t, x0=200, x1=1400, height=35) for i, t in enumerate(body)]
        blocks.append(TextBlock(y=960, text=footer_text, x0=200, x1=400, height=20))  # нижний колонтитул
        blocks.append(TextBlock(y=20, text=str(n), x0=800, x1=850, height=20))  # номер страницы сверху
        return PageResult(page_number=n, ocr_applied=True, blocks=blocks, page_height=1000.0)

    pages = [
        page(1, "4678924", ["Текст первой страницы"]),
        page(2, "4678924", ["Текст второй страницы"]),
        page(3, "4678924", ["Текст третьей страницы"]),
    ]
    cleaned = remove_running_headers_and_footers(pages)
    for p in cleaned:
        texts = [b.text for b in p.blocks]
        assert "4678924" not in texts
        assert str(p.page_number) not in texts
    assert [b.text for b in cleaned[0].blocks] == ["Текст первой страницы"]


def test_remove_running_headers_keeps_repeated_text_in_body_and_single_footer_lines():
    from npa_ocr.ocr import remove_running_headers_and_footers

    body_repeat = TextBlock(y=500, text="Одинаковая строка тела", x0=200, x1=1400, height=35)
    pages = [
        PageResult(page_number=i, ocr_applied=True, blocks=[body_repeat, TextBlock(y=960, text=f"Колонтитул {name}", x0=200, x1=600, height=20)], page_height=1000.0)
        for i, name in ((1, "alpha"), (2, "beta"), (3, "gamma"))
    ]
    cleaned = remove_running_headers_and_footers(pages)
    for p in cleaned:
        texts = [b.text for b in p.blocks]
        assert "Одинаковая строка тела" in texts  # вне полос — не трогаем, даже если повторяется
        assert any(t.startswith("Колонтитул ") for t in texts)  # уникален — не повторяется, сохраняем


def test_run_tatr_tables_builds_block_with_offset_and_text_from_ocr(tmp_path: Path):
    from PIL import Image

    from npa_ocr.ocr import run_tatr_tables
    from npa_ocr.table_structure import TableStructure

    img_path = tmp_path / "page.png"
    Image.new("RGB", (600, 600), "white").save(img_path)

    def fake_region(_img):
        return [(100.0, 100.0, 400.0, 300.0)]

    def fake_structure(_crop):
        # сетка 2×2 в координатах кропа; третьего столбца нет, он пуст
        return TableStructure(
            rows=[(0.0, 0.0, 300.0, 100.0), (0.0, 100.0, 300.0, 200.0)],
            cols=[(0.0, 0.0, 150.0, 200.0), (150.0, 0.0, 300.0, 200.0), (300.0, 0.0, 320.0, 200.0)],
            spans=[],
        )

    def fake_text(_crop_path):
        # Третья ячейка (1,0) — чтобы сетка 2x2 прошла is_plausible_table
        # (не часть проверяемого поведения этого теста — offset/dropped_columns).
        return {
            "table_res_list": [
                {
                    "table_ocr_pred": {
                        "rec_boxes": [[10, 10, 140, 40], [160, 110, 290, 140], [10, 110, 140, 140]],
                        "rec_texts": ["Ячейка-1", "Ячейка-2", "Ячейка-3"],
                    }
                }
            ]
        }

    blocks = run_tatr_tables(
        img_path, fake_text, region_fn=fake_region, structure_fn=fake_structure,
        img2table_fallback_fn=lambda _p: [],
    )
    assert len(blocks) == 1
    b = blocks[0]
    assert b.dropped_columns == 1  # пустой третий столбец удалён и учтён
    assert (b.n_rows, b.n_cols) == (2, 2)
    by_pos = {(c.row, c.col): c.text for c in b.cells}
    assert by_pos[(0, 0)] == "Ячейка-1"
    assert by_pos[(1, 1)] == "Ячейка-2"
    x1, y1, x2, y2 = b.bbox
    assert 80.0 <= x1 and x2 <= 420.0 and 80.0 <= y1 and y2 <= 320.0  # bbox — по содержимому, с отступом вокруг текста


def test_run_tatr_tables_drops_duplicate_texts_from_overlapping_table_entries(tmp_path: Path):
    from PIL import Image

    from npa_ocr.ocr import run_tatr_tables
    from npa_ocr.table_structure import TableStructure

    img_path = tmp_path / "page.png"
    Image.new("RGB", (600, 600), "white").save(img_path)

    def fake_text(_crop_path):
        # две записи таблиц на одном кропе содержат одну и ту же строку;
        # второй столбец с текстом — иначе drop_empty_columns схлопывает
        # сетку до 1x1 и is_plausible_table её отбраковывает (см. docs/history/plan.md)
        same = {
            "rec_boxes": [[10, 10, 140, 40], [160, 10, 290, 40]],
            "rec_texts": ["Ячейка", "Вторая"],
        }
        return {"table_res_list": [{"table_ocr_pred": same}, {"table_ocr_pred": same}]}

    blocks = run_tatr_tables(
        img_path,
        fake_text,
        region_fn=lambda _img: [(100.0, 100.0, 400.0, 300.0)],
        structure_fn=lambda _crop: TableStructure(
            rows=[(0.0, 0.0, 300.0, 100.0)], cols=[(0.0, 0.0, 150.0, 100.0), (150.0, 0.0, 300.0, 100.0)], spans=[]
        ),
        img2table_fallback_fn=lambda _p: [],
    )
    assert blocks[0].cells[0].text == "Ячейка"  # не "Ячейка Ячейка"


def test_run_tatr_tables_recognizes_each_cell_separately_when_cell_fn_given(tmp_path: Path):
    from PIL import Image

    from npa_ocr.ocr import run_tatr_tables
    from npa_ocr.table_structure import TableStructure

    img_path = tmp_path / "page.png"
    Image.new("RGB", (600, 600), "white").save(img_path)
    seen: list[str] = []

    def fake_cell_text(cell_path):
        seen.append(Path(cell_path).name)
        return "ЯЧ", (False,)

    blocks = run_tatr_tables(
        img_path,
        lambda _p: {
            "table_res_list": [
                {
                    "table_ocr_pred": {
                        # второй столбец с текстом — иначе drop_empty_columns
                        # схлопывает сетку до 1x1 и is_plausible_table её
                        # отбраковывает (см. docs/history/plan.md)
                        "rec_boxes": [[10, 10, 90, 40], [160, 10, 240, 40]],
                        "rec_texts": ["x", "y"],
                    }
                }
            ]
        },
        cell_text_fn=fake_cell_text,
        region_fn=lambda _img: [(100.0, 100.0, 400.0, 300.0)],
        structure_fn=lambda _crop: TableStructure(
            rows=[(0.0, 0.0, 300.0, 100.0)],
            cols=[(0.0, 0.0, 150.0, 100.0), (150.0, 0.0, 300.0, 100.0)],
            spans=[],
        ),
        img2table_fallback_fn=lambda _p: [],
    )
    assert seen  # каждая ячейка с текстом распознана отдельным кропом
    assert blocks[0].cells[0].text == "ЯЧ"


def test_unique_text_blocks_merges_same_text_with_shifted_boxes():
    from npa_ocr.ocr import _unique_text_blocks

    text_json = {
        "table_res_list": [
            {"table_ocr_pred": {"rec_boxes": [[10, 10, 100, 40]], "rec_texts": ["N п/п"]}},
            {"table_ocr_pred": {"rec_boxes": [[12, 13, 102, 44]], "rec_texts": ["N п/п"]}},
            {"table_ocr_pred": {"rec_boxes": [[300, 10, 400, 40]], "rec_texts": ["N п/п"]}},
        ]
    }
    blocks = _unique_text_blocks(text_json)
    assert len(blocks) == 2  # сдвинутый дубликат убран; тот же текст в другом месте — оставлен


def test_run_tatr_tables_downscales_large_crop_for_text_and_restores_coordinates(tmp_path: Path):
    from PIL import Image

    from npa_ocr.ocr import TEXT_PREDICT_MAX_SIDE, run_tatr_tables
    from npa_ocr.table_structure import TableStructure

    img_path = tmp_path / "page.png"
    Image.new("RGB", (4000, 3500), "white").save(img_path)
    seen: dict[str, tuple[int, int]] = {}

    def fake_text(crop_path):
        seen["size"] = Image.open(crop_path).size
        return {"table_res_list": [{"table_ocr_pred": {"rec_boxes": [[100, 100, 300, 140]], "rec_texts": ["ок"]}}]}

    run_tatr_tables(
        img_path,
        fake_text,
        region_fn=lambda _img: [(0.0, 0.0, 3900.0, 3300.0)],
        structure_fn=lambda _crop: TableStructure(
            rows=[(0.0, 0.0, 4000.0, 4000.0)], cols=[(0.0, 0.0, 4000.0, 4000.0)], spans=[]
        ),
        img2table_fallback_fn=lambda _p: [],
    )
    assert max(seen["size"]) <= TEXT_PREDICT_MAX_SIDE


def test_run_tatr_tables_flags_low_confidence_cell_and_docx_highlights_it(tmp_path: Path):
    from docx import Document
    from docx.enum.text import WD_COLOR_INDEX
    from PIL import Image

    from npa_ocr.ocr import PageResult, TableBlock, build_docx, run_tatr_tables
    from npa_ocr.table_structure import TableStructure

    img_path = tmp_path / "page.png"
    Image.new("RGB", (600, 600), "white").save(img_path)

    def fake_cell_text(cell_path):
        return ("сомнительно", (True,)) if "cell_0_0" in Path(cell_path).name else ("верно", (False,))

    blocks = run_tatr_tables(
        img_path,
        lambda _p: {"table_res_list": [{"table_ocr_pred": {"rec_boxes": [[10, 10, 90, 40], [160, 10, 240, 40]], "rec_texts": ["a", "b"]}}]},
        cell_text_fn=fake_cell_text,
        region_fn=lambda _img: [(100.0, 100.0, 400.0, 300.0)],
        structure_fn=lambda _crop: TableStructure(
            rows=[(0.0, 0.0, 300.0, 100.0)],
            cols=[(0.0, 0.0, 150.0, 100.0), (150.0, 0.0, 300.0, 100.0)],
            spans=[],
        ),
        img2table_fallback_fn=lambda _p: [],
    )
    by_pos = {(c.row, c.col): c for c in blocks[0].cells}
    assert by_pos[(0, 0)].low_lines == (True,)
    assert not any(by_pos[(0, 1)].low_lines)

    out = tmp_path / "out.docx"
    build_docx([PageResult(page_number=1, ocr_applied=True, blocks=blocks)], out)
    doc = Document(str(out))
    table = doc.tables[0]
    highlighted = [r.text for r in table.cell(0, 0).paragraphs[0].runs if r.font.highlight_color == WD_COLOR_INDEX.YELLOW]
    assert highlighted == ["сомнительно"]
    assert all(r.font.highlight_color is None for r in table.cell(0, 1).paragraphs[0].runs)
    assert any("\u0416\u0451\u043b\u0442\u044b\u043c" in p.text for p in doc.paragraphs)


def test_body_text_low_confidence_word_is_highlighted_in_docx(tmp_path: Path):
    from docx import Document
    from docx.enum.text import WD_COLOR_INDEX

    from npa_ocr.ocr import PageResult, build_docx

    data = _tesseract_data([("Текст", 95, 10, 10, 100, 20), ("сомнительно", 20, 120, 10, 100, 20)])
    blocks = run_tesseract_text(Path("unused.png"), [], image_data_fn=lambda _p: data, confidence_threshold=60)
    out = tmp_path / "body.docx"
    build_docx([PageResult(page_number=1, ocr_applied=True, blocks=blocks)], out)
    doc = Document(str(out))
    para = next(p for p in doc.paragraphs if "Текст" in p.text)
    highlighted = [r.text for r in para.runs if r.font.highlight_color == WD_COLOR_INDEX.YELLOW]
    assert highlighted == ["сомнительно"]
    assert "[нрзб.]" not in para.text


def _two_text_pages_pdf(path: Path) -> Path:
    import fitz

    doc = fitz.open()
    for _ in range(2):
        doc.new_page().insert_text((72, 72), "This page already has a real text layer, not a scan.")
    doc.save(str(path))
    doc.close()
    return path


def test_ocr_document_reports_progress_and_saves_each_page(tmp_path: Path):
    pdf_path = _two_text_pages_pdf(tmp_path / "two.pdf")
    events = []
    ocr_document(
        pdf_path,
        tmp_path / "out.docx",
        progress_fn=lambda done, total, n, status: events.append((done, total, n, status)),
    )
    assert events == [(1, 2, 1, "текстовый слой"), (2, 2, 2, "текстовый слой")]
    pages_dir = tmp_path / "out.pages"
    assert sorted(p.name for p in pages_dir.iterdir()) == ["page_0001.json", "page_0002.json"]


def test_ocr_document_resumes_from_saved_pages_without_engines(tmp_path: Path):
    pdf_path = _two_text_pages_pdf(tmp_path / "two.pdf")
    ocr_document(pdf_path, tmp_path / "out.docx", progress_fn=lambda *a: None)

    def _should_not_be_called(_path):
        raise AssertionError("страницы уже сохранены — движки не должны вызываться")

    events = []
    result = ocr_document(
        pdf_path,
        tmp_path / "out.docx",
        predict_tables_fn=_should_not_be_called,
        image_data_fn=_should_not_be_called,
        progress_fn=lambda done, total, n, status: events.append(status),
    )
    assert [p.page_number for p in result.pages] == [1, 2]
    assert events == ["взято из сохранённого прогона"] * 2


def test_ocr_document_ignores_saved_pages_with_other_parameters(tmp_path: Path):
    pdf_path = _two_text_pages_pdf(tmp_path / "two.pdf")
    ocr_document(pdf_path, tmp_path / "out.docx", dpi=200, progress_fn=lambda *a: None)
    events = []
    ocr_document(
        pdf_path,
        tmp_path / "out.docx",
        dpi=150,
        progress_fn=lambda done, total, n, status: events.append(status),
    )
    assert events == ["текстовый слой"] * 2


def test_saved_page_roundtrip_keeps_text_and_table_blocks(tmp_path: Path):
    from npa_ocr.ocr import _load_page, _save_page

    table = TableBlock(
        y=10.0,
        bbox=(1.0, 10.0, 200.0, 90.0),
        cells=[TableCell(row=0, col=0, row_span=1, col_span=1, text="а\nб", low_lines=(False, True))],
        n_rows=1,
        n_cols=1,
        dropped_columns=2,
    )
    text = TextBlock(y=100.0, text="Обычный текст", x0=5.0, x1=300.0, height=20.0, low_words=(False, True))
    page = PageResult(page_number=3, ocr_applied=True, blocks=[table, text], page_height=1200.0)
    meta = {"dpi": 200}
    path = tmp_path / "page_0003.json"
    _save_page(path, page, meta)

    loaded = _load_page(path, meta)
    assert loaded == page
    assert _load_page(path, {"dpi": 150}) is None


def test_ocr_document_writes_timings_per_document_and_per_page(tmp_path: Path):
    pdf_path = _two_text_pages_pdf(tmp_path / "two.pdf")
    result = ocr_document(pdf_path, tmp_path / "out.docx", progress_fn=lambda *a: None)
    timings = json.loads(result.timings_path.read_text(encoding="utf-8"))
    assert result.timings_path == tmp_path / "out.timings.json"
    assert [s["step"] for s in timings["document_steps"]] == [
        "Открытие и подготовка",
        "Завершение распознавания",
        "Колонтитулы",
        "Сборка DOCX",
    ]
    assert sorted(timings["pages"]) == ["1", "2"]
    for page in timings["pages"].values():
        assert [s["step"] for s in page["steps"]][0] == "проверка сохранённого результата"
        assert page["total_seconds"] >= 0
    assert timings["total_seconds"] >= sum(s["seconds"] for s in timings["document_steps"])


def test_lazy_engine_loads_once_and_reloads_after_release():
    from npa_ocr.ocr import _LazyEngine

    created = []

    def factory():
        created.append(1)
        return lambda x: x * 2

    engine = _LazyEngine(factory)
    assert engine(3) == 6
    assert engine(4) == 8
    assert len(created) == 1
    engine.release()
    assert engine(1) == 2
    assert len(created) == 2


class _FakeI2TBBox:
    def __init__(self, x1, y1, x2, y2):
        self.x1, self.y1, self.x2, self.y2 = x1, y1, x2, y2


class _FakeI2TCell:
    def __init__(self, bbox, value):
        self.bbox = bbox
        self.value = value


class _FakeI2TTable:
    def __init__(self, content):
        self.content = content


def test_table_cells_from_img2table_builds_flat_grid_without_merges():
    content = {
        0: [_FakeI2TCell(_FakeI2TBBox(0, 0, 50, 20), "A"), _FakeI2TCell(_FakeI2TBBox(50, 0, 100, 20), "B")],
        # у строки 1 заполнены ОБЕ колонки — однозначно настоящая отдельная
        # строка, а не обрывок (см. _merge_img2table_fragment_rows: обрывок
        # имеет ровно одну непустую колонку)
        1: [_FakeI2TCell(_FakeI2TBBox(0, 20, 50, 40), "C"), _FakeI2TCell(_FakeI2TBBox(50, 20, 100, 40), "E")],
    }
    cells, n_rows, n_cols = _table_cells_from_img2table(_FakeI2TTable(content))
    assert (n_rows, n_cols) == (2, 2)
    by_pos = {(c.row, c.col): c for c in cells}
    assert by_pos[(0, 0)].text == "A" and by_pos[(0, 0)].row_span == by_pos[(0, 0)].col_span == 1
    assert by_pos[(1, 0)].text == "C"
    assert by_pos[(1, 1)].text == "E"


def test_table_cells_from_img2table_treats_none_value_as_empty_string():
    content = {0: [_FakeI2TCell(_FakeI2TBBox(0, 0, 50, 20), "A"), _FakeI2TCell(_FakeI2TBBox(50, 0, 100, 20), None)]}
    cells, n_rows, n_cols = _table_cells_from_img2table(_FakeI2TTable(content))
    by_pos = {(c.row, c.col): c for c in cells}
    assert by_pos[(0, 1)].text == ""  # None -> пустая строка, не "None"


def test_table_cells_from_img2table_merges_repeated_bbox_into_span():
    # Объединённая по вертикали ячейка (img2table повторяет один и тот же bbox
    # в обеих строках, см. _table_cells_from_img2table) должна стать одной
    # TableCell с row_span=2, а не двумя отдельными ячейками с одинаковым текстом.
    # Строка 1 — ДВЕ новые непустые ячейки (col1, col2), иначе при одной
    # новой ячейке строку поглотила бы _merge_img2table_fragment_rows
    # (это проверяется отдельным тестом с реальными данными).
    merged_bbox = _FakeI2TBBox(0, 0, 50, 40)
    content = {
        0: [
            _FakeI2TCell(merged_bbox, "Merged"),
            _FakeI2TCell(_FakeI2TBBox(50, 0, 100, 20), "B"),
            _FakeI2TCell(_FakeI2TBBox(100, 0, 150, 20), "X"),
        ],
        1: [
            _FakeI2TCell(merged_bbox, "Merged"),
            _FakeI2TCell(_FakeI2TBBox(50, 20, 100, 40), "D"),
            _FakeI2TCell(_FakeI2TBBox(100, 20, 150, 40), "Y"),
        ],
    }
    cells, n_rows, n_cols = _table_cells_from_img2table(_FakeI2TTable(content))
    assert (n_rows, n_cols) == (2, 3)
    assert len(cells) == 5  # не 6 — объединённая ячейка посчитана один раз
    merged = next(c for c in cells if c.text == "Merged")
    assert (merged.row, merged.col, merged.row_span, merged.col_span) == (0, 0, 2, 1)


def test_merge_img2table_fragment_rows_joins_real_split_cell_from_order_215():
    # Реальные данные (Приказ №215, стр. 4, см. docs/history/plan.md, 2026-10-24):
    # img2table разбил ячейку «Углерода оксид (углерод окись; углерод
    # моноокись; угарный газ)» на 4 отдельные "строки" без реальной линии-
    # разделителя между ними в самой таблице.
    cells = [
        TableCell(row=0, col=0, row_span=1, col_span=1, text="метода осушки газа"),
        TableCell(row=0, col=1, row_span=1, col_span=1, text="Метан"),
        TableCell(row=0, col=2, row_span=1, col_span=1, text="< 0,68"),
        TableCell(row=1, col=0, row_span=1, col_span=1, text=""),
        TableCell(row=1, col=1, row_span=1, col_span=1, text="Углерода оксид"),
        TableCell(row=1, col=2, row_span=1, col_span=1, text="< 0,09"),
        TableCell(row=2, col=0, row_span=1, col_span=1, text=""),
        TableCell(row=2, col=1, row_span=1, col_span=1, text="(углерод окись;"),
        TableCell(row=2, col=2, row_span=1, col_span=1, text=""),
        TableCell(row=3, col=0, row_span=1, col_span=1, text=""),
        TableCell(row=3, col=1, row_span=1, col_span=1, text="углерод моноокись;"),
        TableCell(row=3, col=2, row_span=1, col_span=1, text=""),
        TableCell(row=4, col=0, row_span=1, col_span=1, text=""),
        TableCell(row=4, col=1, row_span=1, col_span=1, text="угарный газ)"),
        TableCell(row=4, col=2, row_span=1, col_span=1, text=""),
        # «Азота диоксид» — НОВОЕ вещество сразу после предыдущего обрывка,
        # тоже с единственной непустой колонкой и без собственного значения —
        # ключевая проверка: не должно склеиться с «Углерода оксид» только
        # из-за совпадения колонки (см. _looks_like_fragment_continuation).
        TableCell(row=5, col=0, row_span=1, col_span=1, text=""),
        TableCell(row=5, col=1, row_span=1, col_span=1, text="Азота диоксид"),
        TableCell(row=5, col=2, row_span=1, col_span=1, text=""),
        # и у «Азота диоксид» свои собственные обрывки-продолжения
        TableCell(row=6, col=0, row_span=1, col_span=1, text=""),
        TableCell(row=6, col=1, row_span=1, col_span=1, text="(двуокись азота;"),
        TableCell(row=6, col=2, row_span=1, col_span=1, text=""),
        TableCell(row=7, col=0, row_span=1, col_span=1, text=""),
        TableCell(row=7, col=1, row_span=1, col_span=1, text="пероксид азота)"),
        TableCell(row=7, col=2, row_span=1, col_span=1, text=""),
    ]
    merged, n_rows, n_cols = _merge_img2table_fragment_rows(cells, n_rows=8, n_cols=3)
    assert n_rows == 3  # строки 2,3,4 поглощены строкой 1; строки 6,7 поглощены строкой 5
    assert n_cols == 3
    by_pos = {(c.row, c.col): c for c in merged}
    assert by_pos[(0, 1)].text == "Метан"
    assert by_pos[(1, 1)].text == "Углерода оксид\n(углерод окись;\nуглерод моноокись;\nугарный газ)"
    assert by_pos[(1, 2)].text == "< 0,09"  # значение рядом не потеряно
    assert by_pos[(2, 1)].text == "Азота диоксид\n(двуокись азота;\nпероксид азота)"  # не склеилось с «Углерода оксид»


def test_merge_img2table_fragment_rows_keeps_real_standalone_rows_separate():
    # У настоящей отдельной строки, помимо вещества, обычно заполнена и
    # соседняя колонка (например, значение норматива) — не обрывок.
    cells = [
        TableCell(row=0, col=0, row_span=1, col_span=1, text="Процесс"),
        TableCell(row=0, col=1, row_span=1, col_span=1, text="Метан"),
        TableCell(row=0, col=2, row_span=1, col_span=1, text="< 1,0"),
        TableCell(row=1, col=0, row_span=1, col_span=1, text=""),
        TableCell(row=1, col=1, row_span=1, col_span=1, text="Азота диоксид"),
        TableCell(row=1, col=2, row_span=1, col_span=1, text="< 2,0"),
    ]
    merged, n_rows, n_cols = _merge_img2table_fragment_rows(cells, n_rows=2, n_cols=3)
    assert n_rows == 2  # ничего не поглощено
    by_pos = {(c.row, c.col): c for c in merged}
    assert by_pos[(1, 1)].text == "Азота диоксид"
    assert by_pos[(1, 2)].text == "< 2,0"


def test_merge_img2table_fragment_rows_merges_columns_independently_within_same_row():
    # Реальный документ (Приказ №215, стр. 4): в одной "строке" колонка 0
    # продолжает длинное название ПРЕДЫДУЩЕГО процесса, а колонка 1 в ТОЙ ЖЕ
    # "строке" начинает совсем ДРУГОЕ, новое вещество — обе колонки должны
    # обрабатываться независимо, не блокируя друг друга.
    cells = [
        TableCell(row=0, col=0, row_span=1, col_span=1, text="Подготовка газа\nгорючего"),
        TableCell(row=0, col=1, row_span=1, col_span=1, text="Азота оксид"),
        TableCell(row=0, col=2, row_span=1, col_span=1, text=""),
        TableCell(row=1, col=0, row_span=1, col_span=1, text="природного\nк транспорту"),
        TableCell(row=1, col=1, row_span=1, col_span=1, text="Метан"),
        TableCell(row=1, col=2, row_span=1, col_span=1, text="< 0,088"),
        TableCell(row=2, col=0, row_span=1, col_span=1, text="метода осушки"),
        TableCell(row=2, col=1, row_span=1, col_span=1, text="Углерода оксид"),
        TableCell(row=2, col=2, row_span=1, col_span=1, text="< 0,016"),
    ]
    merged, n_rows, n_cols = _merge_img2table_fragment_rows(cells, n_rows=3, n_cols=3)
    assert n_rows == 3  # ни одна строка не удалена — у каждой свой якорь в col1
    by_pos = {(c.row, c.col): c for c in merged}
    assert by_pos[(0, 0)].text == "Подготовка газа\nгорючего\nприродного\nк транспорту\nметода осушки"
    assert by_pos[(1, 0)].text == ""  # поглощено колонкой 0 строки 0
    assert by_pos[(1, 1)].text == "Метан"  # своя отдельная цепочка по колонке 1
    assert by_pos[(1, 2)].text == "< 0,088"
    assert by_pos[(2, 1)].text == "Углерода оксид"
    assert by_pos[(2, 2)].text == "< 0,016"


def test_looks_like_fragment_continuation_distinguishes_new_entity_from_continuation():
    assert _looks_like_fragment_continuation("(углерод окись;") is True
    assert _looks_like_fragment_continuation("углерод моноокись;") is True
    assert _looks_like_fragment_continuation("Азота диоксид") is False
    assert _looks_like_fragment_continuation("Метан") is False
    assert _looks_like_fragment_continuation("") is False
    assert _looks_like_fragment_continuation("   ") is False


def test_merge_img2table_fragment_rows_noop_for_single_row_table():
    cells = [TableCell(row=0, col=0, row_span=1, col_span=1, text="A")]
    merged, n_rows, n_cols = _merge_img2table_fragment_rows(cells, n_rows=1, n_cols=1)
    assert (merged, n_rows, n_cols) == (cells, 1, 1)


def test_best_img2table_match_picks_candidate_with_largest_vertical_overlap():
    candidates = [
        ((0.0, 90.0, 100.0, 310.0), [], 1, 1),  # почти полностью перекрывает [100,300]
        ((0.0, 500.0, 100.0, 520.0), [], 1, 1),  # не перекрывает вовсе
    ]
    match = _best_img2table_match(100.0, 300.0, candidates)
    assert match is not None
    assert match[0] == (0.0, 90.0, 100.0, 310.0)


def test_best_img2table_match_returns_none_below_overlap_threshold():
    # Перекрытие слишком маленькое относительно высоты кандидата — не та же таблица.
    candidates = [((0.0, 295.0, 100.0, 400.0), [], 1, 1)]  # overlap=5 из высоты кандидата 105
    assert _best_img2table_match(100.0, 300.0, candidates) is None


def test_run_tatr_tables_uses_img2table_fallback_when_region_is_clipped(tmp_path: Path):
    from PIL import Image

    from npa_ocr.ocr import run_tatr_tables
    from npa_ocr.table_structure import TableStructure

    img_path = tmp_path / "page.png"
    Image.new("RGB", (1000, 600), "white").save(img_path)

    def fake_text(_crop_path):
        # два слова (второй столбец — иначе сетка схлопывается до 1x1 и
        # отбраковывается is_plausible_table, см. docs/history/plan.md); вместе занимают
        # только левую часть кропа — итоговый bbox TATR узкий, правый отступ от
        # ширины страницы (1000px) превышает порог 0.20
        return {
            "table_res_list": [
                {
                    "table_ocr_pred": {
                        "rec_boxes": [[0, 10, 200, 40], [220, 10, 420, 40]],
                        "rec_texts": ["Узко", "Текст"],
                    }
                }
            ]
        }

    wide_candidate = (
        (50.0, 90.0, 950.0, 310.0),
        [TableCell(row=0, col=0, row_span=1, col_span=1, text="Широкая таблица")],
        1,
        1,
    )

    blocks = run_tatr_tables(
        img_path,
        fake_text,
        region_fn=lambda _img: [(100.0, 100.0, 500.0, 300.0)],
        structure_fn=lambda _crop: TableStructure(
            rows=[(0.0, 0.0, 440.0, 100.0)], cols=[(0.0, 0.0, 210.0, 100.0), (210.0, 0.0, 440.0, 100.0)], spans=[]
        ),
        img2table_fallback_fn=lambda _p: [wide_candidate],
    )
    assert len(blocks) == 1
    assert blocks[0].bbox == (50.0, 90.0, 950.0, 310.0)
    assert blocks[0].cells[0].text == "Широкая таблица"
    assert blocks[0].dropped_columns == 0


def test_run_tatr_tables_keeps_tatr_result_when_img2table_candidate_is_not_wider(tmp_path: Path):
    from PIL import Image

    from npa_ocr.ocr import run_tatr_tables
    from npa_ocr.table_structure import TableStructure

    img_path = tmp_path / "page.png"
    Image.new("RGB", (1000, 600), "white").save(img_path)

    def fake_text(_crop_path):
        return {
            "table_res_list": [
                {
                    "table_ocr_pred": {
                        "rec_boxes": [[0, 10, 200, 40], [220, 10, 420, 40]],
                        "rec_texts": ["Узко", "Текст"],
                    }
                }
            ]
        }

    narrow_candidate = (
        (80.0, 90.0, 400.0, 310.0),  # уже, чем результат TATR — не должен заменить
        [TableCell(row=0, col=0, row_span=1, col_span=1, text="Ещё уже")],
        1,
        1,
    )

    blocks = run_tatr_tables(
        img_path,
        fake_text,
        region_fn=lambda _img: [(100.0, 100.0, 500.0, 300.0)],
        structure_fn=lambda _crop: TableStructure(
            rows=[(0.0, 0.0, 440.0, 100.0)], cols=[(0.0, 0.0, 210.0, 100.0), (210.0, 0.0, 440.0, 100.0)], spans=[]
        ),
        img2table_fallback_fn=lambda _p: [narrow_candidate],
    )
    assert len(blocks) == 1
    assert blocks[0].cells[0].text == "Узко"  # исходный результат TATR, не заменён


def test_run_tatr_tables_falls_back_to_img2table_when_tatr_finds_no_region_at_all(tmp_path: Path):
    """Реальный случай — Приказ №215, стр. 5 (docs/history/plan.md, «таблица на последней
    странице не находится»): честный хвост многостраничной таблицы, TATR не
    находит там region ни на одном безопасном пороге (`region_fn` возвращает
    []). В отличие от уже проверенного случая «узкий принятый регион» выше,
    здесь регионов TATR нет совсем — fallback на img2table по всей странице
    должен сработать и это единственный источник найденной таблицы."""
    from PIL import Image

    from npa_ocr.ocr import run_tatr_tables

    img_path = tmp_path / "page.png"
    Image.new("RGB", (1000, 600), "white").save(img_path)

    candidate = (
        (50.0, 90.0, 950.0, 310.0),
        [TableCell(row=0, col=0, row_span=1, col_span=1, text="Хвост таблицы")],
        1,
        1,
    )

    blocks = run_tatr_tables(
        img_path,
        lambda _p: {"table_res_list": []},
        region_fn=lambda _img: [],
        img2table_fallback_fn=lambda _p: [candidate],
    )
    assert len(blocks) == 1
    assert blocks[0].bbox == (50.0, 90.0, 950.0, 310.0)
    assert blocks[0].cells[0].text == "Хвост таблицы"


def test_run_tatr_tables_does_not_fall_back_to_img2table_when_tatr_region_was_rejected(tmp_path: Path):
    """Отличает «TATR не нашёл ничего» (fallback выше) от «TATR нашёл регион, но
    он отвергнут как неправдоподобный» — для второго случая понижение порога
    уже проверено и признано опасным (Постановление №710, стр. 13: обычный
    формульный текст даёт ложные «таблицы», см. docs/history/plan.md, «таблица на
    последней странице»). region_fn здесь возвращает регион (TATR что-то
    нашёл), но структура настолько вырождена (1x1 без текста), что
    is_plausible_table его отбросит — img2table-fallback в этой ветке
    вызываться не должен вообще."""
    from PIL import Image

    from npa_ocr.ocr import run_tatr_tables
    from npa_ocr.table_structure import TableStructure

    img_path = tmp_path / "page.png"
    Image.new("RGB", (1000, 600), "white").save(img_path)

    calls: list[Path] = []

    def tracking_fallback(path):
        calls.append(path)
        return []

    blocks = run_tatr_tables(
        img_path,
        lambda _p: {"table_res_list": []},
        region_fn=lambda _img: [(100.0, 100.0, 500.0, 300.0)],
        structure_fn=lambda _crop: TableStructure(rows=[], cols=[], spans=[]),
        img2table_fallback_fn=tracking_fallback,
    )
    assert blocks == []
    assert calls == []  # регион был (не пустой список), просто отвергнут — не тот случай


def test_run_tatr_tables_does_not_call_img2table_fallback_when_region_is_not_clipped(tmp_path: Path):
    from PIL import Image

    from npa_ocr.ocr import run_tatr_tables
    from npa_ocr.table_structure import TableStructure

    img_path = tmp_path / "page.png"
    # Узкая страница — тот же bbox, что и выше (bx2=500), занимает почти всю
    # её ширину (520px), правый отступ 20/520≈0.038 — меньше порога 0.20:
    # fallback вызываться не должен вообще.
    Image.new("RGB", (520, 600), "white").save(img_path)

    def fake_text(_crop_path):
        return {
            "table_res_list": [
                {
                    "table_ocr_pred": {
                        "rec_boxes": [[0, 10, 200, 40], [220, 10, 420, 40]],
                        "rec_texts": ["Широко", "Текст"],
                    }
                }
            ]
        }

    calls: list[Path] = []

    def tracking_fallback(path):
        calls.append(path)
        return []

    blocks = run_tatr_tables(
        img_path,
        fake_text,
        region_fn=lambda _img: [(100.0, 100.0, 500.0, 300.0)],
        structure_fn=lambda _crop: TableStructure(
            rows=[(0.0, 0.0, 440.0, 100.0)], cols=[(0.0, 0.0, 210.0, 100.0), (210.0, 0.0, 440.0, 100.0)], spans=[]
        ),
        img2table_fallback_fn=tracking_fallback,
    )
    assert len(blocks) == 1  # таблица принята и без fallback — регион не узкий
    assert calls == []  # fallback не вызван — регион не выглядит обрезанным


# --- Поворот страницы по Tesseract OSD ---

ROTATION_PAGES = Path(__file__).parent / "fixtures" / "vlm_tables"


def test_parse_osd_rotation_needs_confidence():
    from npa_ocr.ocr import parse_osd_rotation

    osd = "Page number: 0\nOrientation in degrees: 270\nRotate: 90\nOrientation confidence: 7.41\n"
    assert parse_osd_rotation(osd) == 90
    assert parse_osd_rotation(osd.replace("7.41", "1.20")) == 0  # мало текста — не доверяем
    assert parse_osd_rotation("") == 0


def _blank_scan_pdf(path: Path) -> Path:
    import fitz

    doc = fitz.open()
    doc.new_page()  # пустая страница — имитация скана без текстового слоя (595×842 pt, книжная)
    doc.save(str(path))
    doc.close()
    return path


def test_ocr_document_rotates_scan_before_tables_and_text(tmp_path: Path):
    """Альбомная таблица в книжном скане: страница поворачивается до распознавания, и таблицы,
    и Tesseract получают уже повёрнутое (альбомное) изображение."""
    pdf_path = _blank_scan_pdf(tmp_path / "scan.pdf")
    sizes = {}

    def fake_predict(path):
        with Image.open(path) as img:
            sizes["tables"] = img.size
        return _load_simple_table_json()

    def fake_image_data(path):
        with Image.open(path) as img:
            sizes["text"] = img.size
        return _tesseract_data([("Распознано", 90, 10, 5, 100, 20)])

    result = ocr_document(
        pdf_path,
        tmp_path / "out.docx",
        predict_tables_fn=fake_predict,
        image_data_fn=fake_image_data,
        orientation_fn=lambda _path: 90,
        progress_fn=lambda *a: None,
    )
    w, h = sizes["text"]
    assert w > h  # книжная страница стала альбомной
    assert sizes["tables"] == sizes["text"]
    assert result.pages[0].page_height == h
    steps = [s["step"] for s in result.timings["pages"]["1"]["steps"]]
    assert "поворот страницы (OSD)" in steps


def test_ocr_document_without_auto_rotate_does_not_ask_orientation(tmp_path: Path):
    pdf_path = _blank_scan_pdf(tmp_path / "scan.pdf")

    def _should_not_be_called(_path):
        raise AssertionError("auto_rotate=False — OSD не вызывается")

    result = ocr_document(
        pdf_path,
        tmp_path / "out.docx",
        predict_tables_fn=lambda _p: _load_simple_table_json(),
        image_data_fn=lambda _p: _tesseract_data([("Распознано", 90, 10, 5, 100, 20)]),
        auto_rotate=False,
        orientation_fn=_should_not_be_called,
        progress_fn=lambda *a: None,
    )
    assert result.pages[0].ocr_applied is True


def _has_tesseract() -> bool:
    import shutil

    from npa_ocr.ocr import _resolve_tesseract_cmd

    return bool(_resolve_tesseract_cmd() or shutil.which("tesseract"))


@pytest.mark.skipif(not _has_tesseract(), reason="нет Tesseract")
@pytest.mark.parametrize(
    "key, turned, expected",
    [
        ("228_p025", 90, 90),  # альбомная таблица на боку (Приказ №228)
        ("227_p009", 180, 180),  # перевёрнутая страница (Приказ №227)
        ("191_p015", 0, 0),  # обычная книжная страница с таблицей
    ],
)
def test_default_orientation_fn_on_real_pages(tmp_path: Path, key: str, turned: int, expected: int):
    """Страницы спайка npa-vlm-ocr хранятся уже выровненными — возвращаем им исходный поворот
    скана и проверяем, что OSD находит угол, который его исправляет."""
    from npa_ocr.ocr import _default_orientation_fn, rotate_page_image

    path = tmp_path / f"{key}.png"
    with Image.open(ROTATION_PAGES / f"{key}.png") as img:
        img.rotate(turned, expand=True).save(path)
    rotation = _default_orientation_fn(path)
    assert rotation == expected
    if rotation:
        rotate_page_image(path, rotation)
        with Image.open(path) as fixed, Image.open(ROTATION_PAGES / f"{key}.png") as orig:
            assert fixed.size == orig.size


# --- table_engine="vlm": таблицы через DeepSeek-OCR (vLLM) ---


def _pdf_from_page_image(path: Path, image: Path, dpi: int = 200) -> Path:
    """Скан без текстового слоя из PNG страницы спайка: при рендере в том же DPI — те же пиксели."""
    import fitz

    with Image.open(image) as img:
        w, h = img.size
    doc = fitz.open()
    page = doc.new_page(width=w * 72 / dpi, height=h * 72 / dpi)
    page.insert_image(page.rect, filename=str(image))
    doc.save(str(path))
    doc.close()
    return path


@pytest.mark.skipif(not _has_tesseract(), reason="нет Tesseract")
def test_ocr_document_vlm_engine_tables_and_text_without_duplicates(tmp_path: Path):
    """1430 стр. 8: таблицы — по ответу DeepSeek (тело без линий — её HTML как есть); строки тела
    ниже разлинованной шапки входят в рамку таблицы и не дублируются обычным текстом, а
    подзаголовок «II. При сбросе…» между таблицами остаётся текстом."""
    page_png = ROTATION_PAGES / "1430_p008.png"
    raw = (ROTATION_PAGES / "1430_p008.txt").read_text(encoding="utf-8")
    pdf_path = _pdf_from_page_image(tmp_path / "scan.pdf", page_png)
    words = _tesseract_data_multiline([
        ("Сверхмалые", 95, 190, 1700, 200, 30, 1),  # строка тела первой таблицы
        ("II.", 95, 400, 1840, 40, 30, 2),  # подзаголовок между таблицами
        ("При", 95, 450, 1840, 60, 30, 2),
    ])

    result = ocr_document(
        pdf_path,
        tmp_path / "out.docx",
        table_engine="vlm",
        vlm_chat_fn=lambda _img: raw,
        image_data_fn=lambda _p: words,
        auto_rotate=False,
        progress_fn=lambda *a: None,
    )
    blocks = result.pages[0].blocks
    tables = [b for b in blocks if isinstance(b, TableBlock)]
    texts = [b.text for b in blocks if isinstance(b, TextBlock)]
    assert len(tables) == 2
    assert any(c.text == "Сверхмалые - средние очистные сооружения" for c in tables[0].cells)
    assert not any("Сверхмалые" in t for t in texts)
    assert any(t.startswith("II.") for t in texts)
    steps = [s["step"] for s in result.timings["pages"]["1"]["steps"]]
    assert "таблицы: DeepSeek-OCR (vLLM)" in steps and "таблицы: сетка по линиям" in steps


@pytest.mark.skipif(not _has_tesseract(), reason="нет Tesseract")
def test_ocr_document_vlm_engine_rereads_doubtful_cells(tmp_path: Path):
    """565 стр. 7: сомнительные ячейки сетки дочитываются по кропу (vlm_ask_fn); принятый текст —
    в ячейке без жёлтой пометки, время — отдельным шагом timings."""
    page_png = ROTATION_PAGES / "565_p007.png"
    raw = (ROTATION_PAGES / "565_p007.txt").read_text(encoding="utf-8")
    pdf_path = _pdf_from_page_image(tmp_path / "scan.pdf", page_png)
    asked = []

    def ask(img, prompt):
        asked.append(prompt)
        return "мг/нм³", "stop", -0.05

    result = ocr_document(
        pdf_path,
        tmp_path / "out.docx",
        table_engine="vlm",
        vlm_chat_fn=lambda _img: raw,
        vlm_ask_fn=ask,
        image_data_fn=lambda _p: _tesseract_data([("Текст", 95, 10, 10, 100, 20)]),
        auto_rotate=False,
        progress_fn=lambda *a: None,
    )
    assert asked
    (table,) = [b for b in result.pages[0].blocks if isinstance(b, TableBlock)]
    reread = [c for c in table.cells if c.text == "мг/нм³"]
    assert reread and all(any(c.low_lines) for c in reread)
    steps = [s["step"] for s in result.timings["pages"]["1"]["steps"]]
    assert "таблицы: дочитка ячеек (vLLM)" in steps


def test_ocr_document_vlm_unavailable_falls_back_to_tatr_for_rest_of_document(tmp_path: Path, monkeypatch):
    """vLLM не ответил — документ не падает: таблицы через TATR, vLLM больше не спрашиваем
    (не ждём таймаут на каждой странице), причина — в шагах timings."""
    import fitz

    from npa_ocr import ocr as ocr_module
    from npa_ocr.vlm_tables import VlmUnavailable

    pdf_path = tmp_path / "scan.pdf"
    doc = fitz.open()
    doc.new_page()
    doc.new_page()
    doc.save(str(pdf_path))
    doc.close()

    class _Engine:
        def release(self):
            pass

    tatr_pages = []
    monkeypatch.setattr(ocr_module, "_load_tatr_engines", lambda: (_Engine(), _Engine()))
    monkeypatch.setattr(ocr_module, "run_tatr_tables", lambda path, *a, **kw: tatr_pages.append(path) or [])
    chat_calls = []

    def down(_img):
        chat_calls.append(1)
        raise VlmUnavailable("vLLM (http://vlm-gate:8000/v1) недоступен: 503")

    result = ocr_document(
        pdf_path,
        tmp_path / "out.docx",
        table_engine="vlm",
        vlm_chat_fn=down,
        image_data_fn=lambda _p: _tesseract_data([("Текст", 95, 10, 10, 100, 20)]),
        auto_rotate=False,
        progress_fn=lambda *a: None,
    )
    assert len(chat_calls) == 1
    assert len(tatr_pages) == 2
    assert all(p.ocr_applied for p in result.pages)
    steps = [s["step"] for s in result.timings["pages"]["1"]["steps"]]
    assert any(s.startswith("таблицы: vLLM недоступен, запасной путь TATR") for s in steps)

    # следующий прогон того же PDF (vLLM уже поднялся) не берёт страницы запасного пути из
    # сохранённых, а распознаёт их через vLLM
    chat_ok = []
    ocr_document(
        pdf_path,
        tmp_path / "out.docx",
        table_engine="vlm",
        vlm_chat_fn=lambda _img: chat_ok.append(1) or "",
        image_data_fn=lambda _p: _tesseract_data([("Текст", 95, 10, 10, 100, 20)]),
        auto_rotate=False,
        progress_fn=lambda *a: None,
    )
    assert len(chat_ok) == 2


MARKS_PAGES = Path(__file__).parent / "fixtures" / "vlm_marks"


def _marks_page_pdf(tmp_path: Path, key: str) -> tuple[Path, dict]:
    """Скан из белой страницы того же размера, что страница спайка (вид страницы не нужен: слова
    Tesseract, ответы DeepSeek и qwen3-vl — записанные фикстуры vlm_marks)."""
    data = json.loads((MARKS_PAGES / f"{key}.tess.json").read_text(encoding="utf-8"))
    png = tmp_path / "page.png"
    Image.new("L", (data["width"][0], data["height"][0]), 255).save(png)
    return _pdf_from_page_image(tmp_path / "scan.pdf", png), data


def test_ocr_document_vlm_engine_drops_marks_text(tmp_path: Path):
    """Приказ №215 стр. 1: текст штампа Минюста и подписи выкидывается, шапка под краем штампа
    дописывается по authority_name, фамилия под росчерком — по DeepSeek; время qwen3-vl — шагом."""
    pdf_path, data = _marks_page_pdf(tmp_path, "215_p001")
    ds = (MARKS_PAGES / "215_p001.ds.txt").read_text(encoding="utf-8")
    qwen = (MARKS_PAGES / "215_p001.qwen.txt").read_text(encoding="utf-8")
    result = ocr_document(
        pdf_path,
        tmp_path / "out.docx",
        table_engine="vlm",
        use_ai_escalation=True,
        authority_name="Министерство природных ресурсов и экологии Российской Федерации",
        vlm_chat_fn=lambda _img: ds,
        marks_ask_fn=lambda _img: qwen,
        image_data_fn=lambda _p: data,
        auto_rotate=False,
        progress_fn=lambda *a: None,
    )
    texts = [b.text for b in result.pages[0].blocks if isinstance(b, TextBlock)]
    assert "МИНИСТЕРСТВО ПРИРОДНЫХ РЕСУРСОВ И ЭКОЛОГИИ" in texts
    assert "Министр А.А. Козлов" in texts
    assert not [t for t in texts if "ЗАРЕГИСТРИРОВАНО" in t or "ЮСТИЦИИ" in t]
    header = texts.index("МИНИСТЕРСТВО ПРИРОДНЫХ РЕСУРСОВ И ЭКОЛОГИИ")
    assert result.pages[0].blocks[header].low_words[-2:] == (True, True)  # дописанное — на проверку
    steps = [s["step"] for s in result.timings["pages"]["1"]["steps"]]
    assert "метки: qwen3-vl (Ollama)" in steps


def test_ocr_document_marks_unavailable_keeps_text(tmp_path: Path):
    """Ollama не ответила — страница распознаётся без чистки меток, документ не падает."""
    from npa_ocr.vlm_marks import MarksUnavailable

    pdf_path, data = _marks_page_pdf(tmp_path, "215_p001")
    ds = (MARKS_PAGES / "215_p001.ds.txt").read_text(encoding="utf-8")

    def down(_img):
        raise MarksUnavailable("Ollama недоступна: 503")

    result = ocr_document(
        pdf_path,
        tmp_path / "out.docx",
        table_engine="vlm",
        use_ai_escalation=True,
        vlm_chat_fn=lambda _img: ds,
        marks_ask_fn=down,
        image_data_fn=lambda _p: data,
        auto_rotate=False,
        progress_fn=lambda *a: None,
    )
    texts = [b.text for b in result.pages[0].blocks if isinstance(b, TextBlock)]
    assert any("ЗАРЕГИСТРИРОВАНО" in t for t in texts)
    steps = [s["step"] for s in result.timings["pages"]["1"]["steps"]]
    assert any(s.startswith("метки: Ollama недоступна") for s in steps)


def test_vlm_table_exclusion_uses_grid_without_crop_margin(monkeypatch, tmp_path):
    from types import SimpleNamespace
    from npa_ocr import ocr

    cell = SimpleNamespace()
    table = SimpleNamespace(bbox=[100, 200, 500, 700], cells=[cell], n_rows=2, n_cols=2,
                            src="grid", info={"xe": [12, 200, 388], "ye": [12, 240, 488]})
    monkeypatch.setattr(ocr.vlm_tables, "run_vlm_tables", lambda *a, **k: [table])
    block = ocr.run_vlm_table_blocks(tmp_path / "page.png", lambda image: "")[0]
    assert block.bbox == (112, 212, 488, 688)
    assert block.y == 212


def test_docx_preserves_narrow_grid_subdivisions_before_merging():
    from docx import Document
    from npa_ocr.ocr import _add_table

    doc = Document()
    block = TableBlock(y=0, bbox=(0, 0, 100, 100), n_rows=1, n_cols=3,
                       column_edges=[0, .1, .12, 1],
                       cells=[TableCell(0, 0, 1, 2, "Название", []),
                              TableCell(0, 2, 1, 1, "20", [])])
    _add_table(doc, block)
    table = doc.tables[0]
    widths = [column.width for column in table.columns]
    assert .15 < widths[1] / widths[0] < .25
    assert widths[2] / widths[0] > 8
    assert [cell.text for cell in table.rows[0].cells] == ["Название", "Название", "20"]


def test_ocr_document_vlm_engine_rereads_formulas_and_variables(tmp_path: Path):
    """710 стр. 3: второе прочтение ответом DeepSeek на страницу (тот же, что для таблиц) — формула
    отдельным блоком, «статьи 163» → 16³, K с индексом; шаг в таймингах."""
    from npa_ocr.ocr import FormulaBlock

    pages = Path(__file__).parent / "fixtures" / "vlm_text"
    data = json.loads((pages / "710_p003.tess.json").read_text(encoding="utf-8"))
    ds = (pages / "710_p003.ds.txt").read_text(encoding="utf-8")
    png = tmp_path / "page.png"
    Image.new("L", (data["width"][0], data["height"][0]), 255).save(png)
    result = ocr_document(
        _pdf_from_page_image(tmp_path / "scan.pdf", png),
        tmp_path / "out.docx",
        table_engine="vlm",
        vlm_chat_fn=lambda _img: ds,
        image_data_fn=lambda _p: data,
        auto_rotate=False,
        progress_fn=lambda *a: None,
    )
    blocks = result.pages[0].blocks
    assert [b.latex for b in blocks if isinstance(b, FormulaBlock)] == [r"Z = \frac{S}{V} \times 100\%,"]
    texts = [b.text for b in blocks if isinstance(b, TextBlock)]
    assert "Федерации в соответствии с пунктом 4 статьи 16³ Федерального закона" in texts
    assert any("(К<sub>нмм</sub>)," in b.rich_words for b in blocks if isinstance(b, TextBlock))
    steps = [s["step"] for s in result.timings["pages"]["1"]["steps"]]
    assert "текст: второе прочтение DeepSeek" in steps
