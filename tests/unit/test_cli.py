"""Внешний контракт `npa-ocr`: аргументы и код/JSON при нехватке памяти.
Другие проекты запускают эту команду отдельным процессом и разбирают именно
этот вывод — контракт нельзя менять молча."""

from __future__ import annotations

import json

import pytest

from npa_ocr.cli import build_parser, cmd_ocr
from npa_ocr.ocr import DEFAULT_CONFIDENCE_THRESHOLD, DEFAULT_DPI, DEFAULT_MIN_FREE_MEMORY_MB, InsufficientMemoryError


def test_parser_defaults_match_ocr_document_defaults():
    args = build_parser().parse_args(["scan.pdf"])
    assert args.pdf == "scan.pdf"
    assert args.out is None
    assert args.dpi == DEFAULT_DPI
    assert args.confidence_threshold == DEFAULT_CONFIDENCE_THRESHOLD
    assert args.min_free_mb == DEFAULT_MIN_FREE_MEMORY_MB
    assert args.pages_dir is None
    assert args.no_resume is False


def test_parser_accepts_all_flags():
    args = build_parser().parse_args(
        [
            "scan.pdf",
            "--out",
            "out.docx",
            "--authority-name",
            "Правительство Российской Федерации",
            "--dpi",
            "150",
            "--confidence-threshold",
            "70",
            "--min-free-mb",
            "0",
            "--pages-dir",
            "pages",
            "--no-resume",
        ]
    )
    assert args.out == "out.docx"
    assert args.authority_name == "Правительство Российской Федерации"
    assert args.dpi == 150
    assert args.confidence_threshold == 70
    assert args.min_free_mb == 0
    assert args.pages_dir == "pages"
    assert args.no_resume is True


def test_cmd_ocr_reports_insufficient_memory_as_json_on_stdout(monkeypatch, capsys):
    def _raise(*a, **kw):
        raise InsufficientMemoryError("Свободной памяти 100 МБ, нужно не меньше 2048 МБ")

    monkeypatch.setattr("npa_ocr.cli.ocr_document", _raise)
    args = build_parser().parse_args(["scan.pdf"])
    code = cmd_ocr(args)
    assert code == 1
    out = json.loads(capsys.readouterr().out)
    assert out["status"] == "insufficient_memory"
    assert "МБ" in out["message"]
