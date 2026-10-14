"""CLI: `npa-ocr <pdf> [опции]` — один скан в DOCX.

Выделено из npa-pipeline (2026-10-12) в отдельный проект, чтобы OCR можно было
использовать в связке с другими проектами, а не только с загрузчиком НПА. Внешний
контракт (аргументы, JSON в stdout, строки прогресса и тайминги в stderr) не
менялся — см. docs/plan.md. Другим проектам, которым нужно встроить OCR как
фоновое задание (с опросом прогресса), читать этот контракт напрямую необязательно
— см. HTTP API (README, npa_ocr.server), которое делает то же самое поверх сети.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from npa_ocr.ocr import (
    DEFAULT_CONFIDENCE_THRESHOLD,
    DEFAULT_DPI,
    DEFAULT_MIN_FREE_MEMORY_MB,
    InsufficientMemoryError,
    ocr_document,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="npa-ocr",
        description="Распознать скан без текстового слоя (OCR) и собрать DOCX",
    )
    parser.add_argument("pdf", help="Путь к PDF")
    parser.add_argument("--out", help="Путь к выходному DOCX (по умолчанию <pdf>.docx)")
    parser.add_argument(
        "--authority-name",
        help=(
            "Орган-подписант (например «Правительство Российской Федерации») — "
            "если задан, повреждённая печатью формула должности подписанта "
            "восстанавливается; ФИО подписанта не восстанавливается никогда"
        ),
    )
    parser.add_argument("--dpi", type=int, default=DEFAULT_DPI, help="DPI рендера страниц-сканов")
    parser.add_argument(
        "--confidence-threshold",
        type=int,
        default=DEFAULT_CONFIDENCE_THRESHOLD,
        help="Порог confidence Tesseract, ниже которого слово помечается [нрзб.]",
    )
    parser.add_argument(
        "--min-free-mb",
        type=int,
        default=DEFAULT_MIN_FREE_MEMORY_MB,
        help="Минимум свободной памяти (МБ) перед каждой страницей-сканом; 0 — отключить проверку",
    )
    parser.add_argument(
        "--pages-dir",
        help="Папка для постраничных результатов (по умолчанию <out>.pages)",
    )
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="Распознать все страницы заново, даже если они уже сохранены в --pages-dir",
    )
    return parser


def cmd_ocr(args: argparse.Namespace) -> int:
    out_path = Path(args.out) if args.out else Path(args.pdf).with_suffix(".docx")
    try:
        result = ocr_document(
            args.pdf,
            out_path,
            dpi=args.dpi,
            confidence_threshold=args.confidence_threshold,
            authority_name=args.authority_name,
            min_free_memory_mb=args.min_free_mb,
            pages_dir=args.pages_dir,
            resume=not args.no_resume,
        )
    except InsufficientMemoryError as exc:
        print(json.dumps({"status": "insufficient_memory", "message": str(exc)}, ensure_ascii=False, indent=2))
        return 1
    pages_ocr_applied = sum(1 for p in result.pages if p.ocr_applied)
    print(
        json.dumps(
            {
                "docx": str(result.docx_path),
                "pages": len(result.pages),
                "pages_ocr_applied": pages_ocr_applied,
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    sys.exit(cmd_ocr(args))


if __name__ == "__main__":
    main()
