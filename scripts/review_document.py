"""Воспроизводимый прогон PDF с сохранением запросов и ответов OCR-моделей."""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import asdict
from pathlib import Path
from threading import Lock

from PIL import Image

from npa_ocr import vlm_marks, vlm_reread, vlm_tables


def main() -> None:
    """Запускает проверку таблиц выбранных страниц или полный OCR документа."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("pdf", type=Path)
    parser.add_argument("out", type=Path)
    parser.add_argument("--pages", default="")
    parser.add_argument("--tables-only", action="store_true")
    parser.add_argument("--resume", action="store_true", help="Reuse compatible saved pages; remove a page JSON to repeat it")
    parser.add_argument("--diagnostic", action="store_true", help="Include PDF source captions")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    cache = args.out / "calls"
    cache.mkdir(exist_ok=True)
    locks: dict[str, Lock] = {}
    locks_guard = Lock()

    def cached(image: Image.Image, prompt: str, call):
        """Повторно использует сохранённый ответ для идентичного изображения."""
        key = hashlib.sha256(f"{image.mode}{image.size}{prompt}".encode() + image.tobytes()).hexdigest()
        with locks_guard:
            lock = locks.setdefault(key, Lock())
        with lock:
            record = cache / f"{key}.json"
            if record.exists():
                return json.loads(record.read_text(encoding="utf-8"))["response"]
            response = call()
            image.save(cache / f"{key}.png")
            record.write_text(json.dumps({"prompt": prompt, "response": response}, ensure_ascii=False), encoding="utf-8")
            return response

    def chat(image):
        """Ответ по странице или фрагменту таблицы."""
        return cached(image, "page", lambda: vlm_tables.deepseek_chat(image))

    def ask(image, prompt):
        """Ответ по ячейке вместе с причиной остановки и logprob."""
        return tuple(cached(image, prompt, lambda: vlm_reread.deepseek_ask(image, prompt)))

    if args.tables_only:
        import pymupdf

        with pymupdf.open(args.pdf) as pdf:
            selected = [int(n) for n in args.pages.split(",") if n] or range(1, len(pdf) + 1)
            for number in selected:
                image_path = args.out / f"page_{number:04}.png"
                pdf[number - 1].get_pixmap(matrix=pymupdf.Matrix(200 / 72, 200 / 72)).save(image_path)
                print(f"page {number}: starting", flush=True)
                tables = vlm_tables.run_vlm_tables(image_path, chat_fn=chat,
                    reread_fn=lambda image, doubt: vlm_reread.reread_cells(image, doubt, ask))
                (args.out / f"tables_{number:04}.json").write_text(
                    json.dumps([asdict(t) for t in tables], ensure_ascii=False), encoding="utf-8")
                print(f"page {number}: {len(tables)} tables", flush=True)
    else:
        from npa_ocr.ocr import ocr_document

        result = ocr_document(args.pdf, args.out / "result.docx", table_engine="vlm",
                              vlm_chat_fn=chat, vlm_ask_fn=ask, use_ai_escalation=True, resume=args.resume,
                              diagnostic=args.diagnostic,
                              requisites_ask_fn=lambda image,prompt: cached(image,"qwen:"+prompt,
                                  lambda: vlm_marks.qwen_read(image,prompt)),
                              marks_ask_fn=lambda image: cached(image, "marks", lambda: vlm_marks.qwen_locate(image)))
        print(result.docx_path, flush=True)


if __name__ == "__main__":
    main()
