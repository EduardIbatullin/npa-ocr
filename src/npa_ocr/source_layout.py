"""Source geometry checks independent of OCR confidence and document names.

Coordinates in the inventory are PDF points, never OCR image pixels.
"""
from __future__ import annotations


def inspect_source(page) -> dict:
    """Inventory visible raster regions and ruled tables, excluding page frames."""
    import pymupdf as fitz

    area = max(page.rect.get_area(), 1)
    tables = []
    for table in page.find_tables(strategy="lines_strict").tables:
        rect = fitz.Rect(table.bbox)
        cells = [fitz.Rect(c) for c in table.cells if c is not None]
        # Drawing borders with a small title block surround the whole text body.
        # Their huge merged body cell is not a data table; nested tables are
        # returned separately by the detector and remain in this inventory.
        is_frame = (rect.get_area() > .65 * area and len(cells) <= 8
                    and any(c.get_area() > .75 * rect.get_area() for c in cells))
        if not is_frame and table.row_count > 1 and table.col_count > 1:
            tables.append(dict(bbox=list(table.bbox), rows=table.row_count,
                               cols=table.col_count))
    images = []
    for info in page.get_image_info():
        rect = fitz.Rect(info["bbox"]) & page.rect
        if rect.is_empty or rect.get_area() < .015 * area:
            continue
        images.append(dict(bbox=list(rect), area_ratio=rect.get_area() / area))
    # Curves and line art have no raster image entry. A dense vector region
    # outside recognized tables must not silently disappear from the review.
    outside_paths = 0
    for drawing in page.get_drawings():
        rect = fitz.Rect(drawing['rect'])
        if not any(fitz.Rect(t['bbox']).contains(rect) for t in tables):
            outside_paths += 1
    return dict(width=page.rect.width, height=page.rect.height,
                tables=tables, images=images, vector_paths_outside_tables=outside_paths)


def source_review(page_number: int, source: dict, *, text_only: bool,
                  output_tables: int) -> list[dict]:
    """Report missing representation; a raster image is not necessarily text."""
    issues = []
    if source.get("tables") and output_tables == 0:
        issues.append(dict(page=page_number, reason="source_tables_without_output_tables",
                           source_regions=source["tables"]))
    if source.get("images"):
        issues.append(dict(page=page_number,
                           reason="raster_regions_skipped" if text_only
                           else "raster_graphics_preservation_unverified",
                           source_regions=source["images"]))
    if source.get('vector_paths_outside_tables', 0) >= 20:
        issues.append(dict(page=page_number, reason='vector_content_preservation_unverified',
                           path_count=source['vector_paths_outside_tables']))
    return issues


def extract_native_blocks(page, source: dict, dpi: int, *, allow_raster=False):
    """Reconstruct ruled native tables and retain text outside their cells.

    Return None for raster-heavy mixed pages: their text layer cannot justify
    bypassing OCR. No source text is interpreted as a corrected OCR reading.
    """
    import pymupdf as fitz
    from npa_ocr.ocr import TableBlock, TextBlock
    from npa_ocr.ocr_table import TableCell

    if not allow_raster and any(image["area_ratio"] >= .12 for image in source["images"]):
        return None
    scale = dpi / 72
    allowed = {tuple(t["bbox"]) for t in source["tables"]}
    tables = []
    occupied = []
    for table in page.find_tables(strategy="lines_strict").tables:
        if tuple(table.bbox) not in allowed:
            continue
        boxes = [fitz.Rect(c) for c in table.cells if c is not None]
        xs = sorted({round(v, 2) for c in boxes for v in (c.x0, c.x1)})
        ys = sorted({round(v, 2) for c in boxes for v in (c.y0, c.y1)})
        # Rows/columns inferred from shared geometry, including merged cells.
        # Reject inconsistent structures rather than silently shifting values.
        if len(xs) != table.col_count + 1 or len(ys) != table.row_count + 1:
            return None
        cells = []
        for rect in boxes:
            col = min(range(len(xs)), key=lambda i: abs(xs[i] - rect.x0))
            row = min(range(len(ys)), key=lambda i: abs(ys[i] - rect.y0))
            right = min(range(len(xs)), key=lambda i: abs(xs[i] - rect.x1))
            bottom = min(range(len(ys)), key=lambda i: abs(ys[i] - rect.y1))
            text = page.get_textbox(rect).strip()
            cells.append(TableCell(row, col, bottom-row, right-col, text))
            occupied.append(rect)
        bbox = tuple(v * scale for v in table.bbox)
        tables.append(TableBlock(bbox[1], bbox, cells, table.row_count, table.col_count,
                                column_edges=[(x-xs[0])/(xs[-1]-xs[0]) for x in xs]))
    blocks = list(tables)
    for block in page.get_text("dict")["blocks"]:
        for line in block.get("lines", []):
            spans = []
            for span in line["spans"]:
                rect = fitz.Rect(span["bbox"])
                centre = (rect.tl + rect.br) / 2
                if not any(centre in cell for cell in occupied):
                    spans.append(span)
            if not spans:
                continue
            text = "".join(s["text"] for s in spans).strip()
            if text:
                rect = fitz.Rect(spans[0]["bbox"])
                for span in spans[1:]:
                    rect |= fitz.Rect(span["bbox"])
                blocks.append(TextBlock(rect.y0*scale, text, rect.x0*scale,
                                        rect.x1*scale, rect.height*scale))
    return sorted(blocks, key=lambda b: b.y)


def merge_mixed_blocks(native_blocks, ocr_blocks, source: dict, dpi: int):
    """Keep native content outside large raster inserts, OCR inside them.

    The caller must use unrotated PDF image coordinates. Raster preservation
    remains unverified: OCR text is not a replacement for diagrams or signatures.
    """
    import pymupdf as fitz
    from npa_ocr.ocr import TableBlock, FormulaBlock

    scale = dpi / 72
    regions = [fitz.Rect([v * scale for v in item['bbox']])
               for item in source['images'] if item['area_ratio'] >= .12]

    def inside(block):
        if isinstance(block, (TableBlock, FormulaBlock)):
            rect = fitz.Rect(block.bbox)
            return any((rect & region).get_area() > .5 * max(rect.get_area(), 1)
                       for region in regions)
        point = fitz.Point((block.x0+block.x1)/2, block.y+block.height/2)
        return any(point in region for region in regions)

    return sorted([b for b in native_blocks if not inside(b)]
                  + [b for b in ocr_blocks if inside(b)], key=lambda b: b.y)
