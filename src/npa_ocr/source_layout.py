"""Source geometry checks independent of OCR confidence and document names.

Coordinates in the inventory are PDF points, never OCR image pixels.
"""
from __future__ import annotations


def _ruled_tables(page):
    """Use the line finder consistently across PyMuPDF 1.x and 2.x.

    PyMuPDF 2.0 enables a layout hook and disables quad corrections on import.
    The former can add empty columns; the latter separates decimal points.
    See upstream issue https://github.com/pymupdf/PyMuPDF/issues/5198.
    Restore the caller's global setting after the finder captures characters.
    """
    import inspect
    import pymupdf as fitz
    import pymupdf.table as table_api

    options = dict(strategy='lines_strict')
    if 'use_layout' in inspect.signature(table_api.find_tables).parameters:
        options['use_layout'] = False
    toggle = getattr(fitz.TOOLS, 'unset_quad_corrections', None)
    previous = toggle() if toggle else None
    try:
        if toggle:
            toggle(False)
        return page.find_tables(**options)
    finally:
        if toggle:
            toggle(previous)


def inspect_source(page) -> dict:
    """Inventory visible raster regions and ruled tables, excluding page frames."""
    import pymupdf as fitz
    from collections import Counter

    area = max(page.rect.get_area(), 1)
    tables = []
    for table in _ruled_tables(page).tables:
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
        rect = (fitz.Rect(info["bbox"]) * page.rotation_matrix) & page.rect
        if rect.is_empty or rect.get_area() < .015 * area:
            continue
        images.append(dict(bbox=list(rect), area_ratio=rect.get_area() / area))
    # Curves and line art have no raster image entry. A dense vector region
    # outside recognized tables must not silently disappear from the review.
    outside_paths = 0
    for drawing in page.get_drawings():
        rect = fitz.Rect(drawing['rect']) * page.rotation_matrix
        if not any(fitz.Rect(t['bbox']).contains(rect) for t in tables):
            outside_paths += 1
    return dict(width=page.rect.width, height=page.rect.height, rotation=page.rotation,
                tables=tables, images=images, vector_paths_outside_tables=outside_paths,
                native_characters=dict(Counter(c for c in page.get_text() if not c.isspace())))


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
    if source.get('rotation') and source.get('tables'):
        issues.append(dict(page=page_number, reason='rotated_native_layout_unverified',
                           rotation=source['rotation']))
    return issues


def extract_native_blocks(page, source: dict, dpi: int, *, allow_raster=False):
    """Reconstruct ruled native tables and retain text outside their cells.

    Return None for raster-heavy mixed pages: their text layer cannot justify
    bypassing OCR. No source text is interpreted as a corrected OCR reading.
    """
    import pymupdf as fitz
    from npa_ocr.ocr import TableBlock, TextBlock
    from npa_ocr.ocr_table import TableCell

    # find_tables uses displayed coordinates, whereas get_text uses unrotated
    # coordinates. Mixing them duplicates text and can transpose the grid.
    # Until explicit transforms are validated, route rotated layouts to OCR.
    if page.rotation:
        return None if (source['tables'] or source['images']) else []
    if not allow_raster and any(image["area_ratio"] >= .12 for image in source["images"]):
        return None
    scale = dpi / 72
    allowed = {tuple(t["bbox"]) for t in source["tables"]}
    tables = []
    occupied = []
    for table in _ruled_tables(page).tables:
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
        contents = table.extract()
        for rect in boxes:
            col = min(range(len(xs)), key=lambda i: abs(xs[i] - rect.x0))
            row = min(range(len(ys)), key=lambda i: abs(ys[i] - rect.y0))
            right = min(range(len(xs)), key=lambda i: abs(xs[i] - rect.x1))
            bottom = min(range(len(ys)), key=lambda i: abs(ys[i] - rect.y1))
            # get_textbox includes characters merely intersecting the clip,
            # which duplicates neighbouring rows in tightly ruled tables.
            # Table.extract assigns characters to a cell by their centres.
            text = (contents[row][col] or '').strip()
            cells.append(TableCell(row, col, bottom-row, right-col, text))
            occupied.append(rect)
        bbox = tuple(v * scale for v in table.bbox)
        tables.append(TableBlock(bbox[1], bbox, cells, table.row_count, table.col_count,
                                column_edges=[(x-xs[0])/(xs[-1]-xs[0]) for x in xs],
                                source_pages=[page.number + 1]))
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

    Native blocks and raster regions must share the OCR canvas coordinates.
    Use rotate_mixed_layout after rotating the image. Raster preservation
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


def rotate_mixed_layout(blocks, source: dict, dpi: int, rotation: int, width, height):
    """Map native blocks and raster regions into the rotated OCR canvas."""
    from dataclasses import replace
    from npa_ocr.ocr import TextBlock

    def box(rect):
        x0,y0,x1,y1=rect
        if rotation==90:return (height-y1,x0,height-y0,x1)
        if rotation==180:return (width-x1,height-y1,width-x0,height-y0)
        if rotation==270:return (y0,width-x1,y1,width-x0)
        return (x0,y0,x1,y1)
    transformed=[]
    for block in blocks:
        if isinstance(block,TextBlock):
            x0,y0,x1,y1=box((block.x0,block.y,block.x1,block.y+block.height))
            transformed.append(replace(block,x0=x0,x1=x1,y=y0,height=y1-y0))
        else:
            rect=box(block.bbox)
            transformed.append(replace(block,y=rect[1],bbox=rect))
    scale=dpi/72
    images=[dict(image,bbox=[v/scale for v in box([c*scale for c in image['bbox']])])
            for image in source['images']]
    return transformed,dict(source,images=images)
