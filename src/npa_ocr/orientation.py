"""Confirm an OSD proposal by reading the same page in both orientations."""
from dataclasses import dataclass, asdict


@dataclass(frozen=True)
class Readability:
    confident_words: int = 0
    confident_letters: int = 0
    score: float = 0.0


def readability(data: dict) -> Readability:
    words = letters = 0
    score = 0.0
    for text, confidence in zip(data.get('text', []), data.get('conf', [])):
        text = str(text).strip()
        count = sum(c.isalpha() for c in text)
        if count < 2 or count < .55 * len(text):
            continue
        try:
            confidence = float(confidence)
        except (TypeError, ValueError):
            continue
        if confidence >= 50:
            words += 1
            letters += count
        score += max(0, confidence - 40) * count
    return Readability(words, letters, score)


class ConfirmedRotation(int):
    """Integer-compatible result; rejected proposals remain reviewable."""
    def __new__(cls, rotation, *, issue=None):
        result = super().__new__(cls, rotation)
        result.review_issue = issue
        return result


def confirm_rotation(proposed: int, before: Readability, after: Readability):
    if not proposed:
        return ConfirmedRotation(0)
    if (after.confident_words >= 6 and after.score >= before.score * 1.25
            and after.score - before.score >= 500):
        return ConfirmedRotation(proposed)
    return ConfirmedRotation(0, issue=dict(reason='orientation_proposal_unconfirmed',
        proposed_rotation=proposed, before=asdict(before), after=asdict(after)))


def dominant_table_box(image):
    """Find a large connected ruled grid; an outer page frame is insufficient."""
    try:
        import cv2
        import numpy as np
        from npa_ocr.vlm_tables import _line_masks
    except ImportError:
        return None
    sample=image.convert('L')
    sample.thumbnail((1600,1600))
    array=np.array(sample)
    horizontal,vertical=_line_masks(array)
    _,labels,stats,_=cv2.connectedComponentsWithStats(horizontal|vertical)
    def runs(indices):
        return 0 if not len(indices) else 1+int((np.diff(indices)>1).sum())
    components=enumerate(stats[1:],1)
    for label,(x,y,w,h,_) in sorted(components,
            key=lambda pair:int(pair[1][2])*int(pair[1][3]),reverse=True):
        if w*h < .15*sample.width*sample.height:
            continue
        component=labels[y:y+h,x:x+w]==label
        rows=np.flatnonzero(((horizontal[y:y+h,x:x+w]>0)&component).sum(axis=1)>.6*w)
        cols=np.flatnonzero(((vertical[y:y+h,x:x+w]>0)&component).sum(axis=0)>.6*h)
        if runs(rows)>=3 and runs(cols)>=3:
            sx,sy=image.width/sample.width,image.height/sample.height
            return (int(x*sx),int(y*sy),int((x+w)*sx),int((y+h)*sy))
    return None


def check_osd_proposal(image, proposed: int, image_data_fn, *, table_box=None):
    tall_grid=(table_box is not None and
               table_box[3]-table_box[1] > 1.2*(table_box[2]-table_box[0]))
    if not proposed and not tall_grid:
        return ConfirmedRotation(0)
    sample = image.crop(table_box) if table_box else image.copy()
    sample.thumbnail((2000, 2000))
    before = readability(image_data_fn(sample))
    # A readable OSD proposal has priority. Search other axes only when OSD
    # supplied no rotation: short chemical tokens can favour a wrong axis.
    candidates=(90,180,270) if tall_grid and not proposed else (proposed,)
    readings=[(angle,readability(image_data_fn(sample.rotate(-angle,expand=True))))
              for angle in candidates]
    if not proposed:
        # Rotated chemical symbols often become numerous confident two-letter
        # fragments. They cannot establish a new axis without longer words.
        eligible=[pair for pair in readings if pair[1].confident_words>=6
                  and pair[1].confident_letters>=3*pair[1].confident_words]
        if not eligible:
            issue=(dict(reason='orientation_unresolved',osd_rotation=0,region_bbox=table_box,
                        before=asdict(before)) if before.confident_words<6 else None)
            return ConfirmedRotation(0,issue=issue)
        readings=eligible
    angle,after=max(readings,key=lambda pair:pair[1].score)
    result=confirm_rotation(angle,before,after)
    if result.review_issue is not None:
        if proposed==0 and before.confident_words>=6:
            return ConfirmedRotation(0)
        result.review_issue.update(osd_rotation=proposed,region_bbox=table_box)
    return result
