"""Conservative detection of a single faint dash in an otherwise empty cell."""
from __future__ import annotations


def faint_dash(gray) -> bool:
    import cv2
    import numpy as np

    if gray.size == 0 or min(gray.shape) < 4:
        return False
    height,width=gray.shape
    evidence=[]
    for threshold in (180,210):
        _,_,stats,_=cv2.connectedComponentsWithStats((gray<threshold).astype(np.uint8))
        # Fragments near the side rules are not cell contents.
        content=[s for s in stats[1:] if s[4]>=5 and s[0]>.08*width and s[0]+s[2]<.92*width]
        marks=[s for s in content if 6<=s[2]<=min(45,.6*width) and 1<=s[3]<=8
               and s[2]>=3*s[3] and s[4]>=s[2] and .25*width<=s[0]+s[2]/2<=.75*width]
        if len(marks)!=1:
            return False
        mark=marks[0]
        if any(s is not mark and s[4]>=max(8,.2*mark[4]) for s in content):
            return False
        evidence.append(mark)
    return abs(int(evidence[0][0])-int(evidence[1][0]))<=3 and abs(int(evidence[0][1])-int(evidence[1][1]))<=2


def remove_broken_side_rule(image):
    """Erase vertically aligned fragments at the far right, not baseline dots."""
    import cv2
    import numpy as np
    from PIL import Image
    gray=np.array(image.convert('L'))
    height,width=gray.shape
    _,_,stats,_=cv2.connectedComponentsWithStats((gray<210).astype(np.uint8))
    pieces=[s for s in stats[1:] if s[4]>=2 and s[2]<=3 and s[0]>=.85*width]
    for piece in pieces:
        group=[s for s in pieces if abs(int(s[0])-int(piece[0]))<=3]
        if len(group)>=2 and max(s[1]+s[3] for s in group)-min(s[1] for s in group)>=.5*height:
            left=max(0,min(int(s[0]) for s in group)-2)
            right=min(width,max(int(s[0]+s[2]) for s in group)+2)
            gray[:,left:right]=255
    return Image.fromarray(gray).convert('RGB')
