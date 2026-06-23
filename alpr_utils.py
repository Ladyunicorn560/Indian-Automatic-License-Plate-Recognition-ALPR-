import cv2
import numpy as np
import statistics
import os
from datetime import datetime
from pathlib import Path

def get_conf(ocr_result) -> float:
    """Always returns a plain float, whether confidence is a list or scalar."""
    if ocr_result is None: return 0.0
    c = ocr_result.confidence
    if isinstance(c, (list, tuple)):
        return float(statistics.mean(c)) if c else 0.0
    return float(c)

def draw_plates_no_country(img: np.ndarray, results) -> np.ndarray:
    """
    Re-draws bounding boxes + plate text + confidence, but SKIPS the country/region label.
    """
    annotated = img.copy()
    for result in results:
        detection = result.detection
        ocr_result = result.ocr
        bbox = detection.bounding_box
        x1, y1, x2, y2 = int(bbox.x1), int(bbox.y1), int(bbox.x2), int(bbox.y2)

        cv2.rectangle(annotated, (x1, y1), (x2, y2), (36, 255, 12), 2)

        if ocr_result is None or not ocr_result.text:
            continue

        conf = get_conf(ocr_result)
        font_scale = min(1.25, max(0.4, annotated.shape[1] / 1000))
        text_thickness = 1 if font_scale < 0.75 else 2
        outline_thickness = text_thickness + max(3, round(font_scale * 3))

        line = f"{ocr_result.text}  {conf * 100:.0f}%"

        size, _ = cv2.getTextSize(line, cv2.FONT_HERSHEY_SIMPLEX, font_scale, text_thickness)
        text_width, text_height = size
        
        text_y = y1 - 10
        if text_y - text_height < 0:
            text_y = y2 + text_height + 10

        text_x = min(max(x1, 5), max(5, annotated.shape[1] - text_width - 5))
        text_y = min(max(text_y, text_height + 5), annotated.shape[0] - 5)

        cv2.putText(annotated, line, (text_x, text_y),
                    cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), outline_thickness, cv2.LINE_AA)
        cv2.putText(annotated, line, (text_x, text_y),
                    cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), text_thickness, cv2.LINE_AA)
    return annotated

def get_plate_crop(img: np.ndarray, bbox, padding: int = 15) -> np.ndarray:
    """Extract a cropped plate from the frame with extra padding."""
    h, w = img.shape[:2]
    x1 = max(0, int(bbox.x1) - padding)
    y1 = max(0, int(bbox.y1) - padding)
    x2 = min(w, int(bbox.x2) + padding)
    y2 = min(h, int(bbox.y2) + padding)
    return img[y1:y2, x1:x2]

def clean_indian_plate(text: str) -> str:
    import re
    text = text.upper().strip()
    text = re.sub(r"[\s\-_]", "", text)
    text = re.sub(r"[^A-Z0-9]", "", text)

    # ── Indian Plate Format Correction ──────────────────────────────────────
    # OCR models often confuse visually similar characters at plate edges.
    # We apply format-aware corrections based on known Indian plate schemas.
    #
    # Common substitutions:
    #  digit-for-letter: 0->O, 1->I, 8->B, 6->G, 5->S, 4->A
    #  letter-for-digit: O->0, I->1, S->5, B->8 (less common, ignored here)
    #  Added Q→0, D→0, U→0 (round chars → zero), L→1 (thin char → one)

    _D2L = {"0": "O", "1": "I", "8": "B", "6": "G", "5": "S", "4": "A"}
    _L2D = {
        "O": "0", "Q": "0", "D": "0", "U": "0",   # round chars → zero
        "I": "1", "L": "1",                          # thin chars → one
        "S": "5", "B": "8", "G": "6", "Z": "2", "A": "4"
    }

    # ── BH-Series Positional Correction (e.g. ZA8HOZ688 -> 24BH0268B) ───────
    if len(text) in (9, 10):
        # Position 0, 1 must be digits (Year)
        y0 = _L2D.get(text[0], text[0])
        y1 = _L2D.get(text[1], text[1])
        # Position 2, 3 must be letters ("BH")
        s0 = _D2L.get(text[2], text[2])
        s1 = _D2L.get(text[3], text[3])
        
        if s0 == "B" and s1 == "H" and y0.isdigit() and y1.isdigit():
            # Position 4, 5, 6, 7 must be digits (Number)
            n0 = _L2D.get(text[4], text[4])
            n1 = _L2D.get(text[5], text[5])
            n2 = _L2D.get(text[6], text[6])
            n3 = _L2D.get(text[7], text[7])
            # Position 8+ must be letters (Suffix)
            suffix = "".join(_D2L.get(c, c) for c in text[8:])
            
            corrected_bh = y0 + y1 + "BH" + n0 + n1 + n2 + n3 + suffix
            if len(corrected_bh) == 9 and re.match(r'^\d{2}BH\d{4}[A-Z]$', corrected_bh):
                return corrected_bh
            if len(corrected_bh) == 10 and re.match(r'^\d{2}BH\d{4}[A-Z]{2}$', corrected_bh):
                return corrected_bh

    def _fix(c, must_be_letter: bool):
        if must_be_letter and c.isdigit():
            return _D2L.get(c, c)
        return c

    # ── BH-Series (Legacy Fallbacks) ──────────────────────────────────────────
    # Format: 2 digits + BH + 4 digits + 1 letter  (total 9 chars)
    bh = re.match(r'^(\d{2})(BH)(\d{4})([A-Z0-9])$', text)
    if bh:
        suffix = _fix(bh.group(4), must_be_letter=True)
        return bh.group(1) + bh.group(2) + bh.group(3) + suffix

    # Also catch the case where the 8th char is the suffix misread
    # e.g. "24BH02688" → last char '8' must be letter → 'B'
    bh8 = re.match(r'^(\d{2})(BH)(\d{4})(\d)$', text)
    if bh8:
        suffix = _D2L.get(bh8.group(4), bh8.group(4))
        return bh8.group(1) + bh8.group(2) + bh8.group(3) + suffix

    # ── Standard State Plates ────────────────────────────────────────────────
    # Format typically: 2 Letters (State) + 2 Digits (District) + 1-3 Letters + 1-4 Digits
    # We forcefully correct the first 4 characters because their types are strictly known.
    if len(text) >= 6:
        # 1. State code (first 2 chars) must be letters
        state = "".join(_D2L.get(c, c) for c in text[:2])
        
        # 2. District code: first character (index 2) must be a digit.
        # The second character (index 3) can be a digit or a letter (supporting single-digit district codes like DL 5S...).
        dist_char2 = _L2D.get(text[2], text[2])
        dist_char3 = text[3]
        dist = dist_char2 + dist_char3
        
        rest = text[4:]
        
        # Attempt to correct the tail end (numbers)
        # Indian plates usually end with 1 to 4 digits. If we see common letter-for-digit
        # mistakes at the very end, we can fix them.
        match = re.match(r'^([A-Z]{2})(\d{2})([A-Z]{1,3})([A-Z0-9]{1,4})$', state + dist + rest)
        if match:
            s, d, series, num = match.groups()
            # If the last part has a mix of letters and digits, or is at the end, 
            # let's try to enforce digits on `num` if it looks like a number block.
            # We'll just enforce the `dist` and `state` for now as it's the safest bet without
            # accidentally ruining the series letters (like 'S' vs '5').
            pass
            
        text = state + dist + rest

    return text

def save_output_image(annotated: np.ndarray, output_dir: Path, prefix: str = "capture") -> Path:
    """Save annotated image atomically (write to temp, then rename) to avoid
    Windows file-locking errors when the Explorer tries to open the file mid-write."""
    ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:20]
    filename = output_dir / f"{prefix}_{ts}.jpg"
    tmp_filename = output_dir / f".tmp_{prefix}_{ts}.jpg"
    cv2.imwrite(str(tmp_filename), annotated)
    os.replace(str(tmp_filename), str(filename))  # atomic on Windows (same drive)
    return filename
