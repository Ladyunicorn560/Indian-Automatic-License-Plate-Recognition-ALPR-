"""
Indian ALPR core module.
"""

import os
import re
import statistics
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

import cv2
import numpy as np
import onnxruntime as ort
from fast_plate_ocr.inference.hub import OcrModel
from open_image_models.detection.core.hub import PlateDetectorModel

from fast_alpr.base import BaseDetector, BaseOCR, DetectionResult, OcrResult
from fast_alpr.default_detector import DefaultDetector
from fast_alpr.default_ocr import DefaultOCR


def correct_indian_plate(plate: str) -> str:
    """
    Correct common OCR errors in Indian license plates using rule-based post-processing.

    Indian plate format: XX 00 XX 0000
    e.g., BR 01 ES 5493, MH 02 AB 1234

    This function fixes character substitutions based on position:
    - Positions 0,1,4,5 must be letters
    - Positions 2,3,6,7,8,9 must be digits

    Parameters:
        plate: Raw OCR output text

    Returns:
        Corrected plate text
    """
    if not plate:
        return plate

    plate = plate.upper().replace(" ", "")

    # Character substitution maps for common OCR confusions
    # Added Q→0, D→0, U→0 (round chars → zero), L→1 (thin char → one)
    letter_to_digit = {
        'O': '0', 'Q': '0', 'D': '0', 'U': '0',   # round chars → zero
        'I': '1', 'L': '1',                          # thin chars → one
        'Z': '2',
        'S': '5',
        'B': '8',
        'G': '6',
    }
    digit_to_letter = {'0': 'O', '1': 'I', '2': 'Z', '5': 'S', '8': 'B', '6': 'G'}

    corrected = list(plate)

    # Only apply positional correction to standard 10-character plates
    if len(corrected) == 10:
        for i, ch in enumerate(corrected):
            if i in (0, 1, 4, 5):  # Must be letters
                corrected[i] = digit_to_letter.get(ch, ch)
            elif i in (2, 6, 7, 8, 9):  # Must be digits (excluding index 3, which can be a letter or digit)
                corrected[i] = letter_to_digit.get(ch, ch)

    return "".join(corrected)


# pylint: disable=too-many-arguments, too-many-locals
# ruff: noqa: PLR0913


@dataclass(frozen=True)
class ALPRResult:
    """
    Detection and OCR output for one license plate.

    Attributes:
        detection: Detector output for the plate.
        ocr: OCR output for the plate, or None if OCR does not return a result.
    """

    detection: DetectionResult
    ocr: OcrResult | None


@dataclass(frozen=True, slots=True)
class DrawPredictionsResult:
    """
    Return value from draw_predictions.

    Attributes:
        image: The input image with boxes and text drawn on it.
        results: The ALPR results used to draw the annotations.
    """

    image: np.ndarray
    results: list[ALPRResult]


class ALPR:
    """
    Automatic License Plate Recognition (ALPR) system class.

    This class combines a detector and an OCR model to recognize license plates in images.
    """

    def __init__(
        self,
        detector: BaseDetector | None = None,
        ocr: BaseOCR | None = None,
        detector_model: PlateDetectorModel = "yolo-v9-t-384-license-plate-end2end",
        detector_conf_thresh: float = 0.4,
        detector_providers: Sequence[str | tuple[str, dict]] | None = None,
        detector_sess_options: ort.SessionOptions = None,
        ocr_model: OcrModel | None = "cct-xs-v2-global-model",
        ocr_device: Literal["cuda", "cpu", "auto"] = "auto",
        ocr_providers: Sequence[str | tuple[str, dict]] | None = None,
        ocr_sess_options: ort.SessionOptions | None = None,
        ocr_model_path: str | os.PathLike | None = None,
        ocr_config_path: str | os.PathLike | None = None,
        ocr_force_download: bool = False,
        correct_indian_plates: bool = True,
    ) -> None:
        """
        Initialize the ALPR system.

        Parameters:
            detector: An instance of BaseDetector. If None, the DefaultDetector is used.
            ocr: An instance of BaseOCR. If None, the DefaultOCR is used.
            detector_model: The name of the detector model or a PlateDetectorModel enum instance.
                Defaults to "yolo-v9-t-384-license-plate-end2end".
            detector_conf_thresh: Confidence threshold for the detector.
            detector_providers: Execution providers for the detector.
            detector_sess_options: Session options for the detector.
            ocr_model: The name of the OCR model from the model hub. This can be none and
                `ocr_model_path` and `ocr_config_path` parameters are expected to pass them to
                `fast-plate-ocr` library.
            ocr_device: The device to run the OCR model on ("cuda", "cpu", or "auto").
            ocr_providers: Execution providers for the OCR. If None, the default providers are used.
            ocr_sess_options: Session options for the OCR. If None, default session options are
                used.
            ocr_model_path: Custom model path for the OCR. If None, the model is downloaded from the
                hub or cache.
            ocr_config_path: Custom config path for the OCR. If None, the default configuration is
                used.
            ocr_force_download: Whether to force download the OCR model.
            correct_indian_plates: Whether to apply rule-based correction for Indian license plates.
                Defaults to True. Fixes common OCR errors like O→0, I→1 based on position.
        """
        # Initialize the detector
        self.detector = detector or DefaultDetector(
            model_name=detector_model,
            conf_thresh=detector_conf_thresh,
            providers=detector_providers,
            sess_options=detector_sess_options,
        )

        # Initialize the OCR
        self.ocr = ocr or DefaultOCR(
            hub_ocr_model=ocr_model,
            device=ocr_device,
            providers=ocr_providers,
            sess_options=ocr_sess_options,
            model_path=ocr_model_path,
            config_path=ocr_config_path,
            force_download=ocr_force_download,
        )

        # Store Indian plate correction setting
        self.correct_indian_plates = correct_indian_plates

    def predict(self, frame: np.ndarray | str) -> list[ALPRResult]:
        """
        Run plate detection and OCR on an image.

        Parameters:
            frame: Unprocessed frame (Colors in order: BGR) or image path.

        Returns:
            A list of ALPRResult objects, one for each detected plate.
        """
        if isinstance(frame, str):
            img_path = frame
            img = cv2.imread(img_path)
            if img is None:
                raise ValueError(f"Failed to load image from path: {img_path}")
        else:
            img = frame

        plate_detections = self.detector.predict(img)
        alpr_results: list[ALPRResult] = []
        for detection in plate_detections:
            bbox = detection.bounding_box
            w = bbox.x2 - bbox.x1
            h = bbox.y2 - bbox.y1
            
            # Generous padding (15% horiz, 10% vert)
            pad_w = int(w * 0.15)
            pad_h = int(h * 0.10)
            
            x1 = max(bbox.x1 - pad_w, 0)
            y1 = max(bbox.y1 - pad_h, 0)
            x2 = min(bbox.x2 + pad_w, img.shape[1])
            y2 = min(bbox.y2 + pad_h, img.shape[0])
            
            cropped_plate = img[int(y1):int(y2), int(x1):int(x2)]

            # ── Deskew: Auto-straighten tilted plates ──────────────────────────
            # When a plate is tilted in the camera view, the right-edge characters
            # become foreshortened, making them unreadable. We detect the skew
            # angle via Hough lines and rotate the crop to level it before OCR.
            try:
                _gray = cv2.cvtColor(cropped_plate, cv2.COLOR_BGR2GRAY)
                _edges = cv2.Canny(_gray, 50, 150, apertureSize=3)
                _lines = cv2.HoughLinesP(_edges, 1, np.pi/180, threshold=40,
                                         minLineLength=int(w * 0.3), maxLineGap=20)
                if _lines is not None and len(_lines) > 0:
                    # Collect angles of detected lines
                    _angles = []
                    for _line in _lines:
                        _x1, _y1, _x2, _y2 = _line[0]
                        _angle = np.degrees(np.arctan2(_y2 - _y1, _x2 - _x1))
                        if abs(_angle) < 25:  # only horizontal-ish lines
                            _angles.append(_angle)
                    
                    if _angles:
                        _skew = np.median(_angles)
                        if abs(_skew) > 0.5:  # only correct if meaningful tilt
                            _ch, _cw = cropped_plate.shape[:2]
                            _M = cv2.getRotationMatrix2D((_cw / 2, _ch / 2), _skew, 1.0)
                            cropped_plate = cv2.warpAffine(
                                cropped_plate, _M, (_cw, _ch),
                                flags=cv2.INTER_LINEAR,
                                borderMode=cv2.BORDER_REPLICATE
                            )
            except Exception:
                pass  # If deskew fails, continue with original crop

            # ── Dual-Pass OCR: Sliding window to handle wide Indian plates ──────
            # The OCR model input is 128x64 (2:1 ratio). Indian plates are ~4.5:1.
            # When squished to 128x64, right-side characters (like the BH Series
            # suffix 'B') get distorted and misread. 
            #
            # Strategy: Run OCR on left 65% and right 65% (overlapping), then 
            # merge by taking the longer/better result.
            try:
                ch, cw = cropped_plate.shape[:2]
                left_crop  = cropped_plate[:, :int(cw * 0.65)]
                right_crop = cropped_plate[:, int(cw * 0.35):]

                ocr_left  = self.ocr.predict(left_crop)
                ocr_right = self.ocr.predict(right_crop)
                ocr_result = self.ocr.predict(cropped_plate)

                # Pick the reading strategy:
                # - left gives the prefix correctly
                # - right gives the suffix (ending chars) correctly
                # Merge: use full result if confident, else try to extend with right
                if ocr_result and ocr_left and ocr_right:
                    import statistics as _stats
                    
                    def _avg_conf(r):
                        if r is None: return 0.0
                        c = r.confidence
                        if isinstance(c, (list, tuple)):
                            # Ignore padding chars (low conf _)
                            vals = [float(v) for v in c if float(v) > 0.1]
                            return _stats.mean(vals) if vals else 0.0
                        return float(c)

                    def _clean(t):
                        return t.strip("_").strip() if t else ""

                    full_text  = _clean(ocr_result.text)
                    left_text  = _clean(ocr_left.text)
                    right_text = _clean(ocr_right.text)
                    
                    # The right crop result gives us the LAST characters correctly.
                    # If the full_text is shorter than the expected plate length,
                    # try merging it with the right crop text using overlap matching.
                    if len(full_text) < 10 and len(right_text) > 0:
                        # Find the longest prefix of right_text that is a suffix of full_text.
                        # Require a minimum overlap of at least 2 characters to avoid false matches.
                        for i in range(len(right_text), 1, -1):
                            prefix = right_text[:i]
                            if full_text.endswith(prefix):
                                suffix = right_text[i:]
                                if suffix:
                                    merged = full_text + suffix
                                    import dataclasses
                                    ocr_result = dataclasses.replace(ocr_result, text=merged)
                                    break
            except Exception:
                ocr_result = self.ocr.predict(cropped_plate)

            # ── Apply Indian plate correction if enabled ───────────────────────
            if self.correct_indian_plates and ocr_result and ocr_result.text:
                corrected_text = correct_indian_plate(ocr_result.text)
                if corrected_text != ocr_result.text:
                    import dataclasses
                    ocr_result = dataclasses.replace(ocr_result, text=corrected_text)

            alpr_result = ALPRResult(detection=detection, ocr=ocr_result)
            alpr_results.append(alpr_result)
        return alpr_results

    def draw_predictions(self, frame: np.ndarray | str) -> DrawPredictionsResult:
        """
        Draw detections and OCR results on an image.

        Parameters:
            frame: The original frame or image path.

        Returns:
            A DrawPredictionsResult with the annotated image and the ALPR results.
        """
        # If frame is a string, assume it's an image path and load it
        if isinstance(frame, str):
            img_path = frame
            img = cv2.imread(img_path)
            if img is None:
                raise ValueError(f"Failed to load image from path: {img_path}")
        else:
            img = frame

        # Get ALPR results using the ndarray
        alpr_results = self.predict(img)

        for result in alpr_results:
            detection = result.detection
            ocr_result = result.ocr
            bbox = detection.bounding_box
            x1, y1, x2, y2 = bbox.x1, bbox.y1, bbox.x2, bbox.y2
            # Draw the bounding box
            cv2.rectangle(img, (x1, y1), (x2, y2), (36, 255, 12), 2)
            if ocr_result is None or not ocr_result.text or not ocr_result.confidence:
                continue
            confidence: float = (
                statistics.mean(ocr_result.confidence)
                if isinstance(ocr_result.confidence, list)
                else ocr_result.confidence
            )
            font_scale = min(1.25, max(0.4, img.shape[1] / 1000))
            text_thickness = 1 if font_scale < 0.75 else 2
            outline_thickness = text_thickness + max(3, round(font_scale * 3))
            display_lines = [f"{ocr_result.text} {confidence * 100:.0f}%"]
            if ocr_result.region:
                region_text = ocr_result.region
                if ocr_result.region_confidence is not None:
                    region_text = f"{region_text} {ocr_result.region_confidence * 100:.0f}%"
                display_lines.insert(0, region_text)

            _, text_height = cv2.getTextSize(
                display_lines[0], cv2.FONT_HERSHEY_SIMPLEX, font_scale, text_thickness
            )[0]
            line_gap = max(14, round(text_height * 0.6))
            line_height = text_height + line_gap
            text_y = y1 - 10 - ((len(display_lines) - 1) * line_height)
            if text_y - text_height < 0:
                text_y = y2 + text_height + 10

            for idx, line in enumerate(display_lines):
                text_width, current_text_height = cv2.getTextSize(
                    line, cv2.FONT_HERSHEY_SIMPLEX, font_scale, text_thickness
                )[0]
                text_x = min(max(x1, 5), max(5, img.shape[1] - text_width - 5))
                current_y = min(
                    max(text_y + (idx * line_height), current_text_height + 5),
                    img.shape[0] - 5,
                )
                # Draw black background for better readability
                cv2.putText(
                    img=img,
                    text=line,
                    org=(text_x, current_y),
                    fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                    fontScale=font_scale,
                    color=(0, 0, 0),
                    thickness=outline_thickness,
                    lineType=cv2.LINE_AA,
                )
                # Draw white text
                cv2.putText(
                    img=img,
                    text=line,
                    org=(text_x, current_y),
                    fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                    fontScale=font_scale,
                    color=(255, 255, 255),
                    thickness=text_thickness,
                    lineType=cv2.LINE_AA,
                )

        return DrawPredictionsResult(image=img, results=alpr_results)
