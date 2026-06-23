import http.server
import socketserver
import threading
import time
import os
import tempfile
import cv2
import base64
import requests
from datetime import datetime, timezone
from pathlib import Path

# ── IOU Helper for Tracking ───────────────────────────────────────────
def get_iou(boxA, boxB):
    """Calculate Intersection over Union of two bounding boxes."""
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    interArea = max(0, xB - xA) * max(0, yB - yA)
    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
    return interArea / float(boxAArea + boxBArea - interArea + 1e-6)

class TrackedVehicle:
    """Represents a single physical vehicle being tracked through the frame."""
    def __init__(self, track_id, bbox, ocr_text, ocr_conf, frame):
        self.track_id = track_id
        self.bbox = bbox
        self.best_text = ocr_text
        self.best_conf = ocr_conf
        self.best_frame = frame.copy()
        self.first_seen = time.time()
        self.last_seen = time.time()
        self.is_committed = False
        self.is_manually_overridden = False
        self.frames_since_seen = 0

    def update(self, bbox, ocr_text, ocr_conf, frame):
        self.bbox = bbox
        self.last_seen = time.time()
        self.frames_since_seen = 0
        # Only update "Best Shot" if confidence is higher and not manually overridden
        if not self.is_manually_overridden and ocr_conf > self.best_conf:
            self.best_conf = ocr_conf
            self.best_text = ocr_text
            self.best_frame = frame.copy()


class StreamManager:
    """
    Industry-Level ALPR with Spatial Tracking IDs.
    
    1. Tracking: Each vehicle gets a unique ID via Spatial IOU.
    2. Deduplication: One count per Tracking ID, regardless of OCR flickers.
    3. Best-Shot: Records only the clearest image found during the entire track.
    """
    def __init__(self, alpr_model):
        self.alpr = alpr_model
        self.camera = None
        self.running = False
        self.thread = None

        # Shared State
        self.latest_jpeg = None
        self.frame_id = 0
        self.latest_detection = None
        self.latest_full_b64 = None
        self.detection_log = []
        self.vehicle_count = 0
        self.saved_count = 0
        self.snap_requested = False
        self._log_lock = threading.Lock()

        # Tracking State
        self.tracked_vehicles = {} # track_id -> TrackedVehicle
        self.next_track_id = 1
        self.processing_enabled = False

        # Config
        self.conf_threshold = 0.3
        self.draw_boxes = True
        self.clean_text = True
        self.auto_save = True
        self.output_dir = None
        self.cooldown_window = 5.0
        self.min_chars = 8
        self.webhook_url = ""

    def start(self, camera):
        self.camera = camera
        self.running = True
        self.thread = threading.Thread(target=self._process_loop, daemon=True)
        self.thread.start()

    def reset_tracking(self):
        with self._log_lock:
            self.vehicle_count = 0
            self.saved_count = 0
            self.detection_log = []
        self.tracked_vehicles = {}
        self.next_track_id = 1

    def _save_atomic(self, frame, prefix="ipcam", target_subfolder=None):
        out_dir = target_subfolder if target_subfolder else self.output_dir
        if not out_dir: return
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        dest = out_dir / f"{prefix}_{ts}.jpg"
        fd, tmp = tempfile.mkstemp(suffix=".jpg")
        os.close(fd)
        try:
            cv2.imwrite(tmp, frame)
            os.replace(tmp, str(dest))
        except: pass

    def _process_loop(self):
        from alpr_utils import draw_plates_no_country, get_conf, clean_indian_plate, get_plate_crop
        import numpy as np

        while self.running:
            t_start = time.time()
            frame = self.camera.get_frame()
            if frame is None:
                time.sleep(0.01)
                continue

            results = []
            if self.processing_enabled:
                results = self.alpr.predict(frame)
            
            # Annotated frame for display
            annotated = draw_plates_no_country(frame, results) if self.draw_boxes else frame.copy()
            
            # --- High-Contrast FPS Overlay ---
            elapsed = (time.time() - t_start) * 1000
            fps = 1000 / max(elapsed, 1)
            label = f"FPS: {fps:.1f} ({elapsed:.0f}ms)"
            # Draw semi-transparent background box
            cv2.rectangle(annotated, (10, 10), (350, 60), (0, 0, 0), -1)
            cv2.putText(annotated, label, (20, 45), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)
            
            active_this_frame = [] # list of (bbox, text, conf)

            # 1. Extract raw detections
            for r in results:
                if not r.ocr: continue
                conf = get_conf(r.ocr)
                if conf < self.conf_threshold: continue
                text = clean_indian_plate(r.ocr.text) if self.clean_text else r.ocr.text
                if not text or len(text) < self.min_chars: continue # Indian Plates: at least min chars
                
                box = r.detection.bounding_box
                bbox = [box.x1, box.y1, box.x2, box.y2]
                active_this_frame.append((bbox, text, conf, r))

            # 2. Match to existing trackers (IOU)
            matched_indices = set()
            for track_id, vehicle in list(self.tracked_vehicles.items()):
                best_iou = 0
                best_match_idx = -1
                
                for i, (bbox, text, conf, r) in enumerate(active_this_frame):
                    if i in matched_indices: continue
                    iou = get_iou(vehicle.bbox, bbox)
                    if iou > 0.3 and iou > best_iou:
                        best_iou = iou
                        best_match_idx = i
                
                if best_match_idx != -1:
                    bbox, text, conf, r = active_this_frame[best_match_idx]
                    
                    old_best_conf = vehicle.best_conf
                    vehicle.update(bbox, text, conf, frame)
                    matched_indices.add(best_match_idx)
                    
                    # Live Update UI banner ONLY if we got a new highest confidence (or manual override)
                    if vehicle.is_manually_overridden or vehicle.best_conf > old_best_conf:
                        crop = get_plate_crop(frame, r.detection.bounding_box)
                        _, c_buf = cv2.imencode('.jpg', crop, [cv2.IMWRITE_JPEG_QUALITY, 85])
                        _, f_buf = cv2.imencode('.jpg', annotated, [cv2.IMWRITE_JPEG_QUALITY, 70])
                        
                        self.latest_detection = {
                            "text": vehicle.best_text, "conf": vehicle.best_conf,
                            "crop_b64": f"data:image/jpeg;base64,{base64.b64encode(c_buf).decode()}",
                            "full_b64": f"data:image/jpeg;base64,{base64.b64encode(f_buf).decode()}",
                            "timestamp": datetime.now().strftime("%H:%M:%S")
                        }
                        self.latest_full_b64 = self.latest_detection["full_b64"]
                else:
                    vehicle.frames_since_seen += 1

            # 3. Create new trackers for unmatched boxes
            for i, (bbox, text, conf, r) in enumerate(active_this_frame):
                if i not in matched_indices:
                    self.tracked_vehicles[self.next_track_id] = TrackedVehicle(
                        self.next_track_id, bbox, text, conf, frame
                    )
                    
                    # Initial UI update for new tracker
                    crop = get_plate_crop(frame, r.detection.bounding_box)
                    _, c_buf = cv2.imencode('.jpg', crop, [cv2.IMWRITE_JPEG_QUALITY, 85])
                    _, f_buf = cv2.imencode('.jpg', annotated, [cv2.IMWRITE_JPEG_QUALITY, 70])
                    
                    self.latest_detection = {
                        "text": text, "conf": conf,
                        "crop_b64": f"data:image/jpeg;base64,{base64.b64encode(c_buf).decode()}",
                        "full_b64": f"data:image/jpeg;base64,{base64.b64encode(f_buf).decode()}",
                        "timestamp": datetime.now().strftime("%H:%M:%S")
                    }
                    self.latest_full_b64 = self.latest_detection["full_b64"]
                    
                    self.next_track_id += 1

            # 4. Commit and Cleanup stale trackers
            for track_id, vehicle in list(self.tracked_vehicles.items()):
                # If vehicle visible for cooldown window, or just left the frame
                should_commit = (not vehicle.is_committed and 
                                (time.time() - vehicle.first_seen > self.cooldown_window))
                
                if should_commit:
                    with self._log_lock:
                        dt_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                        self.detection_log.insert(0, (vehicle.best_text, vehicle.best_conf, dt_str))
                        self.vehicle_count += 1
                        
                        if self.auto_save and self.output_dir:
                            # Folder 1: Best Shots
                            shot_dir = self.output_dir / "best_shots"
                            shot_dir.mkdir(exist_ok=True)
                            self._save_atomic(vehicle.best_frame, "plate", target_subfolder=shot_dir)
                            self.saved_count += 1
                            
                            # Folder 2: Data Logs (CSV)
                            log_dir = self.output_dir / "logs"
                            log_dir.mkdir(exist_ok=True)
                            csv_path = log_dir / "records.csv"
                            import csv
                            file_exists = csv_path.exists()
                            with open(csv_path, 'a', newline='') as f:
                                writer = csv.writer(f)
                                if not file_exists:
                                    writer.writerow(["Timestamp", "Plate", "Confidence", "TrackID"])
                                writer.writerow([dt_str, vehicle.best_text, f"{vehicle.best_conf:.2%}", track_id])
                        
                        if getattr(self, "webhook_url", ""):
                            # Format payload to match the remote Swagger API schema (AnprWebhookPayload)
                            payload = {
                                "camera_id": str(self.camera.rtsp_url) if (self.camera and getattr(self.camera, "rtsp_url", None)) else "camera_1",
                                "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
                                "plate": {
                                    "number": vehicle.best_text
                                },
                                "confidence": round(float(vehicle.best_conf), 4)
                            }
                            def send():
                                try:
                                    headers = {"X-API-Key": "my_secure_camera_token_123"}
                                    import json
                                    print("\n" + "="*50)
                                    print("🚀 📤 SENDING WEBHOOK TO REMOTE SERVER:")
                                    print(f"🔗 URL: {self.webhook_url}")
                                    print(f"🔑 Headers: {json.dumps(headers, indent=2)}")
                                    print(f"📦 Payload: {json.dumps(payload, indent=2)}")
                                    print("="*50 + "\n", flush=True)
                                    
                                    response = requests.post(self.webhook_url, json=payload, headers=headers, timeout=3)
                                    print(f"📡 Webhook response | Status: {response.status_code} | Response: {response.text}", flush=True)
                                except Exception as e:
                                    print(f"❌ Webhook failed to send to {self.webhook_url} | Error: {str(e)}", flush=True)
                            threading.Thread(target=send, daemon=True).start()
                    
                    vehicle.is_committed = True

                # Remove tracker if gone for 30 frames
                if vehicle.frames_since_seen > 30:
                    del self.tracked_vehicles[track_id]

            # MJPEG stream logic
            ret, buf = cv2.imencode('.jpg', annotated, [cv2.IMWRITE_JPEG_QUALITY, 70])
            if ret:
                self.latest_jpeg = buf.tobytes()
                self.frame_id += 1
            
            time.sleep(0.01)

# MJPEG Server classes (Standard)
class MJPEGHandler(http.server.BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path == '/stream':
            self.send_response(200)
            self.send_header('Content-type', 'multipart/x-mixed-replace; boundary=frame')
            self.end_headers()
            last_id = -1
            while True:
                data, fid = self.server.stream_manager.latest_jpeg, self.server.stream_manager.frame_id
                if data and fid != last_id:
                    self.wfile.write(b'--frame\r\nContent-type: image/jpeg\r\n\r\n' + data + b'\r\n')
                    last_id = fid
                time.sleep(0.01)
    def log_message(self, *args): pass

class StreamServer(socketserver.ThreadingMixIn, http.server.HTTPServer):
    def __init__(self, addr, handler, manager):
        super().__init__(addr, handler)
        self.stream_manager = manager

def start_server_thread(manager, port=8504):
    server = StreamServer(('0.0.0.0', port), MJPEGHandler, manager)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server
