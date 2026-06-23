import os
import threading
import time

import cv2


class IPCamera:
    """
    Non-blocking RTSP / USB camera reader.

    Uses OpenCV VideoCapture for network streaming and local USB webcams.
    """

    def __init__(self, rtsp_url: str, reconnect_delay: float = 1.0) -> None:
        self.rtsp_url        = rtsp_url
        self.reconnect_delay = reconnect_delay

        self.cap:      cv2.VideoCapture | None = None
        self.frame:    "cv2.typing.MatLike | None" = None
        self.running:  bool = False
        self._lock:    threading.Lock = threading.Lock()
        self._thread:  threading.Thread | None = None

        # Stats
        self.total_frames:    int   = 0
        self.dropped_frames:  int   = 0
        self.reconnect_count: int   = 0
        self.connected:       bool  = False

    # ── Public API ────────────────────────────────────────────────────────────

    def connect(self) -> bool:
        """Open the stream and start reading frames."""
        self.running = True

        src = int(self.rtsp_url) if str(self.rtsp_url).isdigit() else self.rtsp_url
        if isinstance(src, str) and (src.startswith("rtsp://") or src.startswith("http://") or src.startswith("https://")):
            os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp;recv_buffer_size;10485760"
            self.cap = cv2.VideoCapture(src, cv2.CAP_FFMPEG)
        else:
            self.cap = cv2.VideoCapture(src)

        if not self.cap.isOpened():
            print(f"[ERROR] Could not connect to camera: {self.rtsp_url}")
            self.connected = False
            return False

        self.connected = True
        self._thread = threading.Thread(target=self._read_loop, daemon=True)
        self._thread.start()
        print(f"[OK] Camera connected via OpenCV: {self.rtsp_url}")
        return True

    def get_frame(self) -> "cv2.typing.MatLike | None":
        """Return a copy of the most-recently captured frame (thread-safe)."""
        with self._lock:
            if self.frame is not None:
                return self.frame.copy()
        return None

    def disconnect(self) -> None:
        """Stop reading frames and release resources."""
        self.running   = False
        self.connected = False
        
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=3)
        if self.cap:
            self.cap.release()
            self.cap = None
        print("[INFO] Camera disconnected.")

    def get_stats(self) -> dict:
        return {
            "connected":       self.connected,
            "total_frames":    self.total_frames,
            "dropped_frames":  self.dropped_frames,
            "reconnect_count": self.reconnect_count,
        }

    # ── Internal ──────────────────────────────────────────────────────────────

    def _read_loop(self) -> None:
        """Background loop for OpenCV video capture"""
        while self.running:
            ret, frame = self.cap.read()
            if ret:
                with self._lock:
                    self.frame = frame
                self.total_frames += 1
                self.connected = True
            else:
                self.dropped_frames += 1
                self.connected = False
                # ── Auto-reconnect ────────────────────────────────────────
                if self.running:
                    print(f"[WARNING] Stream dropped. Reconnecting in {self.reconnect_delay}s...")
                    self.cap.release()
                    time.sleep(self.reconnect_delay)
                    src = int(self.rtsp_url) if str(self.rtsp_url).isdigit() else self.rtsp_url
                    if isinstance(src, str) and (src.startswith("rtsp://") or src.startswith("http://") or src.startswith("https://")):
                        os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp;recv_buffer_size;10485760"
                        self.cap = cv2.VideoCapture(src, cv2.CAP_FFMPEG)
                    else:
                        self.cap = cv2.VideoCapture(src)
                    
                    if self.cap.isOpened():
                        self.reconnect_count += 1
                        self.connected = True
                        print(f"[OK] Reconnected (attempt #{self.reconnect_count})")
                    else:
                        print("[ERROR] Reconnect failed. Retrying...")
            time.sleep(0.01)

