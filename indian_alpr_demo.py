"""
Indian ALPR Demo Application
High-performance Automatic License Plate Recognition for Indian license plates.
"""

import io
import ssl
ssl._create_default_https_context = ssl._create_unverified_context

import time
import statistics
import tempfile
import os
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import streamlit as st
from PIL import Image
from fast_alpr import ALPR
from camera import IPCamera

# ─── Initialization: Create Shortcut ──────────────────────────────────────────
def ensure_desktop_shortcut():
    import subprocess
    import platform
    if platform.system() == "Windows": # Shortcut logic only for Windows
        try:
            ps_script = Path(__file__).parent / "create_desktop_shortcut.ps1"
            if ps_script.exists():
                subprocess.run(["powershell.exe", "-ExecutionPolicy", "Bypass", "-File", str(ps_script)], 
                              capture_output=True, check=False)
        except Exception:
            pass

ensure_desktop_shortcut()

# ─── Output folder ────────────────────────────────────────────────────────────
OUTPUT_DIR = Path(__file__).parent / "output"
OUTPUT_DIR.mkdir(exist_ok=True)

# ─── Page Config ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Indian ALPR – Number Plate Detection",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Initialize Session State ────────────────────────────────────────────────
if "cam_instance" not in st.session_state:
    st.session_state.cam_instance   = None   # IPCamera object
if "cam_streaming" not in st.session_state:
    st.session_state.cam_streaming  = False  # live-loop flag
if "cam_plate_log" not in st.session_state:
    st.session_state.cam_plate_log  = []     # [(plate, conf, timestamp)]
if "cam_saved_count" not in st.session_state:
    st.session_state.cam_saved_count = 0
if "last_detected_plate" not in st.session_state:
    st.session_state.last_detected_plate = None
if "last_saved_plate" not in st.session_state:
    st.session_state.last_saved_plate = None  # plate text of last saved image
if "last_saved_time" not in st.session_state:
    st.session_state.last_saved_time = 0.0   # epoch time of last save
if "cam_vehicle_count" not in st.session_state:
    st.session_state.cam_vehicle_count = 0   # count of unique vehicles detected

# ─── Custom CSS ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

.stApp {
    background: linear-gradient(135deg, #0f0c29 0%, #302b63 50%, #24243e 100%);
    min-height: 100vh;
    color: #e0e0ff;
}

/* Force specific elements to be light for visibility against the background */
h1, h2, h3, h4, h5, h6, p, label, .stMarkdown, .stText, [data-testid="stHeader"] {
    color: #e0e0ff !important;
}

/* Fix Checkbox/Radio visibility */
[data-testid="stCheckbox"] label p {
    color: #e0e0ff !important;
}

/* NUCLEAR CSS FIX: Force dark inputs even on Light systems */
[data-testid="stSelectbox"] div[data-baseweb="select"], 
[data-testid="stSelectbox"] div[data-baseweb="select"] > div,
[data-testid="stSelectbox"] ul[role="listbox"] {
    background-color: #1e1b4b !important;
    color: #ffffff !important;
}
[data-testid="stSelectbox"] * { color: #ffffff !important; }

/* Fixing the Output Folder and URL text box visibility */
[data-testid="stTextInput"] input, 
[data-testid="stNumberInput"] input,
[data-testid="stFileUploader"] section {
    background-color: #1e1b4b !important;
    color: #ffffff !important;
    border: 1px solid rgba(255,255,255,0.2) !important;
}
/* Ensure the "Upload" text and buttons are visible */
[data-testid="stFileUploader"] * { color: #ffffff !important; }
[data-testid="stFileUploader"] button {
    background-color: #6c63ff !important;
    color: white !important;
}

/* Fix Code blocks (Output Folder / Filenames) */
code, pre, [data-testid="stCode"] {
    background-color: #1e1b4b !important;
    color: #a78bfa !important;
    border: 1px solid rgba(255,255,255,0.1) !important;
}

/* Fix Tab visibility */
button[data-baseweb="tab"] div p {
    color: #b0b0ff !important;
}
button[aria-selected="true"] div p {
    color: #ffffff !important;
    font-weight: 700;
}

section[data-testid="stSidebar"] {
    background: rgba(255,255,255,0.05);
    backdrop-filter: blur(10px);
    border-right: 1px solid rgba(255,255,255,0.1);
}
section[data-testid="stSidebar"] * { color: #e0e0ff !important; }

.main .block-container { padding-top: 2rem; }

.hero-banner {
    background: linear-gradient(90deg, #6c63ff 0%, #3ecfcf 100%);
    border-radius: 16px;
    padding: 2rem 2.5rem;
    margin-bottom: 2rem;
    color: white;
    box-shadow: 0 8px 32px rgba(108, 99, 255, 0.4);
}
.hero-banner h1 { font-size: 2.2rem; font-weight: 700; margin: 0; text-shadow: 0 2px 4px rgba(0,0,0,0.3); }
.hero-banner p  { font-size: 1.05rem; margin: 0.5rem 0 0 0; opacity: 0.9; }

.result-card {
    background: rgba(255,255,255,0.07);
    border: 1px solid rgba(108,99,255,0.4);
    border-radius: 12px;
    padding: 1.5rem;
    margin: 0.5rem 0;
    backdrop-filter: blur(6px);
}
.plate-text {
    font-size: 2rem; font-weight: 700; color: #a78bfa;
    letter-spacing: 4px; font-family: 'Courier New', monospace;
}
.confidence-badge {
    display: inline-block; padding: 4px 14px; border-radius: 20px;
    font-size: 0.85rem; font-weight: 600; margin-top: 6px;
}
.high-conf { background: rgba(52,211,153,0.2); color: #34d399; border: 1px solid #34d399; }
.mid-conf  { background: rgba(251,191,36,0.2);  color: #fbbf24; border: 1px solid #fbbf24; }
.low-conf  { background: rgba(239,68,68,0.2);   color: #ef4444; border: 1px solid #ef4444; }

.info-box {
    background: rgba(62, 207, 207, 0.1);
    border-left: 4px solid #3ecfcf;
    border-radius: 8px;
    padding: 1rem 1.2rem; margin: 1rem 0;
    color: #cff; font-size: 0.92rem;
}
.saved-box {
    background: rgba(52,211,153,0.1);
    border-left: 4px solid #34d399;
    border-radius: 8px;
    padding: 0.8rem 1.2rem; margin: 0.5rem 0;
    color: #a7f3d0; font-size: 0.87rem;
}
.stat-tile {
    background: rgba(255,255,255,0.06);
    border: 1px solid rgba(255,255,255,0.12);
    border-radius: 10px; padding: 1rem 1.5rem;
    text-align: center; color: white;
}
.stat-value { font-size: 1.8rem; font-weight: 700; color: #a78bfa; }
.stat-label { font-size: 0.8rem; color: #9ca3af; margin-top: 2px; }

.stButton > button {
    background: linear-gradient(90deg, #6c63ff, #3ecfcf) !important;
    color: white !important; border: none !important;
    border-radius: 8px !important; font-weight: 600 !important;
    padding: 0.5rem 1.5rem !important; transition: all 0.3s !important;
    box-shadow: 0 4px 15px rgba(108,99,255,0.3) !important;
}
.stButton > button:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 6px 20px rgba(108,99,255,0.5) !important;
}
img { border-radius: 10px !important; }

/* Prediction Banner Styles */
.prediction-banner {
    background: rgba(30, 27, 75, 0.6);
    border-radius: 16px;
    padding: 1.5rem;
    margin: 1.5rem 0;
    text-align: center;
    border: 2px solid #6c63ff;
    backdrop-filter: blur(10px);
    box-shadow: 0 10px 25px rgba(0, 0, 0, 0.3), 0 0 15px rgba(108, 99, 255, 0.2);
    animation: fadeIn 0.5s ease-out;
}
@keyframes fadeIn { from { opacity: 0; transform: translateY(-10px); } to { opacity: 1; transform: translateY(0); } }

.prediction-banner .label {
    color: #a78bfa;
    font-size: 0.85rem;
    font-weight: 600;
    margin-bottom: 0.6rem;
    text-transform: uppercase;
    letter-spacing: 2px;
}
.prediction-banner .plate {
    color: #ffffff;
    font-size: 3.2rem;
    font-weight: 800;
    font-family: 'Courier New', monospace;
    letter-spacing: 8px;
    text-shadow: 0 0 15px rgba(108, 99, 255, 0.8);
    display: block;
    margin: 0.5rem 0;
}
.prediction-banner .conf {
    font-size: 1rem;
    font-weight: 500;
    color: #34d399;
}
</style>
""", unsafe_allow_html=True)


# ─── Shared Stream Manager (NEW) ─────────────────────────────────────────────
if "stream_manager" not in st.session_state:
    from stream_server import StreamManager, start_server_thread
    st.session_state.stream_manager = StreamManager(None) # Updated after model load
    st.session_state.mjpeg_server = None

from alpr_utils import get_conf


# ─── Custom draw (no country label) ──────────────────────────────────────────
from alpr_utils import draw_plates_no_country


# ─── Crop & Base64 Helpers ───────────────────────────────────────────────────
from alpr_utils import get_plate_crop


def img_to_html(img: np.ndarray) -> str:
    """Convert an OpenCV image to a base64 string for HTML display."""
    import base64
    # Upscale slightly for better "Zoom" visibility if the crop is small
    h, w = img.shape[:2]
    if w < 300:
        scale = 300 / w
        img = cv2.resize(img, (0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    
    _, buffer = cv2.imencode('.jpg', img, [cv2.IMWRITE_JPEG_QUALITY, 90])
    b64_str = base64.b64encode(buffer).decode()
    return f"data:image/jpeg;base64,{b64_str}"


# ─── Save annotated image ─────────────────────────────────────────────────────
from alpr_utils import save_output_image as _save_output_image

def save_output_image(annotated: np.ndarray, prefix: str = "capture") -> Path:
    return _save_output_image(annotated, OUTPUT_DIR, prefix)


# ─── Cached ALPR Model ────────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_alpr_model(detector_model: str, ocr_model: str, correct_indian_plates: bool):
    return ALPR(detector_model=detector_model, ocr_model=ocr_model, correct_indian_plates=correct_indian_plates)


# ─── Indian plate post-processing ────────────────────────────────────────────
from alpr_utils import clean_indian_plate


def confidence_class(conf: float) -> str:
    if conf >= 0.75:
        return "high-conf"
    elif conf >= 0.50:
        return "mid-conf"
    return "low-conf"


def confidence_label(conf: float) -> str:
    if conf >= 0.75:
        return "✅ High Confidence"
    elif conf >= 0.50:
        return "⚠️ Medium Confidence"
    return "❌ Low Confidence"


@st.dialog("✏️ Manual Plate Update")
def update_plate_dialog(initial_plate: str):
    sm = st.session_state.stream_manager

    new_plate = st.text_input("Enter correct number plate:", value=initial_plate)
    
    col1, col2 = st.columns(2)
    if col1.button("Save Update", type="primary", use_container_width=True):
        if new_plate and new_plate.upper() != initial_plate:
            new_plate_upper = new_plate.upper()
            target_plate = initial_plate
            
            sm.latest_detection["text"] = new_plate_upper
            sm.latest_detection["conf"] = 1.0  # Human verified
            
            # Update the detection_log
            if sm.detection_log:
                for i in range(len(sm.detection_log)):
                    if sm.detection_log[i][0] == target_plate:
                        sm.detection_log[i] = (new_plate_upper, 1.0, sm.detection_log[i][2])
                        break
                        
            # Update any currently tracked vehicles in the background stream manager
            if hasattr(sm, "tracked_vehicles"):
                for t_id, vehicle in sm.tracked_vehicles.items():
                    if vehicle.best_text == target_plate:
                        vehicle.best_text = new_plate_upper
                        vehicle.best_conf = 1.0
                        vehicle.is_manually_overridden = True
            
            # Also update session_state.cam_plate_log if present
            if "cam_plate_log" in st.session_state and st.session_state.cam_plate_log:
                for i in range(len(st.session_state.cam_plate_log)):
                    if st.session_state.cam_plate_log[i][0] == target_plate:
                        st.session_state.cam_plate_log[i] = (new_plate_upper, 1.0, st.session_state.cam_plate_log[i][2])
                        break

            st.success(f"Updated successfully to {new_plate_upper}!")
            time.sleep(0.5)
            st.rerun()
        else:
            st.rerun()

    if col2.button("Cancel", use_container_width=True):
        st.rerun()



# ─── Sidebar Config ───────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚙️ Configuration")
    st.markdown("---")

    detector_model = st.selectbox(
        "🔍 Detector Model",
        [
            "yolo-v9-s-608-license-plate-end2end",
            "yolo-v9-t-384-license-plate-end2end",
            "yolo-v9-t-256-license-plate-end2end",
        ],
        index=0,
    )

    ocr_model = st.selectbox(
        "🔤 OCR Model",
        ["cct-s-v2-global-model", "cct-xs-v2-global-model"],
        index=0,
    )

    conf_threshold = st.slider(
        "📊 Confidence Threshold", min_value=0.1, max_value=1.0, value=0.3, step=0.05
    )

    cooldown_window = st.number_input(
        "⏱️ Cooldown Window (s)", min_value=1.0, max_value=60.0, value=5.0, step=1.0
    )
    
    min_chars = st.number_input(
        "🔠 Minimum Characters", min_value=1, max_value=15, value=8, step=1
    )

    webhook_url = st.text_input(
        "🔗 Webhook Endpoint", value="", placeholder="https://client-api.com/...",
        help="Leave blank to disable. If set, sends JSON data on best-shot."
    )

    draw_boxes  = st.checkbox("🖼️ Draw Bounding Boxes", value=True)
    clean_text  = st.checkbox("🧹 Clean Indian Plate Format", value=True)
    correct_ocr = st.checkbox("🔧 Rule-Based OCR Correction", value=True,
                              help="Fix common OCR errors (O↔0, I↔1) based on Indian plate format")
    auto_save   = st.checkbox("💾 Auto-save detections to output/", value=True)

    st.markdown("---")
    st.markdown("### 📁 Output Folder")
    st.code(str(OUTPUT_DIR), language=None)
    saved_files = sorted(OUTPUT_DIR.glob("*.jpg"), key=lambda f: f.stat().st_mtime, reverse=True)
    st.markdown(f"**{len(saved_files)}** images saved")
    if saved_files:
        st.markdown(f"Latest: `{saved_files[0].name}`")

    st.markdown("---")
    st.markdown("### 🏷️ Latest Detection")
    # Read from StreamManager which is updated by the background thread
    
    @st.fragment(run_every=2.0)
    def sidebar_latest_detection():
        _sm = st.session_state.get("stream_manager")
        _lp = _sm.latest_detection if _sm else None
        if _lp:
            st.markdown(f"""
            <div style='background:rgba(108,99,255,0.1); border:1px solid #6c63ff; border-radius:8px; padding:10px; text-align:center;'>
                <div style='font-size:0.7rem; color:#a78bfa; margin-bottom:2px;'>LATEST · {_lp['timestamp']}</div>
                <div style='font-family:monospace; font-size:1.2rem; font-weight:bold; color:white;'>{_lp['text']}</div>
                <div style='font-size:0.65rem; color:#34d399;'>{_lp['conf']:.1%} Conf</div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("<i style='font-size:0.8rem; color:#6b7280;'>No detections yet</i>", unsafe_allow_html=True)

    sidebar_latest_detection()

    st.markdown("---")
    st.markdown("### ℹ️ About")
    st.markdown("""
    **FastALPR** Indian Demo  
    - **YOLO v9** plate detection  
    - **CCT OCR** character recognition  
    🇮🇳 Optimized for Indian plates
    """)

# ─── Hero Banner ──────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero-banner">
    <h1>🚗 Indian Number Plate Recognition</h1>
    <p>Upload an image, capture from webcam, or process a video — AI detects plates instantly</p>
</div>
""", unsafe_allow_html=True)

# ─── Load Model ───────────────────────────────────────────────────────────────
with st.spinner("🔄 Loading AI models (first run downloads ~50 MB)..."):
    alpr = load_alpr_model(detector_model, ocr_model, correct_ocr)
    # Register model to manager
    st.session_state.stream_manager.alpr = alpr
    st.session_state.stream_manager.output_dir = OUTPUT_DIR

st.success("✅ Models loaded & ready!", icon="🤖")

# ─── Tabs ─────────────────────────────────────────────────────────────────────
tab_image, tab_webcam, tab_video, tab_batch = st.tabs(
    ["📷 Image", "📹 Webcam", "🎬 Video File", "📦 Batch Test"]
)

# ════════════════════════════════════════════════════════════════════════════
# TAB 1 – Single Image
# ════════════════════════════════════════════════════════════════════════════
with tab_image:
    st.markdown("### Upload an Image")
    uploaded = st.file_uploader(
        "Supported: JPG, PNG, WEBP, BMP",
        type=["jpg", "jpeg", "png", "webp", "bmp"],
        key="img_upload",
    )

    if uploaded:
        file_bytes = np.frombuffer(uploaded.read(), np.uint8)
        frame = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    else:
        frame = None

    if frame is not None:
        # Results area logic
        with st.spinner("🔍 Detecting number plates..."):
            t0      = time.time()
            results = alpr.predict(frame)
            elapsed = time.time() - t0

        detected = [r for r in results if r.ocr and get_conf(r.ocr) >= conf_threshold]

        # ── 1. Primary Result Panel (TOP) ────────────────────────────────────
        if detected:
            top_plate = detected[0]
            plate_text = clean_indian_plate(top_plate.ocr.text) if clean_text else top_plate.ocr.text
            conf = get_conf(top_plate.ocr)
            st.session_state.last_detected_plate = {"text": plate_text, "conf": conf}
            
            # Send Webhook if configured and not already sent for this image upload
            if webhook_url:
                webhook_state_key = f"webhook_sent_{uploaded.name}"
                if st.session_state.get(webhook_state_key) != plate_text:
                    import requests
                    import threading
                    from datetime import datetime, timezone
                    
                    payload = {
                        "camera_id": "image_upload",
                        "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
                        "plate": {
                            "number": plate_text
                        },
                        "confidence": round(float(conf), 4)
                    }
                    
                    def send_image_webhook():
                        try:
                            headers = {"X-API-Key": "my_secure_camera_token_123"}
                            import json
                            print("\n" + "="*50)
                            print("🚀 📤 SENDING IMAGE WEBHOOK TO REMOTE SERVER:")
                            print(f"🔗 URL: {webhook_url}")
                            print(f"🔑 Headers: {json.dumps(headers, indent=2)}")
                            print(f"📦 Payload: {json.dumps(payload, indent=2)}")
                            print("="*50 + "\n", flush=True)

                            response = requests.post(webhook_url, json=payload, headers=headers, timeout=3)
                            print(f"📡 Image Webhook response | Status: {response.status_code} | Response: {response.text}", flush=True)
                        except Exception as e:
                            print(f"❌ Image Webhook failed to send to {webhook_url} | Error: {str(e)}", flush=True)
                    
                    threading.Thread(target=send_image_webhook, daemon=True).start()
                    st.session_state[webhook_state_key] = plate_text
            
            # Get Zoomed Crop
            crop = get_plate_crop(frame, top_plate.detection.bounding_box)
            crop_b64 = img_to_html(crop)

            st.markdown(f"""
            <div class='prediction-banner'>
                <div style='display:flex; align-items:center; gap:25px; flex-wrap:wrap; justify-content:center;'>
                    <div style='flex-shrink:0; border:2px solid #6c63ff; border-radius:8px; overflow:hidden; box-shadow:0 0 15px rgba(108,99,255,0.4);'>
                        <img src='{crop_b64}' style='height:80px; display:block;' />
                    </div>
                    <div style='text-align:left;'>
                        <div class='label' style='margin-bottom:0;'>🎯 Current Prediction</div>
                        <div class='plate' style='font-size:2.8rem; margin:0;'>{plate_text}</div>
                        <div class='conf'>Confidence: {conf:.2%} — {confidence_label(conf)}</div>
                    </div>
                </div>
            </div>""", unsafe_allow_html=True)
        elif results:
            st.info("Plate detected but OCR confidence is below threshold.")

        # ── 2. Visual Comparison ─────────────────────────────────────────────
        col_orig, col_result = st.columns(2, gap="large")

        with col_orig:
            st.markdown("**📥 Original Image**")
            st.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), use_container_width=True)

        with st.spinner("🔍 Detecting number plates..."):
            t0      = time.time()
            results = alpr.predict(frame)
            elapsed = time.time() - t0

        # Draw without country label
        annotated = draw_plates_no_country(frame, results) if draw_boxes else frame.copy()

        # Auto-save
        saved_path = None
        if auto_save and results:
            saved_path = save_output_image(annotated, "image")

        with col_result:
            st.markdown("**📤 Detected Plates**")
            st.image(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB), use_container_width=True)

        # Stats
        detected = [r for r in results if r.ocr and get_conf(r.ocr) >= conf_threshold]
        c1, c2, c3 = st.columns(3)
        with c1:
            st.markdown(f"<div class='stat-tile'><div class='stat-value'>{len(results)}</div>"
                        f"<div class='stat-label'>Plates Detected</div></div>", unsafe_allow_html=True)
        with c2:
            st.markdown(f"<div class='stat-tile'><div class='stat-value'>{len(detected)}</div>"
                        f"<div class='stat-label'>Above Threshold</div></div>", unsafe_allow_html=True)
        with c3:
            st.markdown(f"<div class='stat-tile'><div class='stat-value'>{elapsed*1000:.0f} ms</div>"
                        f"<div class='stat-label'>Inference Time</div></div>", unsafe_allow_html=True)

        if saved_path:
            st.markdown(f"<div class='saved-box'>💾 Saved to <b>{saved_path.name}</b></div>",
                        unsafe_allow_html=True)

            st.markdown("**📤 Detected Plates**")
            st.image(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB), use_container_width=True)

        # Stats
        if results:
            for i, r in enumerate(results):
                if r.ocr is None:
                    continue
                plate_raw  = r.ocr.text
                plate_disp = clean_indian_plate(plate_raw) if clean_text else plate_raw
                conf       = get_conf(r.ocr)
                cls        = confidence_class(conf)
                lbl        = confidence_label(conf)
                
                # Get Crop for Card
                crop = get_plate_crop(frame, r.detection.bounding_box)
                crop_b64 = img_to_html(crop)

                st.markdown(f"""
                <div class='result-card'>
                    <div style='display:flex; gap:15px; align-items:center;'>
                        <div style='flex-shrink:0; border:1px solid rgba(255,255,255,0.2); border-radius:6px; overflow:hidden;'>
                            <img src='{crop_b64}' style='height:60px; display:block;' />
                        </div>
                        <div style='flex-grow:1;'>
                            <div style='color:#9ca3af; font-size:0.8rem; margin-bottom:2px'>
                                Plate #{i+1} &nbsp;·&nbsp; Det: {r.detection.confidence:.1%}
                            </div>
                            <div class='plate-text' style='font-size:1.6rem;'>{plate_disp}</div>
                            <div>
                                <span class='confidence-badge {cls}' style='font-size:0.75rem; padding:2px 10px;'>{lbl}</span>
                            </div>
                        </div>
                    </div>
                </div>""", unsafe_allow_html=True)
        else:
            st.markdown("<div class='info-box'>No plates detected. Try a clearer image or lower the threshold.</div>",
                        unsafe_allow_html=True)

        # Download
        pil_img = Image.fromarray(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB))
        buf = io.BytesIO()
        pil_img.save(buf, format="PNG")
        st.download_button("⬇️ Download Annotated Image", data=buf.getvalue(),
                           file_name="detected_plates.png", mime="image/png")

# ════════════════════════════════════════════════════════════════════════════
# TAB 2 – IP Camera (Threaded, Auto-Reconnect)
# ════════════════════════════════════════════════════════════════════════════
with tab_webcam:
    st.markdown("### 📹 Live IP Camera / Webcam Detection")

    # ── Camera source config ──────────────────────────────────────────────
    st.markdown("#### 🔌 Connection")
    col_src, col_delay = st.columns([3, 1])
    with col_src:
        cam_source = st.text_input(
            "Camera source",
            value="rtsp://admin:Kiran%4011@192.168.1.64:554/Streaming/Channels/101?transportmode=unicast&profile=Profile_1",
            help="Enter 0 for default webcam, 1/2 for USB cameras, or a full RTSP URL",
            key="cam_src",
        )
    with col_delay:
        reconnect_delay = st.number_input(
            "Reconnect delay (s)", min_value=0.5, max_value=10.0, value=1.0, step=0.5
        )

    # ── Control buttons ───────────────────────────────────────────────────
    col_conn, col_disc, col_snap, col_live, col_clear = st.columns(5)

    with col_conn:
        connect_btn = st.button("🟢 Connect", key="cam_connect",
                                disabled=st.session_state.cam_instance is not None)
    with col_disc:
        disconnect_btn = st.button("🔴 Disconnect", key="cam_disconnect",
                                   disabled=st.session_state.cam_instance is None)
    with col_snap:
        snap_btn = st.button("📸 Snap Frame", key="cam_snap",
                             disabled=st.session_state.cam_instance is None)
    with col_live:
        live_label = "⏹ Stop Live" if st.session_state.cam_streaming else "▶️ Start Live"
        live_btn   = st.button(live_label, key="cam_live",
                               disabled=st.session_state.cam_instance is None)
    with col_clear:
        clear_btn  = st.button("🗑 Clear Log", key="cam_clear")

    # ── Button actions ────────────────────────────────────────────────────
    if connect_btn:
        src = int(cam_source.strip()) if cam_source.strip().isdigit() else cam_source.strip()
        cam = IPCamera(str(src), reconnect_delay=reconnect_delay)
        if cam.connect():
            st.session_state.cam_instance = cam
            # Start background processing
            st.session_state.stream_manager.start(cam)
            # Start MJPEG server if not running
            if st.session_state.mjpeg_server is None:
                from stream_server import start_server_thread
                st.session_state.mjpeg_server = start_server_thread(st.session_state.stream_manager, port=8504)
            st.success("✅ Camera connected & Processor started!")
        else:
            st.error("❌ Connection failed. Check the URL / camera source.")
        st.rerun()

    if snap_btn and st.session_state.cam_instance:
        st.session_state.stream_manager.snap_requested = True
        st.toast("📸 Snapshot requested!")

    if disconnect_btn and st.session_state.cam_instance:
        st.session_state.stream_manager.stop()
        st.session_state.cam_instance.disconnect()
        st.session_state.cam_instance  = None
        st.session_state.cam_streaming = False
        st.info("🔌 Camera disconnected.")
        st.rerun()

    if live_btn:
        st.session_state.cam_streaming = not st.session_state.cam_streaming
        # Sync the engine with the UI toggle
        st.session_state.stream_manager.processing_enabled = st.session_state.cam_streaming
        st.rerun()

    if clear_btn:
        sm = st.session_state.stream_manager
        if hasattr(sm, "reset_tracking"):
            sm.reset_tracking()
        else:
            # Fallback for old cached instances
            sm.detection_log  = []
            sm.vehicle_count  = 0
            sm.saved_count    = 0
            if hasattr(sm, "active_events"):
                sm.active_events   = {}
                sm.committed_times = {}
        st.session_state.cam_plate_log     = []
        st.session_state.cam_saved_count   = 0
        st.session_state.cam_vehicle_count = 0
        st.session_state.last_saved_plate  = None
        st.session_state.last_saved_time   = 0.0
        st.toast("🗑 Log cleared!", icon="✅")
        st.rerun()

    # ── Camera status banner ──────────────────────────────────────────────
    cam: IPCamera | None = st.session_state.cam_instance
    if cam is None:
        st.markdown("""
        <div class='info-box'>
            Enter the RTSP URL (or webcam index) above and click
            <b>Connect</b> to start. The camera will auto-reconnect if the stream drops.
        </div>""", unsafe_allow_html=True)
    # Stat tiles (Status, Frames, Reconnects, Vehicles, Saved) are rendered
    # inside the auto-refreshing fragment below so they update every 2 s.

    st.markdown("---")

    # ── Live feed area ────────────────────────────────────────────────────
    # Update manager config
    st.session_state.stream_manager.conf_threshold = conf_threshold
    st.session_state.stream_manager.cooldown_window = cooldown_window
    st.session_state.stream_manager.min_chars = min_chars
    st.session_state.stream_manager.webhook_url = webhook_url
    st.session_state.stream_manager.draw_boxes = draw_boxes
    st.session_state.stream_manager.clean_text = clean_text
    st.session_state.stream_manager.auto_save = auto_save

    @st.fragment(run_every=2.0)
    def update_live_ui():
        sm = st.session_state.stream_manager
        _cam = st.session_state.cam_instance

        # ── Live stat tiles (auto-refresh every 2 s) ──────────────────────
        if _cam is not None:
            _stats = _cam.get_stats()
            s1, s2, s3, s4, s5 = st.columns(5)
            _conn_icon = "🟢" if _stats["connected"] else "🟡"
            s1.markdown(f"<div class='stat-tile'><div class='stat-value'>{_conn_icon}</div>"
                        f"<div class='stat-label'>Status</div></div>", unsafe_allow_html=True)
            s2.markdown(f"<div class='stat-tile'><div class='stat-value'>{_stats['total_frames']}</div>"
                        f"<div class='stat-label'>Frames Read</div></div>", unsafe_allow_html=True)
            s3.markdown(f"<div class='stat-tile'><div class='stat-value'>{_stats['reconnect_count']}</div>"
                        f"<div class='stat-label'>Reconnects</div></div>", unsafe_allow_html=True)
            s4.markdown(f"<div class='stat-tile'><div class='stat-value' style='color:#34d399;'>🚗 {sm.vehicle_count}</div>"
                        f"<div class='stat-label'>Vehicles Detected</div></div>", unsafe_allow_html=True)
            s5.markdown(f"<div class='stat-tile'><div class='stat-value'>{sm.saved_count}</div>"
                        f"<div class='stat-label'>Saved Images</div></div>", unsafe_allow_html=True)

        # 1. Top Panel (Latest Prediction) from Shared Manager
        latest = sm.latest_detection
        if latest:
            st.markdown(f"""
            <div class='prediction-banner' style='margin-bottom: 1.5rem;'>
                <div style='display:flex; align-items:center; gap:20px; flex-wrap:wrap; justify-content:center;'>
                    <div style='flex-shrink:0; border:2px solid #6c63ff; border-radius:8px; overflow:hidden; box-shadow:0 0 10px rgba(108,99,255,0.4);'>
                        <img src='{latest["crop_b64"]}' style='height:70px; display:block;' />
                    </div>
                    <div style='text-align:left;'>
                        <div class='label' style='margin-bottom:0;'>🎯 Live Prediction ({latest["timestamp"]})</div>
                        <div class='plate' style='font-size:2.4rem; margin:0;'>{latest["text"]}</div>
                        <div class='conf'>Confidence: {latest["conf"]:.2%}</div>
                    </div>
                </div>
            </div>""", unsafe_allow_html=True)
        else:
            st.markdown("""
            <div style='background:rgba(255,255,255,0.05); border-radius:12px; padding:1.5rem; text-align:center; border:1px solid rgba(255,255,255,0.1);'>
                <div style='color:#9ca3af; font-size:0.9rem;'>Scanning for number plates...</div>
            </div>""", unsafe_allow_html=True)

        # 2. Live Stream (MJPEG via HTML img)
        if st.session_state.cam_streaming:
            col_feed, col_cap = st.columns([3, 1])
            with col_feed:
                st.markdown(f"""
                    <div style="border: 3px solid #6c63ff; border-radius: 12px; overflow: hidden; background: #000;">
                        <img src="http://localhost:8504/stream" style="width: 100%; display: block;" 
                            onerror="this.src='https://via.placeholder.com/800x450?text=Connecting+to+Stream...'">
                    </div>
                    <p style="text-align: center; color: #6c63ff; font-size: 0.8rem; margin-top: 5px;">
                        ⚡ High-Speed Flicker-Free Stream (MJPEG Mode)
                    </p>
                """, unsafe_allow_html=True)
            
            with col_cap:
                if st.button("✏️ Correct Latest Detection", use_container_width=True, help="If the plate is dirty or misread, update it manually."):
                    if sm.latest_detection:
                        update_plate_dialog(sm.latest_detection["text"])
                    else:
                        st.toast("No detection to correct yet")
                    
                st.markdown("**📸 Last Capture**")
                if sm.latest_full_b64:
                    st.markdown(f"""
                        <div style="border: 2px solid #34d399; border-radius: 10px; overflow: hidden;">
                            <img src="{sm.latest_full_b64}" style="width: 100%; display: block;">
                        </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown("""
                        <div style="height: 150px; background: rgba(255,255,255,0.05); border-radius: 10px; display: flex; align-items: center; justify-content: center; color: #6b7280; font-size: 0.8rem; text-align: center; border: 1px dashed rgba(255,255,255,0.2);">
                            No captures yet
                        </div>
                    """, unsafe_allow_html=True)
        else:
            st.info("Streaming is paused. Click 'Start Live' to view.")

        # 3. Plate detection log
        log = sm.detection_log
        if log:
            st.markdown("#### 🪪 Recent Detections")
            for plate, conf, ts in log[:5]: # Show top 5 in pretty cards
                cls = confidence_class(conf)
                lbl = confidence_label(conf)
                st.markdown(f"""
                <div class='result-card'>
                    <div style='color:#9ca3af;font-size:0.75rem;margin-bottom:4px'>🕐 {ts}</div>
                    <div class='plate-text'>{plate}</div>
                    <span class='confidence-badge {cls}'>{lbl} — {conf:.2%}</span>
                </div>""", unsafe_allow_html=True)
            
            # Full log table
            st.markdown("---")
            st.markdown(f"#### 📋 Full Plate Log ({len(log)} entries)")
            import pandas as pd
            log_df = pd.DataFrame(log, columns=["Plate", "Confidence", "Timestamp"])
            st.dataframe(log_df, use_container_width=True, height=300)

    # Render the fragment
    update_live_ui()

    # ── Plate detection log ───────────────────────────────────────────────
    log = st.session_state.cam_plate_log
    if log:
        st.markdown("---")
        st.markdown(f"#### 📋 Plate Log ({len(log)} entries)")
        import pandas as pd
        log_df = pd.DataFrame(
            [(p, f"{c:.2%}", ts) for p, c, ts in reversed(log[-50:])],
            columns=["Plate", "Confidence", "Timestamp"]
        )
        st.dataframe(log_df, use_container_width=True, height=300)

        # Export log
        csv_bytes = log_df.to_csv(index=False).encode()
        st.download_button(
            "⬇️ Export Log as CSV",
            data=csv_bytes,
            file_name=f"plate_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv",
        )

# ════════════════════════════════════════════════════════════════════════════
# TAB 3 – Video File
# ════════════════════════════════════════════════════════════════════════════
with tab_video:
    st.markdown("### 🎬 Video File Processing")
    st.markdown("""
    <div class='info-box'>
        Upload an MP4/AVI. The app samples every N-th frame, runs ALPR,
        saves annotated frames with plates to <b>output/</b>, and lists all unique plates.
    </div>""", unsafe_allow_html=True)

    video_file = st.file_uploader("Upload video", type=["mp4", "avi", "mov", "mkv"], key="vid_up")
    frame_skip = st.slider("Process every N frames", 1, 30, 10)
    save_frames = st.checkbox("💾 Save every detected frame to output/", value=False,
                              help="May create many files for long videos")

    if video_file is not None:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
            tmp.write(video_file.read())
            tmp_path = tmp.name

        if st.button("▶️ Run ALPR on Video"):
            cap          = cv2.VideoCapture(tmp_path)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            progress     = st.progress(0, text="Analysing video...")
            all_plates   = {}  # plate_text -> max_confidence
            frame_idx    = 0
            processed    = 0
            saved_count  = 0

            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                if frame_idx % frame_skip == 0:
                    results = alpr.predict(frame)
                    has_plate = False
                    for r in results:
                        if r.ocr:
                            conf = get_conf(r.ocr)
                            if conf >= conf_threshold:
                                txt = clean_indian_plate(r.ocr.text) if clean_text else r.ocr.text
                                if txt and len(txt) >= 4:
                                    all_plates[txt] = max(all_plates.get(txt, 0), conf)
                                    has_plate = True
                    if has_plate and save_frames:
                        ann = draw_plates_no_country(frame, results)
                        save_output_image(ann, f"video_f{frame_idx:06d}")
                        saved_count += 1
                    processed += 1

                frame_idx += 1
                pct = min(frame_idx / max(total_frames, 1), 1.0)
                progress.progress(pct, text=f"Frame {frame_idx}/{total_frames}")

            cap.release()
            os.unlink(tmp_path)
            progress.empty()

            st.success(
                f"✅ Done! Processed {processed} frames · "
                f"{len(all_plates)} unique plates · {saved_count} frames saved"
            )

            if all_plates:
                st.markdown("### 🪪 All Unique Plates Detected")
                # Highlight top one
                top_p, top_c = sorted(all_plates.items(), key=lambda x: -x[1])[0]
                st.session_state.last_detected_plate = {"text": top_p, "conf": top_c}
                st.markdown(f"""
                <div class='prediction-banner' style='margin-bottom: 2rem;'>
                    <div class='label'>🥇 Best Match From Video</div>
                    <div class='plate'>{top_p}</div>
                    <div class='conf'>Confidence: {top_c:.2%} — {confidence_label(top_c)}</div>
                </div>""", unsafe_allow_html=True)

                for plate, conf in sorted(all_plates.items(), key=lambda x: -x[1]):
                    st.markdown(f"""
                    <div class='result-card'>
                        <div class='plate-text'>{plate}</div>
                        <span class='confidence-badge {confidence_class(conf)}'>
                            {confidence_label(conf)} — {conf:.2%}
                        </span>
                    </div>""", unsafe_allow_html=True)
            else:
                st.info("No plates above threshold found.")

# ════════════════════════════════════════════════════════════════════════════
# TAB 4 – Batch Test
# ════════════════════════════════════════════════════════════════════════════
with tab_batch:
    st.markdown("### 📦 Batch Image Testing")
    st.markdown("<div class='info-box'>Upload multiple images at once.</div>", unsafe_allow_html=True)

    batch_files = st.file_uploader(
        "Upload multiple images",
        type=["jpg", "jpeg", "png", "webp", "bmp"],
        accept_multiple_files=True,
        key="batch_up",
    )

    if batch_files and st.button("🚀 Run Batch Detection"):
        results_table = []
        cols          = st.columns(3)
        saved_batch   = 0

        for idx, f in enumerate(batch_files):
            file_bytes = np.frombuffer(f.read(), np.uint8)
            frame      = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
            if frame is None:
                continue

            preds     = alpr.predict(frame)
            annotated = draw_plates_no_country(frame, preds)

            # Auto-save
            if auto_save and preds:
                save_output_image(annotated, f"batch_{idx:03d}")
                saved_batch += 1

            col = cols[idx % 3]
            with col:
                st.image(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB),
                         caption=f.name, use_container_width=True)

            for r in preds:
                if r.ocr:
                    plate = clean_indian_plate(r.ocr.text) if clean_text else r.ocr.text
                    conf  = get_conf(r.ocr)
                    st.session_state.last_detected_plate = {"text": plate, "conf": conf}
                    results_table.append({
                        "File": f.name,
                        "Plate": plate,
                        "OCR Conf": f"{conf:.2%}",
                        "Det Conf": f"{r.detection.confidence:.2%}",
                    })

        if results_table:
            import pandas as pd
            st.markdown("### 📊 Summary Table")
            st.dataframe(pd.DataFrame(results_table), use_container_width=True)
            if saved_batch:
                st.markdown(f"<div class='saved-box'>💾 {saved_batch} images saved to output/</div>",
                            unsafe_allow_html=True)
        else:
            st.info("No plates detected across uploaded images.")

# ─── Footer ───────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown("""
<div style='text-align:center; color:#6b7280; font-size:0.8rem; padding: 1rem 0'>
    🚗 Indian ALPR Demo &nbsp;·&nbsp; Powered by <b>FastALPR</b> + <b>YOLO v9</b> + <b>CCT OCR</b>
    &nbsp;·&nbsp; All detections saved to <b>output/</b> folder
</div>
""", unsafe_allow_html=True)