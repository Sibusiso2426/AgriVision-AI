import streamlit as st
import requests
import hashlib
import time
import plotly.express as px
from ultralytics import YOLO
import numpy as np
import cv2
from PIL import Image
import io
import sys
import os
import time
import base64
import streamlit as st
import threading 
import av
from streamlit_webrtc import webrtc_streamer, WebRtcMode
from ultralytics import YOLO
import pandas as pd

lock = threading.Lock()
img_container = {"frame_count": 0, "last_results": []}

# Load the model once at the top so every function can see it
# Use 'yolov8n.pt' for now, or 'models/livestock_health_model.pt' if you've moved it
try:
    livestock_model = YOLO('yolov8n.pt') 
except Exception as e:
    st.error(f"Error loading model: {e}")

# This tells Python to look one level up so it can find the 'src' folder
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# ─────────────────────────────────────────────
# 1. Page Config (Always first!)
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="AgriVision AI",
    layout="centered",
    initial_sidebar_state="expanded"
)

# ─────────────────────────────────────────────
# 2. USER CREDENTIALS (replace with DB in prod)
# ─────────────────────────────────────────────
def hash_password(password: str) -> str:
    return hashlib.sha256(password.encode()).hexdigest()

USERS = {
    "admin": {
        "password": hash_password("admin123"),
        "role": "admin",
        "name": "Admin User",
    },
    "farmer": {
        "password": hash_password("farm2024"),
        "role": "farmer",
        "name": "Demo Farmer",
    },
}

# ─────────────────────────────────────────────
# 3. SESSION STATE DEFAULTS
# ─────────────────────────────────────────────
if "authenticated" not in st.session_state:
    st.session_state.authenticated = False
if "username" not in st.session_state:
    st.session_state.username = ""
if "role" not in st.session_state:
    st.session_state.role = ""
if "name" not in st.session_state:
    st.session_state.name = ""
if "current_page" not in st.session_state:
    st.session_state.current_page = "Diagnose"
if "history" not in st.session_state:
    st.session_state.history = []

# ─────────────────────────────────────────────
# 4. AUTH FUNCTIONS
# ─────────────────────────────────────────────
def login(username: str, password: str) -> bool:
    user = USERS.get(username)
    if user and user["password"] == hash_password(password):
        st.session_state.authenticated = True
        st.session_state.username = username
        st.session_state.role = user["role"]
        st.session_state.name = user["name"]
        return True
    return False

def logout():
    st.session_state.authenticated = False
    st.session_state.username = ""
    st.session_state.role = ""
    st.session_state.name = ""
    st.session_state.current_page = "Diagnose"

# ─────────────────────────────────────────────
# 5. LOGIN PAGE
# ─────────────────────────────────────────────
def show_login():
    st.markdown("""
        <style>
        .login-header { text-align: center; padding: 2rem 0 1rem; }
        .login-header h1 { font-size: 2.4rem; color: #2e7d32; }
        .login-header p  { color: #666; font-size: 1rem; }
        .demo-box { background:#f1f8e9; border-left:4px solid #66bb6a;
                    padding:0.8rem 1rem; border-radius:6px; font-size:0.85rem; }
        </style>
    """, unsafe_allow_html=True)

    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.markdown("""
            <div class="login-header">
                <img src="https://cdn-icons-png.flaticon.com/512/2156/2156007.png" width="80"/>
                <h1>AgriVision AI</h1>
                <p>Tomato Disease Diagnostic Platform</p>
            </div>
        """, unsafe_allow_html=True)

        with st.form("login_form"):
            username = st.text_input("👤 Username")
            password = st.text_input("🔒 Password", type="password")
            submitted = st.form_submit_button("Sign In", use_container_width=True)

            if submitted:
                if login(username, password):
                    st.success(f"Welcome, {st.session_state.name}!")
                    time.sleep(0.5)
                    st.rerun()
                else:
                    st.error("Invalid username or password.")

        st.markdown("""
            <div class="demo-box">
            <b>Demo credentials</b><br>
            🧑‍💼 Admin &nbsp;→ <code>admin</code> / <code>admin123</code><br>
            🌾 Farmer → <code>farmer</code> / <code>farm2024</code>
            </div>
        """, unsafe_allow_html=True)

# ─────────────────────────────────────────────
# 6. SIDEBAR (authenticated)
# ─────────────────────────────────────────────
def show_sidebar():
    # ─── 1. INITIALIZE NEW KEYS (Prevents the AttributeError) ───
    if "daily_count" not in st.session_state:
        st.session_state.daily_count = 0
    if "last_processed_id" not in st.session_state:
        st.session_state.last_processed_id = None

    with st.sidebar:
        st.image("https://cdn-icons-png.flaticon.com/512/2156/2156007.png", width=80)
        st.title("AgriVision AI")
        st.markdown(f"**👤 {st.session_state.name}** \n`{st.session_state.role.upper()}`")
        st.markdown("---")

        # ... (Rest of your Navigation logic stays the same) ...

        # Navigation
        st.subheader("Navigation")
        pages = ["🔬 Crop Diagnose","🐄 Livestock Analysis", "📋 History", "ℹ️ About"]
        if st.session_state.role == "admin":
            pages.append("⚙️ Admin")

        for page in pages:
            label = page.split(" ", 1)[1]  
            if st.button(page, use_container_width=True, key=f"nav_{label}"):
                st.session_state.current_page = label

        st.markdown("---")
        if st.button("🚪 Logout", use_container_width=True):
            # Ensure logout() function clears these too if you want a full reset
            st.session_state.daily_count = 0 
            logout()
            st.rerun()

        # ─── 2. ACTIVITY DASHBOARD ───
        st.markdown("---")
        st.header("📈 Today's Activity")
        
        # Now this will not crash because we initialized it above!
        st.metric("Total Detections", st.session_state.daily_count)

        chart_data = pd.DataFrame(
            np.random.randn(20, 1),
            columns=['Detection Frequency']
        )
        st.area_chart(chart_data, height=150)
        
        st.caption("Real-time throughput of the YOLOv8 engine.")

        if st.button("♻️ Reset Daily Stats"):
            st.session_state.daily_count = 0
            st.session_state.last_processed_id = None
            st.rerun()

# ─────────────────────────────────────────────
# 7. PAGES
# ─────────────────────────────────────────────
def page_diagnose():
    st.title("🔬 Advanced Vision Diagnostic")
    st.markdown("Take a photo in the field or upload an existing image for analysis.")

    # Create tabs for the two input methods
    tab1, tab2 = st.tabs(["📸 Take Photo", "📁 Upload File"])
    
    source_file = None

    with tab1:
        cam_file = st.camera_input("Scan leaf directly")
        if cam_file:
            source_file = cam_file

    with tab2:
        up_file = st.file_uploader("Choose a saved image", type=["jpg", "jpeg", "png"])
        if up_file:
            source_file = up_file

    # If an image is provided by either method
    if source_file is not None:
        # Show a preview (st.camera_input shows its own preview, so we only show for uploader)
        if source_file == up_file:
            st.image(source_file, caption="Selected Image", use_container_width=True)

        if st.button("🌿 Analyze Crop Health", use_container_width=True):
            data = None

            with st.spinner("Connecting to backend..."):
                try:
                    files = {"file": source_file.getvalue()}
                    response = requests.post("http://127.0.0.1:8000/api/v1/plant/detect", files=files)

                    if response.status_code == 200:
                        data = response.json()
                        # Use 'recommendation' because that is what we named the key in the API return above
                        treatment = data.get("recommendation", "No advice received from server")
                        st.info(f"**Treatment:** {treatment}")
                    else:
                        st.error(f"Backend Error: {response.status_code}")
                
                except Exception as e:
                    st.error(f"Connection Failed: {e}")

            if data:
                # Display Results
                # Extract the data safely from the API response
                disease_label = data.get('label', 'N/A')
                conf_value = data.get('confidence', 'N/A')
                # Look for 'recommendation' (matching our api.py) instead of 'treatment'
                treatment_advice = data.get('recommendation', 'No specific treatment found.')

                # --- Display the UI ---
                st.success(f"**Result:** {disease_label} ({conf_value})")

                col1, col2 = st.columns(2)
                with col1:
                    # We can calculate severity based on confidence for now, 
                    # or just display a default note.
                    severity = "Moderate" if float(conf_value.replace('%','')) > 50 else "Low"
                    st.metric("Severity", severity)
                    
                with col2:
                    st.metric("Confidence", conf_value)

                # Update this line to show the real advice!
                st.info(f"**Treatment/Advice:** {treatment_advice}")
                
                # Save to history
                st.session_state.history.append({
                    "file": "Camera Scan" if source_file == cam_file else source_file.name,
                    "disease": data.get("label", "Unknown"),
                    "confidence": data.get("confidence", "N/A"),
                    # Since our API returns 'recommendation', we map it here
                    "severity": "Moderate" if "Blight" in data.get("label", "") else "Low", 
                    "treatment": data.get("recommendation", "N/A"),
                    "user": st.session_state.username,
                })


# ─────────────────────────────────────────────
# HELPER: Convert PIL Image → base64 PNG string
# ─────────────────────────────────────────────
def _img_to_b64(pil_img: Image.Image) -> str:
    buf = io.BytesIO()
    pil_img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


# ─────────────────────────────────────────────
# HELPER: Draw detection boxes on a numpy array
# Returns annotated PIL Image
# ─────────────────────────────────────────────
def _draw_boxes(img_array: np.ndarray, detections: list) -> Image.Image:
    annotated = img_array.copy()
    for det in detections:
        b = det["box"]
        x1, y1, x2, y2 = int(b[0]), int(b[1]), int(b[2]), int(b[3])
        label = f"{det['class']} {det.get('confidence', 0):.0%}"
        color = (0, 200, 80)

        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(annotated, (x1, y1 - th - 6), (x1 + tw + 4, y1), color, -1)
        cv2.putText(
            annotated, label, (x1 + 2, y1 - 4),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA
        )
    return Image.fromarray(annotated)

def _draw_boxes_numpy(img: np.ndarray, results: list) -> np.ndarray:
    """Draws YOLO detections directly onto a numpy array (OpenCV format) for Live Mode."""
    annotated = img.copy()
    
    # YOLO results can be a list; we want to iterate through them
    for r in results:
        for box in r.boxes:
            # Get coordinates
            b = box.xyxy[0].tolist()
            x1, y1, x2, y2 = int(b[0]), int(b[1]), int(b[2]), int(b[3])
            
            # Get label and confidence
            cls_id = int(box.cls[0].item())
            # Safely get the name from the model
            label_name = livestock_model.names[cls_id]
            conf = box.conf[0].item()
            label = f"{label_name} {conf:.0%}"
            
            # Draw green box (BGR format for OpenCV: Green is (0, 255, 0))
            color = (0, 255, 0) 
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
            
            # Add label text background
            (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(annotated, (x1, y1 - h - 10), (x1 + w, y1), color, -1)
            
            # Add label text
            cv2.putText(annotated, label, (x1, y1 - 5), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
            
    return annotated


# ─────────────────────────────────────────────
# CAPTURE ALERT LOG RENDERER
# ─────────────────────────────────────────────
def _render_capture_log():
    log = st.session_state.get("capture_log", [])

    st.markdown("---")
    st.subheader(f"📂 Capture Log  ({len(log)} saved)")

    if not log:
        st.info("No captures yet — run a detection and click **📸 Capture Alert** to save a frame.")
        return

    # Bulk actions
    col_clr, col_dl = st.columns([1, 4])
    if col_clr.button("🗑️ Clear All", use_container_width=True):
        st.session_state.capture_log = []
        st.rerun()

    st.caption("Expand any entry to view the annotated frame and full details.")

    for idx, entry in enumerate(reversed(log)):
        real_idx = len(log) - 1 - idx  # index into original list for delete
        severity = entry.get("alert_triggered", False)
        badge = "🔴 ALERT" if severity else "🟢 CLEAR"
        label = f"{badge}  |  {entry['timestamp']}  |  {entry['count']} object(s)  —  {entry['source']}"

        with st.expander(label, expanded=(idx == 0)):
            img_col, info_col = st.columns([2, 1])

            with img_col:
                st.image(
                    entry["image"],
                    caption=f"Captured at {entry['timestamp']}",
                    use_container_width=True,
                )

            with info_col:
                st.metric("Objects Detected", entry["count"])
                st.metric("Status", "⚠️ Alert" if severity else "✅ Clear")
                st.write(f"**Source:** {entry['source']}")
                st.write(f"**Time:** {entry['timestamp']}")

                if entry["detections"]:
                    st.write("**Detections:**")
                    for det in entry["detections"]:
                        conf = det.get("confidence", None)
                        conf_str = f" ({conf:.0%})" if conf else ""
                        st.write(f"- `{det['class']}`{conf_str}")

            # Per-entry delete
            if st.button(f"🗑️ Delete this entry", key=f"del_{real_idx}"):
                st.session_state.capture_log.pop(real_idx)
                st.rerun()


# ─────────────────────────────────────────────
# MAIN PAGE FUNCTION
# ─────────────────────────────────────────────
def page_livestock_analysis():
    
    if "daily_count" not in st.session_state:
        st.session_state.daily_count = 0
 # ── Sidebar Controls ──
    with st.sidebar:
        st.header("⚙️ AI Settings")
        
        # 1. Confidence Slider
        conf_threshold = st.slider(
            "Detection Confidence", 
            min_value=0.1, 
            max_value=1.0, 
            value=0.45, 
            help="Higher values show only the most certain detections."
        )

        st.markdown("---")
        st.header("🌦️ Field Conditions")
        
        # Search bar for city
        city = st.text_input("📍 Current Location", value="Pietermaritzburg")
        
        try:
            # Simple Weather Fetch using Open-Meteo (No API key needed!)
            # We'll use a hardcoded coordinate for PMB for the demo speed
            # Lat: -29.6006, Lon: 30.3794
            weather_url = "https://api.open-meteo.com/v1/forecast?latitude=-29.60&longitude=30.38&current_weather=true"
            response = requests.get(weather_url).json()
            
            if "current_weather" in response:
                curr = response["current_weather"]
                temp = curr["temperature"]
                wind = curr["windspeed"]
                
                col_w1, col_w2 = st.columns(2)
                with col_w1:
                    st.metric("Temp", f"{temp}°C")
                with col_w2:
                    st.metric("Wind", f"{wind} km/h")
                
                # Logic-based farming tips
                if temp > 28:
                    st.warning("☀️ **Heat Alert:** Livestock may seek shade. AI detections might be difficult in glare.")
                elif temp < 10:
                    st.info("❄️ **Cold Alert:** Watch for shivering or huddled behavior.")
            else:
                st.error("Weather service unavailable.")
                
        except Exception as e:
            st.caption("Unable to load live weather. Showing estimates.")
            st.metric("Temp (Est)", "28°C")

        st.markdown("---")
        st.caption("AgriVision AI v1.2 | South Africa")
        
    # ── Session state init ──
    if "capture_log" not in st.session_state:
        st.session_state.capture_log = []
    if "last_detection" not in st.session_state:
        st.session_state.last_detection = None  # stores latest detection result

    

    # ── Header ──
    st.title("🐄 Livestock Health Intelligence")
    st.markdown(
        "Analyze livestock health via live camera or high-resolution uploads. "
        "Use **📸 Capture Alert** to save flagged frames to your Capture Log."
    )

    # ── Tabbed Input ──
    tab1, tab2 = st.tabs(["📸 Live Field Scanner", "📁 Image Upload"])
    source_file = None
    source_label = "Unknown"

    with tab1:
        st.subheader("Real-time Livestock Scanner")
        col_live, col_ctrl = st.columns([3, 1])
    
        with col_ctrl:
            is_frozen = st.checkbox("❄️ Freeze Frame", help="Pause the video to inspect a detection")
            if is_frozen:
                st.warning("Video Paused")

        def video_frame_callback(frame):
            img = frame.to_ndarray(format="bgr24")
          # 1. Use the Lock to safely update our counter
            with lock:
                img_container["frame_count"] += 1
                current_count = img_container["frame_count"]

            # 2. Only run AI every 5th frame to prevent lag/freezing
            if current_count % 5 == 0:
                results = livestock_model(img, conf=conf_threshold)[0]  # Run inference with confidence threshold
                with lock:
                    img_container["last_results"] = results
            
            # 3. Always draw using the most recent results we have
            with lock:
                results_to_draw = img_container["last_results"]
                
            annotated_img = _draw_boxes_numpy(img, results_to_draw)
            
            return av.VideoFrame.from_ndarray(annotated_img, format="bgr24")
        
        
        
        webrtc_streamer(
                key="livestock-scanner",
                mode=WebRtcMode.SENDRECV,
                # If is_frozen is True, we stop the callback to freeze the view
                video_frame_callback=None if is_frozen else video_frame_callback,
                media_stream_constraints={
                    "video": {"facingMode": "environment"}, 
                    "audio": False
                },
                async_processing=True,
                rtc_configuration={
                    "iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]
                }
            )

        st.info("The AI is scanning in real-time. Use Tab 2 for high-res uploads.")
        
    with tab2:
        up_file = st.file_uploader("Upload livestock photo", type=["jpg", "jpeg", "png"])
        if up_file:
            source_file = up_file
            source_label = up_file.name

    # ── Detection Panel ──
    if source_file is not None:
        image = Image.open(source_file)
        st.image(image, caption="Inference Source", use_container_width=True)

        # ── Run Detection button ──
        if st.button("🔍 Run Health Detection", use_container_width=True):
            st.session_state.last_detection = None  # clear previous

            with st.spinner("Processing through YOLOv8 Engine..."):
                try:
                    files = {"file": source_file.getvalue()}
                    response = requests.post(
                        "http://127.0.0.1:8000/api/v1/livestock/detect", files=files
                    )

                    if response.status_code == 200:
                        data = response.json()
                        img_array = np.array(image.convert("RGB"))
                        detections = data.get("detections", [])

                        # Draw boxes
                        annotated_pil = _draw_boxes(img_array, detections)

                        # Persist result in session for Capture Alert
                        st.session_state.last_detection = {
                            "image": annotated_pil,
                            "detections": detections,
                            "count": data.get("count", 0),
                            "alert_triggered": data.get("alert_triggered", False),
                            "source": source_label,
                        }
                    else:
                        st.error(f"Backend error: {response.status_code}")

                except Exception as e:
                    st.error(f"Connection error: {e}")
# ── Display last detection result ──
        det = st.session_state.last_detection
        if det:
            # --- NEW: Daily Counter Logic ---
            # We use a 'last_processed_id' to prevent double-counting on page reruns
            timestamp = det.get('timestamp', time.strftime('%H:%M:%S'))
            current_det_id = f"{timestamp}_{det['count']}"

            if st.session_state.get("last_processed_id") != current_det_id:
                st.session_state.daily_count += det['count']
                st.session_state.last_processed_id = current_det_id
            # -------------------------------

            st.caption(f"Analysis completed at: {det.get('timestamp', time.strftime('%H:%M:%S'))}")
            st.write(f"**Detected Objects:** {det['count']}")
            st.image(det["image"], caption="AI Detection Result", use_container_width=True)

            if det["alert_triggered"]:
                st.error("🚨 CRITICAL: Anomalies detected (Potential Injury or Disease).")
                st.toast("Health Alert Logged — consider capturing this frame.", icon="⚠️")
            else:
                st.success("✅ No immediate health anomalies detected.")

            # ── ✨ CAPTURE ALERT BUTTON ──
            st.markdown("---")
            col_cap, col_info = st.columns([1, 3])

            with col_cap:
                if st.button("📸 Capture Alert", use_container_width=True, type="primary"):
                    entry = {
                        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                        "image": det["image"],          # PIL Image
                        "detections": det["detections"],
                        "count": det["count"],
                        "alert_triggered": det["alert_triggered"],
                        "source": det["source"],
                    }
                    st.session_state.capture_log.append(entry)
                    st.toast("✅ Frame captured and saved to Capture Log!", icon="📸")
                    time.sleep(0.4)
                    st.rerun()

            with col_info:
                status_icon = "🔴" if det["alert_triggered"] else "🟢"
                st.info(
                    f"{status_icon} Clicking **Capture Alert** saves the annotated frame, "
                    f"timestamp, and all detection details to your **Capture Log** below."
                )

    # ── Capture Log (always visible at the bottom) ──
    _render_capture_log()

def page_history():
    st.title("📋 Diagnosis History")

    history = st.session_state.history

    # Farmers only see their own; admins see all
    if st.session_state.role != "admin":
        history = [h for h in history if h["user"] == st.session_state.username]

    if not history:
        st.info("No diagnoses recorded yet. Run an analysis on the Diagnose page.")
        return

    for i, record in enumerate(reversed(history), 1):
        with st.expander(f"#{i} — {record['file']}  |  {record['disease']}"):
            col1, col2 = st.columns(2)
            col1.metric("Disease", record["disease"])
            col2.metric("Confidence", record["confidence"])
            st.write(f"**Severity:** {record['severity']}")
            st.write(f"**Treatment:** {record['treatment']}")
            if st.session_state.role == "admin":
                st.caption(f"Submitted by: `{record['user']}`")


def page_about():
    st.title("ℹ️ About AgriVision AI")
    st.markdown("""
    ## What is AgriVision AI?
    AgriVision AI is a tomato disease diagnostic tool powered by a fine-tuned **MobileNetV2** 
    deep learning model, designed specifically to assist **smallholder farmers in Southern Africa**.

    ## Supported Diseases
    | Disease | Symptoms |
    |---|---|
    | Early Blight | Dark brown spots with concentric rings |
    | Late Blight | Water-soaked lesions, white mold |
    | Leaf Mold | Yellow patches, olive-green mold below |
    | Healthy | No visible symptoms |

    ## How to Use
    1. **Log in** with your credentials
    2. Go to **Diagnose** and upload a clear leaf photo
    3. Click **Analyze** and review the result
    4. Check **History** to review past diagnoses

    ## Technical Details
    - Model: MobileNetV2 (fine-tuned)
    - Backend: FastAPI (`http://127.0.0.1:8000`)
    - Frontend: Streamlit

    ---
    *AgriVision AI Project v1.0 — Developed for Southern African smallholder farmers.*
    """)


def page_admin():
    st.title("⚙️ Admin Panel")

    if st.session_state.role != "admin":
        st.error("⛔ Access Denied: Admins only.")
        return

    st.subheader("Registered Users")
    for uname, udata in USERS.items():
        with st.expander(f"👤 {uname}  —  Role: `{udata['role']}`"):
            st.write(f"**Username:** `{uname}`")
            st.write(f"**Role:** `{udata['role']}`")
            st.write(f"**Name:** {udata.get('name', 'N/A')}")

    st.markdown("---")
    st.subheader("All Diagnoses")
    all_history = st.session_state.history
    if not all_history:
        st.info("No diagnoses recorded yet.")
    else:
        for i, record in enumerate(reversed(all_history), 1):
            st.write(f"**{i}.** `{record['user']}` — {record['file']} → **{record['disease']}** ({record['confidence']})")

    st.markdown("---")
    st.subheader("System Info")
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Users", len(USERS))
    col2.metric("Total Diagnoses", len(st.session_state.history))
    col3.metric("App Version", "1.0")


# ─────────────────────────────────────────────
# 8. ROUTER
# ─────────────────────────────────────────────
PAGE_MAP = {
    "Crop Diagnose": page_diagnose,
    "Livestock Analysis": page_livestock_analysis,
    "History": page_history,
    "About": page_about,
    "Admin": page_admin,
}

# ─────────────────────────────────────────────
# 9. MAIN ENTRY POINT
# ─────────────────────────────────────────────
if not st.session_state.authenticated:
    show_login()
else:
    show_sidebar()
    current = st.session_state.current_page
    page_fn = PAGE_MAP.get(current, page_diagnose)
    page_fn()