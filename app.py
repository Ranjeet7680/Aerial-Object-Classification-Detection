import streamlit as st
import tensorflow as tf
from ultralytics import YOLO
from PIL import Image
import numpy as np
import os
import cv2

st.set_page_config(
    page_title="AeroSight AI | Enterprise Detection Suite",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Enterprise-grade CSS (Clean, Professional, Corporate)
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');
    
    * {
        font-family: 'Inter', sans-serif;
    }
    
    /* Smooth Initial Fade-In Animation */
    @keyframes pageFadeIn {
        from { opacity: 0; transform: translateY(10px); }
        to { opacity: 1; transform: translateY(0); }
    }
    
    .stApp {
        background-color: #f8fafc;
        color: #0f172a;
        animation: pageFadeIn 0.8s ease-out forwards;
    }
    
    /* Custom Splash Screen Simulation */
    #loading-screen {
        position: fixed;
        top: 0;
        left: 0;
        width: 100vw;
        height: 100vh;
        background: #ffffff;
        display: flex;
        flex-direction: column;
        justify-content: center;
        align-items: center;
        z-index: 999999;
        animation: splash-fade 2s forwards;
        pointer-events: none;
    }
    
    @keyframes splash-fade {
        0% { opacity: 1; }
        80% { opacity: 1; }
        100% { opacity: 0; display: none; }
    }
    
    .loader-pulse {
        width: 60px;
        height: 60px;
        background: #2563eb;
        border-radius: 50%;
        animation: pulse 1.5s infinite;
    }
    
    @keyframes pulse {
        0% { transform: scale(0.8); opacity: 1; }
        100% { transform: scale(2.4); opacity: 0; }
    }
    
    /* Developer Badge Styling */
    .dev-badge {
        background: linear-gradient(135deg, #1e293b 0%, #334155 100%);
        color: #f8fafc;
        padding: 0.5rem 1rem;
        border-radius: 0.5rem;
        font-size: 0.75rem;
        font-weight: 600;
        letter-spacing: 0.05em;
        text-align: center;
        margin-top: 1rem;
    }
    
    /* Rest of the Professional UI */
    [data-testid="stSidebar"] {
        background-color: #ffffff !important;
        border-right: 1px solid #e2e8f0;
    }
    
    .hero-title {
        color: #0f172a;
        font-size: 3.5rem;
        font-weight: 800;
        letter-spacing: -0.04em;
        text-align: center;
        margin-top: -1rem;
        margin-bottom: 5px;
    }
    
    .hero-subtitle {
        color: #64748b;
        font-size: 1.1rem;
        text-align: center;
        margin-bottom: 3rem;
    }
    
    /* Crisp White Panels */
    .enterprise-panel {
        background: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 1rem;
        padding: 2.5rem;
        box-shadow: 0 1px 3px 0 rgba(0, 0, 0, 0.1), 0 1px 2px 0 rgba(0, 0, 0, 0.06);
    }
    
    /* Functional HUD Labels */
    .hud-label {
        font-size: 0.75rem;
        text-transform: uppercase;
        font-weight: 600;
        letter-spacing: 0.05em;
        color: #94a3b8;
        margin-bottom: 0.5rem;
    }
    
    .hud-value {
        font-size: 2.5rem;
        font-weight: 700;
        color: #2563eb;
    }
    
    /* Clean Status Indicators */
    .status-dot {
        height: 8px;
        width: 8px;
        border-radius: 50%;
        display: inline-block;
        margin-right: 8px;
    }
    .ready { background-color: #10b981; }
    .missing { background-color: #ef4444; }
    
    /* Primary buttons */
    .stButton>button {
        width: 100%;
        background-color: #2563eb !important;
        color: white !important;
        border: none !important;
        border-radius: 0.5rem !important;
        font-weight: 500 !important;
        padding: 1rem !important;
        box-shadow: none !important;
    }
    
    /* Metrics panel for better clarity */
    [data-testid="stMetric"] {
        background: #f1f5f9;
        padding: 1.5rem;
        border-radius: 0.75rem;
    }
</style>

<div id="loading-screen">
    <div class="loader-pulse"></div>
    <div style="margin-top: 2rem; color: #64748b; font-weight: 500; font-size: 0.9rem; letter-spacing: 2px;">
        INITIALIZING AEROSIGHT ENGINE
    </div>
</div>

""", unsafe_allow_html=True)

# Load models with enhanced status reporting
@st.cache_resource
def load_models():
    cls_models = {}
    if os.path.exists('transfer_learning_model.keras'):
        cls_models['Neural Vision (MobileNetV2)'] = tf.keras.models.load_model('transfer_learning_model.keras')
    if os.path.exists('custom_cnn_model.keras'):
        cls_models['Standard Aero-Net'] = tf.keras.models.load_model('custom_cnn_model.keras')
        
    yolo_model = None
    yolo_best = 'runs/detect/aerial_yolo_model/weights/best.pt'
    if os.path.exists(yolo_best):
        yolo_model = YOLO(yolo_best)
    elif os.path.exists('yolov8n.pt'):
        yolo_model = YOLO('yolov8n.pt')
            
    return cls_models, yolo_model

cls_models, yolo_model = load_models()

# Futuristic Sidebar
with st.sidebar:
    st.markdown("<div style='text-align: center; padding: 2rem 0;'><h2 style='letter-spacing: 1px; margin-bottom: 0px;'>AERO-SIGHT</h2><div class='dev-badge'>DEVELOPED BY RANJEET KUMAR</div><p style='color: #64748b; font-size: 0.8rem; margin-top: 10px;'>Neural Processing Core</p></div>", unsafe_allow_html=True)
    
    with st.expander("🛠️ Core Configuration", expanded=True):
        if cls_models:
            selected_cls = st.selectbox("Neural Architecture", list(cls_models.keys()))
            cls_model = cls_models[selected_cls]
        else:
            st.error("Neural weights missing.")
            cls_model = None
            
        conf_threshold = st.slider("Detection Threshold", 0.05, 1.0, 0.25, 0.05)
    
    st.markdown("### System Diagnostics")
    c1, c2 = st.columns(2)
    with c1:
        st.markdown(f"<div class='status-dot {'ready' if cls_models else 'missing'}'></div> Class", unsafe_allow_html=True)
    with c2:
        st.markdown(f"<div class='status-dot {'ready' if yolo_model else 'missing'}'></div> YOLO", unsafe_allow_html=True)

# Main Dashboard Interface
st.markdown("<h1 class='hero-title'>AEROSIGHT AI</h1>", unsafe_allow_html=True)
st.markdown("<p class='hero-subtitle'>Autonomous Aerial Object Classification & Tactical Identification<br><span style='font-size: 0.8rem; letter-spacing: 2px; color: #94a3b8;'>DEVELOPED BY RANJEET KUMAR</span></p>", unsafe_allow_html=True)

# Central scanner terminal
col_left, col_center, col_right = st.columns([1, 4, 1])

with col_center:
    st.markdown("<div class='enterprise-panel'>", unsafe_allow_html=True)
    uploaded_file = st.file_uploader("UPLOAD AERIAL IMAGERY FOR NEURAL SCANNING", type=["jpg", "jpeg", "png", "webp"], label_visibility="collapsed")
    
    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert('RGB')
        
        # Display analysis progress
        res_col1, res_col2 = st.columns([1, 1], gap="medium")
        
        with res_col1:
            st.markdown("<p class='hud-label'>Target Feed</p>", unsafe_allow_html=True)
            st.image(image, use_column_width=True)
            
        with res_col2:
            st.markdown("<p class='hud-label'>Neural Analysis Results</p>", unsafe_allow_html=True)
            
            # Classification
            if cls_model:
                img_cls = image.resize((224, 224))
                img_array = np.array(img_cls) / 255.0
                img_array = np.expand_dims(img_array, axis=0)
                
                prediction = cls_model.predict(img_array, verbose=0)[0][0]
                label = "DRONE" if prediction > 0.5 else "BIRD"
                confidence = prediction if prediction > 0.5 else (1 - prediction)
                
                accent = "#2563eb" if label == "DRONE" else "#10b981"
                st.markdown(f"""
                    <div style="background: #ffffff; padding: 1.5rem; border-radius: 0.75rem; border: 1px solid #e2e8f0; border-left: 4px solid {accent};">
                        <p class="hud-label" style="color: {accent};">Classification Analysis</p>
                        <p class="hud-value" style="color: #0f172a; font-size: 2rem;">{label}</p>
                        <p style="font-size: 0.85rem; color: #64748b; margin-top: 5px;">Confidence: <strong>{confidence:.1%}</strong></p>
                    </div>
                """, unsafe_allow_html=True)
            
            st.markdown("<div style='margin: 1.5rem 0;'></div>", unsafe_allow_html=True)

            # YOLO detection block
            if yolo_model:
                results = yolo_model.predict(image, conf=conf_threshold, verbose=False)
                res_rgb = cv2.cvtColor(results[0].plot(), cv2.COLOR_BGR2RGB)
                st.image(res_rgb, caption="Tactical Localization", use_column_width=True)
                
                # Dynamic Stats Bar
                names = yolo_model.names
                counts = {}
                for box in results[0].boxes:
                    c_name = names[int(box.cls[0])]
                    counts[c_name] = counts.get(c_name, 0) + 1
                
                if counts:
                    st.markdown("<p class='hud-label'>Detected Objects</p>", unsafe_allow_html=True)
                    for name, count in counts.items():
                        st.markdown(f"""
                            <div style="display: flex; justify-content: space-between; align-items: center; background: rgba(37,99,235,0.05); padding: 0.8rem 1.2rem; border-radius: 0.8rem; margin-bottom: 0.5rem; border: 1px solid rgba(37,99,235,0.1);">
                                <span style="font-weight: 600; color: #1e40af;">{name.upper()}</span>
                                <span style="background: #2563eb; color: white; padding: 0.2rem 0.8rem; border-radius: 1rem; font-size: 0.9rem; font-weight: 700;">{count}</span>
                            </div>
                        """, unsafe_allow_html=True)
                else:
                    st.info("Passive scan: No objects localized.")
    else:
        st.markdown("""
            <div style="text-align: center; padding: 4rem 1rem;">
                <h3 style="color: #6366f1; letter-spacing: 2px;">SCANNER READY</h3>
                <p style="color: #94a3b8;">Telemetry awaiting optical input. Drag and drop imagery to initiate tactical analysis.</p>
                <div style="margin-top: 2rem; opacity: 0.3;">
                    <span style="font-size: 3rem;">🛰️ 🛡️ 🔍</span>
                </div>
            </div>
        """, unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

# Footer info
st.markdown("<div style='margin-top: 6rem; text-align: center; border-top: 1px solid rgba(0,0,0,0.05); padding-top: 2rem;'>", unsafe_allow_html=True)
st.markdown("<p style='font-size: 0.8rem; color: #475569; letter-spacing: 3px;'>AEROSIGHT CORE V3.0 • ENTERPRISE NEURAL ENGINE<br><span style='color: #2563eb; font-weight: 700;'>DEVELOPED BY RANJEET KUMAR</span></p>", unsafe_allow_html=True)
st.markdown("</div>", unsafe_allow_html=True)
