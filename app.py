import streamlit as st
import os
import cv2
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import datetime
import time
import tempfile
import io
import plotly.express as px
from PIL import Image, ImageDraw
from torchvision import transforms
from facenet_pytorch import MTCNN
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from reportlab.lib import colors

import config
from models import SpatialXception, SRMXception, DeepfakeLSTM

# ==========================================
#        PAGE CONFIG & STYLING
# ==========================================
st.set_page_config(page_title="Sentinel Pro // Forensic Engine", page_icon="🛡️", layout="wide")

st.markdown("""
    <style>
    /* Premium Glassmorphism UI */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;800&family=JetBrains+Mono&display=swap');
    
    .stApp { background-color: #0b0f19; color: #f8fafc; font-family: 'Inter', sans-serif; }
    
    .css-1d391kg { background-color: #111827 !important; border-right: 1px solid rgba(255,255,255,0.05); }
    
    h1, h2, h3 { color: #ffffff; font-weight: 800; letter-spacing: -0.5px; }
    
    .card { 
        background: rgba(30, 41, 59, 0.4); 
        backdrop-filter: blur(16px);
        -webkit-backdrop-filter: blur(16px);
        border: 1px solid rgba(255, 255, 255, 0.05); 
        border-radius: 16px; 
        padding: 24px; 
        margin-bottom: 24px;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06);
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }
    
    .card:hover { border-color: rgba(255,255,255,0.1); box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.2); }
    
    .metric-value { 
        font-size: 64px; 
        font-weight: 900; 
        line-height: 1; 
        margin-bottom: 8px;
        text-shadow: 0 4px 12px rgba(0,0,0,0.3);
    }
    
    .verdict-badge {
        font-size: 16px; font-weight: 800; letter-spacing: 2px; text-transform: uppercase;
        padding: 12px 24px; border-radius: 12px; display: inline-block; width: 100%; text-align: center;
        box-shadow: inset 0 2px 4px rgba(255,255,255,0.1);
    }
    
    .section-title { 
        color: #94a3b8; font-size: 11px; font-weight: 700; text-transform: uppercase; 
        letter-spacing: 2px; margin-bottom: 20px; border-bottom: 1px solid rgba(255,255,255,0.1); 
        padding-bottom: 8px;
    }
    
    .log-box { 
        background-color: rgba(15, 23, 42, 0.8); border: 1px solid rgba(255,255,255,0.05); 
        border-radius: 12px; padding: 16px; font-family: 'JetBrains Mono', monospace; 
        font-size: 13px; height: 220px; overflow-y: auto; color: #cbd5e1;
        box-shadow: inset 0 2px 4px 0 rgba(0,0,0,0.2);
    }
    
    .log-success { color: #10b981; font-weight: 600; text-shadow: 0 0 10px rgba(16,185,129,0.3); }
    .log-error { color: #ef4444; font-weight: 600; text-shadow: 0 0 10px rgba(239,68,68,0.3); }
    .log-info { color: #6366f1; }
    .log-time { color: #64748b; font-size: 11px; margin-right: 8px; }
    
    div[data-testid="stMetricValue"] { font-size: 28px; font-weight: 800; }
    
    /* Customize progress bar colors */
    .stProgress .st-bo { background-color: #3b82f6; }
    </style>
""", unsafe_allow_html=True)

# ==========================================
#        SESSION STATE INITIALIZATION
# ==========================================
if 'logs' not in st.session_state:
    st.session_state.logs = []
if 'results' not in st.session_state:
    st.session_state.results = None
if 'history' not in st.session_state:
    st.session_state.history = []
if 'video_path' not in st.session_state:
    st.session_state.video_path = None

def add_log(msg, level="info"):
    t = datetime.datetime.now().strftime("%H:%M:%S")
    color_class = "log-info"
    if level == "success": color_class = "log-success"
    if level == "error": color_class = "log-error"
    st.session_state.logs.append(f"<span class='log-time'>[{t}]</span> <span class='{color_class}'>{msg}</span>")

# ==========================================
#        MODEL LOADING (CACHED)
# ==========================================
@st.cache_resource(show_spinner="Initializing Tensor Cores & Loading Models...")
def load_models():
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    models = {}
    
    class GUIWrapper(nn.Module):
        def __init__(self, model, mode='spatial'):
            super().__init__()
            self.model = model
            self.mode = mode
        def forward(self, x):
            if self.mode == 'spatial':
                x = self.model.forward_features(x)
                f = self.model.global_pool(x)
                l = self.model.fc(f)
                return f, l
            elif self.mode == 'srm':
                noise = self.model.compress(self.model.srm(x))
                x = self.model.backbone.forward_features(noise)
                f = self.model.backbone.global_pool(x)
                l = self.model.backbone.fc(f)
                return f, l

    try:
        # Spatial
        m1 = SpatialXception(num_classes=2, pretrained=False).model
        m1.load_state_dict(torch.load(config.SPATIAL_MODEL_PATH, map_location=device))
        models['spatial'] = GUIWrapper(m1, 'spatial').eval().to(device)

        # SRM
        m2 = SRMXception(num_classes=1, pretrained=False)
        m2.load_state_dict(torch.load(config.SRM_MODEL_PATH, map_location=device))
        models['srm'] = GUIWrapper(m2, 'srm').eval().to(device)

        # LSTM
        m3 = DeepfakeLSTM().to(device)
        m3.load_state_dict(torch.load(config.TEMPORAL_MODEL_PATH, map_location=device))
        models['lstm'] = m3.eval()
        
        # MTCNN
        models['mtcnn'] = MTCNN(keep_all=False, select_largest=True, device=device, margin=14)
        return models, device
    except Exception as e:
        st.error(f"FATAL ERROR: Model architecture load failed -> {e}")
        return None, device

models, DEVICE = load_models()

# ==========================================
#        INFERENCE ENGINE
# ==========================================
def extract_frames(video_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        add_log("Unable to open video stream", "error")
        return []
    
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    indices = np.linspace(0, total-1, config.SEQ_LENGTH, dtype=int) if total > config.SEQ_LENGTH else range(total)
    
    frames = []
    idx = 0
    while True:
        ret, frame = cap.read()
        if not ret: break
        if idx in indices: 
            frames.append(Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))
        idx += 1
    cap.release()

    while len(frames) < config.SEQ_LENGTH and len(frames) > 0: 
        frames.append(frames[-1].copy())
        
    return frames

def run_analysis(video_path):
    add_log(">>> INITIATING DEEP FORENSIC SCAN...", "info")
    
    trans_spatial = transforms.Compose([
        transforms.Resize((config.IMG_SIZE, config.IMG_SIZE)), 
        transforms.ToTensor(), 
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    trans_srm = transforms.Compose([
        transforms.Resize((config.IMG_SIZE, config.IMG_SIZE)), 
        transforms.ToTensor()
    ])

    frames = extract_frames(video_path)
    if not frames: return

    batch_s, batch_f = [], []
    valid_indices = []
    
    add_log(">>> EXTRACTING FACIAL BIOMETRICS...", "info")
    progress_bar = st.progress(0)
    status_text = st.empty()

    thumb_saved = False
    
    # 1. Face Extraction & Preprocessing
    for i, f in enumerate(frames):
        status_text.text(f"Isolating Subject [Frame {i+1}/{config.SEQ_LENGTH}]")
        boxes, _ = models['mtcnn'].detect(f)
        
        if boxes is not None: 
            face = f.crop(boxes[0])
            face_final = face.resize((config.IMG_SIZE, config.IMG_SIZE), Image.BILINEAR)
            valid_indices.append(i)
        else: 
            face_final = f.resize((config.IMG_SIZE, config.IMG_SIZE), Image.BILINEAR)
        
        if not thumb_saved and boxes is not None:
             draw = ImageDraw.Draw(f)
             draw.rectangle(boxes[0].tolist(), outline="#3b82f6", width=6)
             f.save("temp_thumb.jpg")
             thumb_saved = True
             
        batch_s.append(trans_spatial(face_final))
        batch_f.append(trans_srm(face_final))
        progress_bar.progress((i + 1) / config.SEQ_LENGTH)

    if not thumb_saved and frames:
        frames[0].save("temp_thumb.jpg")

    status_text.text("Constructing Tensor Batch...")
    inp_s = torch.stack(batch_s).to(DEVICE)
    inp_f = torch.stack(batch_f).to(DEVICE)

    add_log(">>> EXECUTING NEURAL INFERENCE (BATCHED)...", "info")
    status_text.text("Running Spatial & Frequency Analysis...")
    
    # 2. Batched Inference for spatial models
    with torch.no_grad():
        feat_s, log_s = models['spatial'](inp_s)
        feat_f, log_f = models['srm'](inp_f)
        
        probs_s = torch.softmax(log_s, dim=1)[:, 1].cpu().numpy()
        probs_f = torch.sigmoid(log_f).cpu().squeeze().numpy()
        
        if len(probs_f.shape) == 0:
            probs_f = np.array([probs_f])
            
        val_s = float(np.mean(probs_s))
        val_f = float(np.mean(probs_f))
        
        status_text.text("Evaluating Temporal Coherence via BiLSTM...")
        combined = torch.cat((feat_s, feat_f), dim=1).unsqueeze(0)
        val_t = torch.sigmoid(models['lstm'](combined)).item()

    frame_scores = ((probs_s + probs_f) / 2).tolist()

    final = (0.4 * val_s) + (0.4 * val_f) + (0.2 * val_t)
    add_log(f">>> ANALYSIS COMPLETE [Confidence: {final*100:.2f}%]", "success")
    status_text.empty()
    progress_bar.empty()
    
    st.session_state.results = {
        "final": final, "s": val_s, "f": val_f, "t": val_t,
        "filename": os.path.basename(video_path),
        "frame_scores": frame_scores,
        "timestamp": datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    }
    st.session_state.history.insert(0, st.session_state.results)

# ==========================================
#        PDF EXPORT PIPELINE
# ==========================================
def generate_pdf_buffer(results):
    buffer = io.BytesIO()
    c = canvas.Canvas(buffer, pagesize=letter)
    w, h = letter
    case_id = str(int(time.time()))
    final = results['final']
    
    verdict = "MANIPULATED / DEEPFAKE" if final > 0.6 else ("SUSPICIOUS ACTIVITY" if final > 0.35 else "AUTHENTIC MEDIA")
    v_color = colors.HexColor("#ef4444") if final > 0.6 else (colors.HexColor("#eab308") if final > 0.35 else colors.HexColor("#10b981"))

    # Header
    c.setFillColor(colors.HexColor("#0b0f19"))
    c.rect(0, h-90, w, 90, fill=1, stroke=0)
    c.setFillColor(colors.white)
    c.setFont("Helvetica-Bold", 24)
    c.drawString(40, h-50, "SENTINEL PRO // FORENSIC REPORT")
    c.setFont("Helvetica", 10)
    c.setFillColor(colors.HexColor("#94a3b8"))
    c.drawString(w-200, h-40, f"CASE ID: SEN-{case_id}")
    c.drawString(w-200, h-55, f"CLASSIFICATION: RESTRICTED")
    c.drawString(w-200, h-70, f"TIMESTAMP: {results['timestamp']}")
    
    # Verdict Box
    y = h - 160
    c.setStrokeColor(v_color)
    c.setLineWidth(2)
    c.setFillColor(colors.HexColor("#f8fafc"))
    c.roundRect(40, y-50, w-80, 60, 8, fill=1, stroke=1)
    c.setFillColor(v_color)
    c.setFont("Helvetica-Bold", 26)
    c.drawCentredString(w/2, y-25, f"CONCLUSION: {verdict}")
    
    # Metadata
    y -= 90
    c.setFont("Helvetica-Bold", 12)
    c.setFillColor(colors.black)
    c.drawString(40, y, "EVIDENCE INTEGRITY METADATA")
    c.setStrokeColor(colors.HexColor("#cbd5e1"))
    c.setLineWidth(1)
    c.line(40, y-8, w-40, y-8)
    y -= 30
    c.setFont("Helvetica", 10)
    c.drawString(40, y, f"Source Filename: {results['filename']}")
    c.drawString(300, y, f"Total Frames Analyzed: {config.SEQ_LENGTH}")
    
    # Visual Evidence
    y -= 50
    c.setFont("Helvetica-Bold", 12)
    c.drawString(40, y, "VISUAL PARSING & DETECTION")
    c.line(40, y-8, w-40, y-8)
    y -= 20
    
    if os.path.exists("temp_thumb.jpg"):
        c.drawImage("temp_thumb.jpg", 40, y - 200, width=200, height=200, preserveAspectRatio=True)
    
    # Metrics
    tx = 270
    ty = y - 40
    c.setFont("Helvetica-Bold", 11)
    c.drawString(tx, ty, "Biometric & Frequency Breakdown:")
    ty -= 25
    
    scores = [
        ("Spatial Artifacts (Xception Net)", results['s']),
        ("High-Freq Noise (SRM Filter)", results['f']),
        ("Temporal Flow (BiLSTM)", results['t']),
        ("Overall Confidence Score", results['final'])
    ]
    for idx, (name, score) in enumerate(scores):
        sc = colors.HexColor("#ef4444") if score > 0.6 else (colors.HexColor("#eab308") if score > 0.35 else colors.HexColor("#10b981"))
        if idx == 3: ty -= 15 # Gap for final score
        
        c.setFont("Helvetica-Bold" if idx == 3 else "Helvetica", 10)
        c.setFillColor(colors.HexColor("#334155"))
        c.drawString(tx, ty, name)
        c.setFillColor(colors.HexColor("#e2e8f0"))
        c.rect(tx+180, ty-1, 100, 10, fill=1, stroke=0)
        c.setFillColor(sc)
        c.rect(tx+180, ty-1, 100*score, 10, fill=1, stroke=0)
        c.setFillColor(colors.black)
        c.drawString(tx+290, ty, f"{score*100:.1f}%")
        ty -= 30

    c.save()
    buffer.seek(0)
    return buffer

# ==========================================
#        UI LAYOUT
# ==========================================

with st.sidebar:
    st.markdown("<h1 style='color: white; font-size: 32px;'>SENTINEL</h1>", unsafe_allow_html=True)
    st.markdown("<p style='color: #3b82f6; font-weight: 800; margin-top:-15px; letter-spacing: 2px;'>PRO // FORENSICS</p>", unsafe_allow_html=True)
    st.markdown("<hr style='border-color: rgba(255,255,255,0.1);'>", unsafe_allow_html=True)
    
    nav = st.radio("OPERATIONAL MODE", ["Dashboard", "Investigation History", "Settings"], label_visibility="collapsed")
    st.markdown("<hr style='border-color: rgba(255,255,255,0.1);'>", unsafe_allow_html=True)
    
    if st.button("Purge System Logs", use_container_width=True):
        st.session_state.logs = []
        st.rerun()

if nav == "Dashboard":
    col1, col_gap, col2 = st.columns([1.2, 0.05, 1.0])
    
    with col1:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.markdown("<div class='section-title'>MEDIA INGESTION</div>", unsafe_allow_html=True)
        
        uploaded_file = st.file_uploader("Drop target file for analysis", type=["mp4", "mov", "avi"], label_visibility="collapsed")
        
        if uploaded_file is not None:
            if st.session_state.video_path is None or not os.path.exists(st.session_state.video_path):
                tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') 
                tfile.write(uploaded_file.read())
                tfile.close()
                st.session_state.video_path = tfile.name
                st.session_state.results = None
            
            st.video(st.session_state.video_path)
            
            action_col1, action_col2 = st.columns([3, 1])
            with action_col1:
                if st.button("INITIATE DIAGNOSTICS SEQUENCE", use_container_width=True, type="primary"):
                    st.session_state.results = None
                    st.session_state.logs = []
                    run_analysis(st.session_state.video_path)
                    st.rerun()
        
        st.markdown("</div>", unsafe_allow_html=True)
        
        if st.session_state.results:
            st.markdown("<div class='card'>", unsafe_allow_html=True)
            st.markdown("<div class='section-title'>FRAME-BY-FRAME ANOMALY DETECTION</div>", unsafe_allow_html=True)
            scores = st.session_state.results['frame_scores']
            
            df = pd.DataFrame({
                "Frame": range(1, len(scores) + 1),
                "Manipulation Probability": [s * 100 for s in scores]
            })
            
            fig = px.area(df, x="Frame", y="Manipulation Probability", 
                          color_discrete_sequence=['#ef4444' if np.mean(scores) > 0.5 else '#3b82f6'])
            fig.update_layout(
                plot_bgcolor="rgba(0,0,0,0)",
                paper_bgcolor="rgba(0,0,0,0)",
                margin=dict(l=0, r=0, t=10, b=0),
                height=250,
                xaxis_title="Frame Index",
                yaxis_title="Probability (%)",
                font=dict(color="#cbd5e1"),
                yaxis=dict(range=[0, 100], gridcolor="rgba(255,255,255,0.1)"),
                xaxis=dict(gridcolor="rgba(255,255,255,0.1)"),
            )
            fig.add_hline(y=60, line_dash="dash", line_color="#ef4444", annotation_text="Deepfake Threshold", annotation_position="top left")
            
            st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})
            st.markdown("</div>", unsafe_allow_html=True)

    with col2:
        st.markdown("<div class='card' style='height: auto;'>", unsafe_allow_html=True)
        st.markdown("<div class='section-title'>DECISION INTELLIGENCE MODULE</div>", unsafe_allow_html=True)
        
        res = st.session_state.results
        if res:
            final = res['final']
            if final > 0.6:
                color, verdict, display_score = "#ef4444", "MANIPULATED", final*100
            elif final > 0.35:
                color, verdict, display_score = "#eab308", "SUSPICIOUS", final*100
            else:
                color, verdict, display_score = "#10b981", "AUTHENTIC", (1-final)*100
                
            st.markdown(f"<div class='metric-value' style='color: {color};'>{display_score:.1f}%</div>", unsafe_allow_html=True)
            st.markdown(f"<div class='verdict-badge' style='background-color: {color}20; color: {color}; border: 1px solid {color}50;'>{verdict}</div>", unsafe_allow_html=True)
            
            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown("<div class='section-title'>BIOMETRIC ATTRIBUTES</div>", unsafe_allow_html=True)
            
            def metric_bar(label, value, is_safe):
                bar_color = "#ef4444" if not is_safe else "#10b981"
                st.markdown(f"**{label}** - {value*100:.1f}%")
                st.progress(value)
                
            metric_bar("Spatial Anomalies", res['s'], res['s'] < 0.6)
            st.markdown("<div style='height: 10px;'></div>", unsafe_allow_html=True)
            metric_bar("Frequency Irregularities", res['f'], res['f'] < 0.6)
            st.markdown("<div style='height: 10px;'></div>", unsafe_allow_html=True)
            metric_bar("Temporal Inconsistency", res['t'], res['t'] < 0.6)
            
            st.markdown("<hr style='border-color: rgba(255,255,255,0.1); margin: 25px 0;'>", unsafe_allow_html=True)
            
            pdf_buffer = generate_pdf_buffer(res)
            st.download_button(
                label="📥 EXPORT OFFICIAL PDF REPORT",
                data=pdf_buffer,
                file_name=f"Forensic_Report_{res['filename']}.pdf",
                mime="application/pdf",
                use_container_width=True,
                type="secondary"
            )
            
        else:
            st.markdown("<div class='metric-value' style='color: #334155;'>--.-%</div>", unsafe_allow_html=True)
            st.markdown(f"<div class='verdict-badge' style='background-color: rgba(30,41,59,0.8); color: #64748b; border: 1px solid #334155;'>SYSTEM IDLE</div>", unsafe_allow_html=True)
            
            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown("<div class='section-title'>AWAITING DATA</div>", unsafe_allow_html=True)
            st.markdown("<p style='color: #64748b; font-size: 14px;'>Upload a media file and initiate the diagnostic sequence to view the biometric breakdown.</p>", unsafe_allow_html=True)
        
        st.markdown("</div>", unsafe_allow_html=True)
        
        st.markdown("<div class='section-title'>SYSTEM KERNEL LOG</div>", unsafe_allow_html=True)
        log_html = "<br>".join(st.session_state.logs)
        st.markdown(f"<div class='log-box'>{log_html if log_html else '<span style=\"color:#64748b;\">SYSTEM STANDBY...</span>'}</div>", unsafe_allow_html=True)

elif nav == "Investigation History":
    st.markdown("<h2>Investigation Logs</h2>", unsafe_allow_html=True)
    if not st.session_state.history:
        st.info("No prior investigations found in the current session kernel.")
    else:
        for idx, item in enumerate(st.session_state.history):
            color = "🔴" if item['final'] > 0.6 else ("🟡" if item['final'] > 0.35 else "🟢")
            with st.expander(f"{color} {item['filename']} | {item['timestamp']} | Overall Confidence: {item['final']*100:.1f}%"):
                c1, c2, c3 = st.columns(3)
                c1.metric("Spatial Probability", f"{item['s']*100:.1f}%")
                c2.metric("Frequency Probability", f"{item['f']*100:.1f}%")
                c3.metric("Temporal Anomaly", f"{item['t']*100:.1f}%")

elif nav == "Settings":
    st.markdown("<h2>System Configuration Profile</h2>", unsafe_allow_html=True)
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    
    config_data = {
        "Neural Engine Host": "GPU (CUDA Accelerated)" if torch.cuda.is_available() else "CPU (Standard Node)",
        "Temporal Sequence Length": f"{config.SEQ_LENGTH} frames",
        "Spatial Matrix Resolution": f"{config.IMG_SIZE}x{config.IMG_SIZE}",
        "Classification Thresholds": "Authentic (<35%) | Suspicious (35-60%) | Deepfake (>60%)"
    }
    
    for key, val in config_data.items():
        st.markdown(f"<p><strong style='color:#aeb2b8;'>{key}:</strong> {val}</p>", unsafe_allow_html=True)
        st.markdown("<hr style='border-color: rgba(255,255,255,0.05); margin: 10px 0;'>", unsafe_allow_html=True)
        
    st.markdown("</div>", unsafe_allow_html=True)