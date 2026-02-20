import streamlit as st
import os
import cv2
import torch
import torch.nn as nn
import numpy as np
import datetime
import time
import tempfile
import io
from PIL import Image, ImageDraw
from torchvision import transforms
from facenet_pytorch import MTCNN
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from reportlab.lib import colors

# Import your existing configurations and models
import config
from models import SpatialXception, SRMXception, DeepfakeLSTM

# ==========================================
#        PAGE CONFIG & STYLING
# ==========================================
st.set_page_config(page_title="Sentinel Pro // Forensic Engine", page_icon="🛡️", layout="wide")

# Inject Custom CSS to mimic the PyQt5 Dark Theme
st.markdown("""
    <style>
    .stApp { background-color: #0f0f12; color: #e4e4e7; }
    .css-1d391kg { background-color: #18181b; } /* Sidebar */
    h1, h2, h3 { color: #ffffff; letter-spacing: 1px; }
    .card { background-color: #18181b; border: 1px solid #27272a; border-radius: 12px; padding: 20px; margin-bottom: 20px;}
    .metric-value { font-size: 48px; font-weight: 900; }
    .section-title { color: #a1a1aa; font-size: 12px; font-weight: bold; text-transform: uppercase; letter-spacing: 1px; margin-bottom: 10px; }
    .log-box { background-color: #09090b; border: 1px solid #27272a; border-radius: 8px; padding: 10px; font-family: monospace; font-size: 12px; height: 150px; overflow-y: auto; color: #e4e4e7;}
    .log-success { color: #22c55e; }
    .log-error { color: #ef4444; }
    .log-info { color: #6366f1; }
    div[data-testid="stMetricValue"] { font-size: 24px; font-weight: bold; }
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
    st.session_state.logs.append(f"<span style='color:#71717a;'>[{t}]</span> <span class='{color_class}'>{msg}</span>")

# ==========================================
#        MODEL LOADING (CACHED)
# ==========================================
@st.cache_resource(show_spinner="Booting Neural Networks...")
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
        models['mtcnn'] = MTCNN(keep_all=False, select_largest=True, device=device, margin=0)
        return models, device
    except Exception as e:
        st.error(f"Failed to load models: {e}")
        return None, device

models, DEVICE = load_models()

# ==========================================
#        INFERENCE ENGINE
# ==========================================
def run_analysis(video_path):
    add_log(">>> INITIATING FORENSIC SCAN...", "info")
    
    trans = transforms.Compose([
        transforms.Resize((config.IMG_SIZE, config.IMG_SIZE)), 
        transforms.ToTensor(), 
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    trans_srm = transforms.Compose([
        transforms.Resize((config.IMG_SIZE, config.IMG_SIZE)), 
        transforms.ToTensor()
    ])

    cap = cv2.VideoCapture(video_path)
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

    while len(frames) < config.SEQ_LENGTH: 
        frames.append(frames[-1] if frames else Image.new('RGB', (config.IMG_SIZE, config.IMG_SIZE)))

    batch_s, batch_f = [], []
    frame_scores = []
    
    add_log(">>> ANALYZING VISUAL EVIDENCE...", "info")
    progress_bar = st.progress(0)
    status_text = st.empty()

    thumb_saved = False
    
    for i, f in enumerate(frames):
        status_text.text(f"Scanning Frame [{i+1}/{config.SEQ_LENGTH}]")
        boxes, _ = models['mtcnn'].detect(f)
        
        if boxes is not None: 
            face = f.crop(boxes[0])
            draw = ImageDraw.Draw(f)
            
            face_t = face.resize((config.IMG_SIZE, config.IMG_SIZE), Image.BILINEAR)
            inp_s = trans(face_t).unsqueeze(0).to(DEVICE)
            inp_f = trans_srm(face_t).unsqueeze(0).to(DEVICE)
            
            with torch.no_grad():
                _, l_s = models['spatial'](inp_s)
                _, l_f = models['srm'](inp_f)
                prob_s = torch.softmax(l_s, dim=1)[:, 1].item()
                prob_f = torch.sigmoid(l_f).item()
                score_frame = (prob_s + prob_f) / 2
            
            color = "#ef4444" if score_frame > 0.5 else "#22c55e"
            draw.rectangle(boxes[0].tolist(), outline=color, width=5)
        else: 
            face = f.resize((config.IMG_SIZE, config.IMG_SIZE))
            score_frame = 0.5
        
        # Save first processed frame as thumbnail for PDF
        if not thumb_saved:
            f.save("temp_thumb.jpg")
            thumb_saved = True
            
        frame_scores.append(score_frame)
        face_final = face.resize((config.IMG_SIZE, config.IMG_SIZE), Image.BILINEAR)
        batch_s.append(trans(face_final))
        batch_f.append(trans_srm(face_final))
        
        progress_bar.progress((i + 1) / config.SEQ_LENGTH)

    add_log(">>> COMPUTING TEMPORAL CONSISTENCY...", "info")
    inp_s = torch.stack(batch_s).to(DEVICE)
    inp_f = torch.stack(batch_f).to(DEVICE)

    with torch.no_grad():
        feat_s, log_s = models['spatial'](inp_s)
        val_s = torch.softmax(log_s, dim=1)[:, 1].mean().item()
        feat_f, log_f = models['srm'](inp_f)
        val_f = torch.sigmoid(log_f).mean().item()
        
        combined = torch.cat((feat_s, feat_f), dim=1).unsqueeze(0)
        val_t = torch.sigmoid(models['lstm'](combined)).item()

    final = (0.4 * val_s) + (0.4 * val_f) + (0.2 * val_t)
    add_log(">>> ANALYSIS COMPLETE.", "success")
    
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
    verdict = "MANIPULATED" if final > 0.6 else ("SUSPICIOUS" if final > 0.35 else "AUTHENTIC")
    v_color = colors.HexColor("#ef4444") if final > 0.6 else (colors.HexColor("#eab308") if final > 0.35 else colors.HexColor("#22c55e"))

    # Header
    c.setFillColor(colors.HexColor("#0f0f12"))
    c.rect(0, h-80, w, 80, fill=1, stroke=0)
    c.setFillColor(colors.white)
    c.setFont("Helvetica-Bold", 22)
    c.drawString(40, h-50, "SENTINEL PRO — FORENSIC ANALYSIS REPORT")
    c.setFont("Helvetica", 9)
    c.setFillColor(colors.HexColor("#a0a0a0"))
    c.drawString(w-180, h-40, f"Case ID: SEN-{case_id}")
    c.drawString(w-180, h-55, f"Classification: CONFIDENTIAL")
    
    # Verdict Box
    y = h - 140
    c.setStrokeColor(v_color)
    c.setLineWidth(2.5)
    c.setFillColor(colors.HexColor("#f9f9f9"))
    c.roundRect(50, y-45, w-100, 55, 8, fill=1, stroke=1)
    c.setFillColor(v_color)
    c.setFont("Helvetica-Bold", 26)
    c.drawCentredString(w/2, y-25, f"CONCLUSION: {verdict}")
    
    # Metadata
    y -= 80
    c.setFont("Helvetica-Bold", 12)
    c.setFillColor(colors.black)
    c.drawString(50, y, "EVIDENCE INFORMATION")
    c.line(50, y-5, w-50, y-5)
    y -= 25
    c.setFont("Helvetica", 10)
    c.drawString(50, y, f"Filename: {results['filename']}")
    c.drawString(300, y, f"Date Analysed: {results['timestamp']}")
    
    # Visual Evidence
    y -= 40
    c.setFont("Helvetica-Bold", 12)
    c.drawString(50, y, "VISUAL EVIDENCE & BREAKDOWN")
    c.line(50, y-5, w-50, y-5)
    y -= 15
    
    if os.path.exists("temp_thumb.jpg"):
        c.drawImage("temp_thumb.jpg", 50, y - 180, width=180, height=180, preserveAspectRatio=True)
    
    # Metrics
    tx = 260
    ty = y - 30
    c.setFont("Helvetica-Bold", 11)
    c.drawString(tx, ty, "Technical Biometric Breakdown:")
    ty -= 20
    
    scores = [
        ("Spatial Artifacts (Xception)", results['s']),
        ("Frequency Noise (SRM Filter)", results['f']),
        ("Temporal Consistency (BiLSTM)", results['t']),
    ]
    for name, score in scores:
        sc = colors.HexColor("#ef4444") if score > 0.6 else colors.HexColor("#22c55e")
        c.setFont("Helvetica", 10)
        c.setFillColor(colors.HexColor("#555555"))
        c.drawString(tx, ty, name)
        c.setFillColor(colors.HexColor("#e0e0e0"))
        c.rect(tx+170, ty, 100, 10, fill=1, stroke=0)
        c.setFillColor(sc)
        c.rect(tx+170, ty, 100*score, 10, fill=1, stroke=0)
        c.setFillColor(colors.black)
        c.drawString(tx+280, ty, f"{score*100:.1f}%")
        ty -= 25

    c.save()
    buffer.seek(0)
    return buffer

# ==========================================
#        UI LAYOUT
# ==========================================

with st.sidebar:
    st.markdown("<h1 style='color: white;'>SENTINEL</h1>", unsafe_allow_html=True)
    st.markdown("<p style='color: #71717a; font-weight: bold; margin-top:-15px;'>FORENSIC ENGINE v3.0</p>", unsafe_allow_html=True)
    st.markdown("---")
    
    nav = st.radio("Navigation", ["Dashboard", "Case History", "Settings"], label_visibility="collapsed")
    st.markdown("---")
    
    if st.button("🗑️ Clear Kernel Logs"):
        st.session_state.logs = []
        st.rerun()

if nav == "Dashboard":
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        uploaded_file = st.file_uploader("Upload Media (MP4, MOV, AVI)", type=["mp4", "mov", "avi"])
        
        if uploaded_file is not None:
            # Write to temp file
            tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') 
            tfile.write(uploaded_file.read())
            st.session_state.video_path = tfile.name
            
            st.video(st.session_state.video_path)
            
            if st.button("RUN DIAGNOSTICS", use_container_width=True, type="primary"):
                st.session_state.results = None # Reset previous
                st.session_state.logs = []
                run_analysis(st.session_state.video_path)
                st.rerun()
                
        st.markdown("</div>", unsafe_allow_html=True)
        
        # Kernel Logs
        st.markdown("<div class='section-title'>SYSTEM KERNEL LOG</div>", unsafe_allow_html=True)
        log_html = "<br>".join(st.session_state.logs)
        st.markdown(f"<div class='log-box'>{log_html if log_html else 'AWAITING INPUT...'}</div>", unsafe_allow_html=True)

    with col2:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.markdown("<div class='section-title'>DECISION INTELLIGENCE</div>", unsafe_allow_html=True)
        
        res = st.session_state.results
        if res:
            final = res['final']
            if final > 0.6:
                color, verdict, display_score = "#ef4444", "MANIPULATED", final*100
            elif final > 0.35:
                color, verdict, display_score = "#eab308", "SUSPICIOUS", final*100
            else:
                color, verdict, display_score = "#22c55e", "AUTHENTIC", (1-final)*100
                
            st.markdown(f"<div class='metric-value' style='color: {color};'>{display_score:.1f}%</div>", unsafe_allow_html=True)
            st.markdown(f"<div style='background-color: {color}20; color: {color}; padding: 10px; border-radius: 8px; text-align: center; font-weight: bold; margin-bottom: 20px;'>{verdict}</div>", unsafe_allow_html=True)
            
            st.markdown("<div class='section-title'>MANIPULATION PROBABILITY</div>", unsafe_allow_html=True)
            
            # Metrics
            st.markdown("Spatial Artifacts")
            st.progress(res['s'])
            st.markdown("Frequency Noise")
            st.progress(res['f'])
            st.markdown("Temporal Consistency")
            st.progress(res['t'])
            
            st.markdown("---")
            pdf_buffer = generate_pdf_buffer(res)
            st.download_button(
                label="📥 EXPORT PDF REPORT",
                data=pdf_buffer,
                file_name=f"Report_{res['filename']}.pdf",
                mime="application/pdf",
                use_container_width=True
            )
            
        else:
            st.markdown("<div class='metric-value' style='color: #3f3f46;'>--%</div>", unsafe_allow_html=True)
            st.markdown(f"<div style='background-color: #27272a; color: #52525b; padding: 10px; border-radius: 8px; text-align: center; font-weight: bold; margin-bottom: 20px;'>AWAITING ANALYSIS</div>", unsafe_allow_html=True)
        
        st.markdown("</div>", unsafe_allow_html=True)

elif nav == "Case History":
    st.markdown("<h3 style='color: white;'>Case History</h3>", unsafe_allow_html=True)
    if not st.session_state.history:
        st.info("No analysis history found for this session.")
    else:
        for idx, item in enumerate(st.session_state.history):
            with st.expander(f"📄 {item['filename']} | {item['timestamp']} | Score: {item['final']*100:.1f}%"):
                col_a, col_b, col_c = st.columns(3)
                col_a.metric("Spatial", f"{item['s']*100:.1f}%")
                col_b.metric("Frequency", f"{item['f']*100:.1f}%")
                col_c.metric("Temporal", f"{item['t']*100:.1f}%")

elif nav == "Settings":
    st.markdown("<h3 style='color: white;'>System Configuration</h3>", unsafe_allow_html=True)
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.markdown("**Device:** " + ("GPU (CUDA)" if torch.cuda.is_available() else "CPU"))
    st.markdown(f"**Sequence Length:** {config.SEQ_LENGTH} frames")
    st.markdown(f"**Image Resolution:** {config.IMG_SIZE}x{config.IMG_SIZE}")
    st.markdown("**Thresholds:** Authentic (<35%), Suspicious (35-60%), Manipulated (>60%)")
    st.markdown("</div>", unsafe_allow_html=True)