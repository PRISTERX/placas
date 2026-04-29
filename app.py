import streamlit as st
import cv2
import numpy as np
from PIL import Image
from ultralytics import YOLO
import easyocr
import os
import time
from datetime import datetime

# ─── CONFIG ───────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Detector de Placas",
    page_icon="🚗",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# ─── CSS ──────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=Syne:wght@400;700;800&display=swap');
:root {
    --bg: #0a0a0f; --surface: #12121a; --border: #1e1e2e;
    --accent: #00ff88; --accent2: #ff6b35; --text: #e8e8f0; --muted: #6b6b80;
}
html, body, [class*="css"] { background-color: var(--bg) !important; color: var(--text) !important; font-family: 'Syne', sans-serif !important; }
.stApp { background: var(--bg) !important; }
.hero { text-align: center; padding: 2rem 1rem 1.5rem; border-bottom: 1px solid var(--border); margin-bottom: 1.5rem; }
.hero h1 { font-family: 'Syne', sans-serif; font-weight: 800; font-size: clamp(1.6rem, 6vw, 2.5rem); color: var(--accent); letter-spacing: -1px; margin: 0; }
.hero p { color: var(--muted); font-family: 'Space Mono', monospace; font-size: 0.7rem; margin-top: 0.5rem; letter-spacing: 2px; text-transform: uppercase; }
.plate-card { background: var(--surface); border: 1px solid var(--accent); border-radius: 12px; padding: 1.5rem 2rem; text-align: center; margin: 1rem 0; box-shadow: 0 0 30px rgba(0,255,136,0.08); }
.plate-text { font-family: 'Space Mono', monospace; font-size: clamp(2rem, 8vw, 3.5rem); font-weight: 700; color: var(--accent); letter-spacing: 8px; text-transform: uppercase; }
.plate-label { font-size: 0.7rem; color: var(--muted); letter-spacing: 3px; text-transform: uppercase; margin-bottom: 0.5rem; }
.conf-badge { display: inline-block; background: rgba(0,255,136,0.1); border: 1px solid rgba(0,255,136,0.3); border-radius: 20px; padding: 0.2rem 0.8rem; font-family: 'Space Mono', monospace; font-size: 0.7rem; color: var(--accent); margin-top: 0.5rem; }
.no-plate { background: var(--surface); border: 1px solid var(--accent2); border-radius: 12px; padding: 1.5rem; text-align: center; color: var(--accent2); font-family: 'Space Mono', monospace; font-size: 0.85rem; margin: 1rem 0; }
.auto-badge { background: rgba(0,255,136,0.1); border: 1px solid #00ff88; border-radius: 8px; padding: 0.5rem; text-align: center; color: #00ff88; font-size: 0.8rem; letter-spacing: 1px; margin-bottom: 1rem; }
.auto-off { background: rgba(255,107,53,0.1); border: 1px solid var(--accent2); border-radius: 8px; padding: 0.5rem; text-align: center; color: var(--accent2); font-size: 0.8rem; letter-spacing: 1px; margin-bottom: 1rem; }
.history-item { background: var(--surface); border: 1px solid var(--border); border-radius: 8px; padding: 0.8rem 1rem; margin-bottom: 0.5rem; display: flex; justify-content: space-between; align-items: center; font-family: 'Space Mono', monospace; font-size: 0.8rem; }
.history-plate { color: var(--accent); font-weight: 700; letter-spacing: 3px; }
.history-time { color: var(--muted); font-size: 0.7rem; }
.stButton > button { background: var(--accent) !important; color: #000 !important; border: none !important; border-radius: 8px !important; font-family: 'Space Mono', monospace !important; font-weight: 700 !important; font-size: 0.85rem !important; width: 100% !important; }
[data-testid="stMetric"] { background: var(--surface) !important; border: 1px solid var(--border) !important; border-radius: 10px !important; padding: 1rem !important; }
[data-testid="stMetricLabel"] { color: var(--muted) !important; font-size: 0.7rem !important; }
[data-testid="stMetricValue"] { color: var(--accent) !important; font-family: 'Space Mono', monospace !important; }
hr { border-color: var(--border) !important; }
</style>
""", unsafe_allow_html=True)

# ─── HEADER ───────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero">
    <h1>🚗 DETECTOR DE PLACAS</h1>
    <p>YOLOv8n · EasyOCR · Parqueadero Inteligente</p>
</div>
""", unsafe_allow_html=True)

# ─── CARGAR MODELOS ───────────────────────────────────────────────────────────
@st.cache_resource
def cargar_modelos():
    modelo = YOLO("best.pt")
    ocr = easyocr.Reader(['en'], gpu=False)
    return modelo, ocr

with st.spinner("Cargando modelos..."):
    modelo, ocr_reader = cargar_modelos()

# ─── ESTADO ───────────────────────────────────────────────────────────────────
if "historial" not in st.session_state:
    st.session_state.historial = []
if "total_detectadas" not in st.session_state:
    st.session_state.total_detectadas = 0
if "auto_activo" not in st.session_state:
    st.session_state.auto_activo = False
if "ultima_placa" not in st.session_state:
    st.session_state.ultima_placa = ""
if "ultima_conf" not in st.session_state:
    st.session_state.ultima_conf = 0.0

# ─── FUNCIÓN DETECCIÓN ────────────────────────────────────────────────────────
def detectar_placa(imagen_pil):
    img_np = np.array(imagen_pil.convert("RGB"))
    img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
    resultados = modelo(img_bgr, conf=0.25, verbose=False)
    placas_detectadas = []
    img_anotada = img_bgr.copy()

    for r in resultados:
        for box in r.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = float(box.conf[0])
            recorte = img_bgr[y1:y2, x1:x2]
            texto_placa = ""
            if recorte.size > 0:
                recorte_gray = cv2.cvtColor(recorte, cv2.COLOR_BGR2GRAY)
                recorte_up = cv2.resize(recorte_gray, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
                _, recorte_bin = cv2.threshold(recorte_up, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                ocr_result = ocr_reader.readtext(recorte_bin, detail=0, allowlist='ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789')
                texto_placa = "".join(ocr_result).strip().upper()
            cv2.rectangle(img_anotada, (x1, y1), (x2, y2), (0, 255, 136), 2)
            label = f"{texto_placa} {conf:.0%}" if texto_placa else f"Placa {conf:.0%}"
            cv2.putText(img_anotada, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 136), 2)
            placas_detectadas.append({"texto": texto_placa, "confianza": conf, "bbox": (x1, y1, x2, y2)})

    img_resultado = Image.fromarray(cv2.cvtColor(img_anotada, cv2.COLOR_BGR2RGB))
    return img_resultado, placas_detectadas

# ─── MODO DE ENTRADA ──────────────────────────────────────────────────────────
modo = st.radio("Modo:", ["📁 Subir imagen", "📷 Cámara", "🤖 Auto-detección"], horizontal=True)

imagen_entrada = None

# ── Subir imagen ──
if modo == "📁 Subir imagen":
    archivo = st.file_uploader("Selecciona una imagen", type=["jpg", "jpeg", "png", "webp"], label_visibility="collapsed")
    if archivo:
        imagen_entrada = Image.open(archivo)

# ── Cámara manual ──
elif modo == "📷 Cámara":
    foto = st.camera_input("Toma una foto del vehículo")
    if foto:
        imagen_entrada = Image.open(foto)

# ── Auto-detección ──
elif modo == "🤖 Auto-detección":
    st.markdown("""
    <div style='background:#12121a;border:1px solid #1e1e2e;border-radius:10px;
    padding:0.8rem;margin-bottom:1rem;text-align:center;'>
        <span style='color:#6b6b80;font-size:0.7rem;letter-spacing:2px;'>
        CAPTURA AUTOMÁTICA · APUNTA AL VEHÍCULO
        </span>
    </div>
    """, unsafe_allow_html=True)

    intervalo = st.slider("⏱ Intervalo (segundos)", 2, 10, 3)

    col1, col2 = st.columns(2)
    with col1:
        if st.button("▶️ Iniciar"):
            st.session_state.auto_activo = True
    with col2:
        if st.button("⏹️ Detener"):
            st.session_state.auto_activo = False

    # Estado visual
    if st.session_state.auto_activo:
        st.markdown('<div class="auto-badge">🟢 AUTO-DETECCIÓN ACTIVA</div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="auto-off">⏸ AUTO-DETECCIÓN DETENIDA</div>', unsafe_allow_html=True)

    # Cámara
    foto_auto = st.camera_input("📷 Cámara en vivo", key="camara_auto", label_visibility="collapsed")

    if foto_auto:
        imagen_entrada = Image.open(foto_auto)

    # Mostrar última placa detectada
    if st.session_state.ultima_placa:
        st.markdown(f"""
        <div class="plate-card">
            <div class="plate-label">Última placa detectada</div>
            <div class="plate-text">{st.session_state.ultima_placa}</div>
            <div class="conf-badge">Confianza: {st.session_state.ultima_conf:.1%}</div>
        </div>
        """, unsafe_allow_html=True)

    # Auto-rerun
    if st.session_state.auto_activo:
        time.sleep(intervalo)
        st.rerun()

# ─── DETECCIÓN ────────────────────────────────────────────────────────────────
if imagen_entrada is not None:
    col1, col2 = st.columns(2)
    with col1:
        st.image(imagen_entrada, caption="Original", use_column_width=True)

    with st.spinner("Analizando..."):
        img_resultado, placas = detectar_placa(imagen_entrada)

    with col2:
        st.image(img_resultado, caption="Detección", use_column_width=True)

    st.markdown("---")

    if placas:
        for p in placas:
            st.session_state.total_detectadas += 1
            if p["texto"]:
                st.session_state.ultima_placa = p["texto"]
                st.session_state.ultima_conf = p["confianza"]
                st.session_state.historial.insert(0, {
                    "placa": p["texto"],
                    "confianza": p["confianza"],
                    "hora": datetime.now().strftime("%H:%M:%S")
                })
                st.session_state.historial = st.session_state.historial[:20]
                st.markdown(f"""
                <div class="plate-card">
                    <div class="plate-label">Placa detectada</div>
                    <div class="plate-text">{p["texto"]}</div>
                    <div class="conf-badge">Confianza: {p["confianza"]:.1%}</div>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="plate-card">
                    <div class="plate-label">Placa detectada — OCR no pudo leer</div>
                    <div class="plate-text">— — —</div>
                    <div class="conf-badge">Confianza detección: {p["confianza"]:.1%}</div>
                </div>
                """, unsafe_allow_html=True)
    else:
        if modo != "🤖 Auto-detección":
            st.markdown("""
            <div class="no-plate">
                ⚠️ No se detectó ninguna placa.<br>
                Intenta con una imagen más clara o más cercana.
            </div>
            """, unsafe_allow_html=True)

# ─── MÉTRICAS ─────────────────────────────────────────────────────────────────
st.markdown("---")
c1, c2, c3 = st.columns(3)
c1.metric("Total detectadas", st.session_state.total_detectadas)
c2.metric("En historial", len(st.session_state.historial))
c3.metric("Modelo", "YOLOv8n")

# ─── HISTORIAL ────────────────────────────────────────────────────────────────
if st.session_state.historial:
    st.markdown("### 📋 Historial de placas")
    for item in st.session_state.historial:
        st.markdown(f"""
        <div class="history-item">
            <span class="history-plate">{item['placa']}</span>
            <span class="history-time">{item['hora']} · {item['confianza']:.0%}</span>
        </div>
        """, unsafe_allow_html=True)

    if st.button("🗑️ Limpiar historial"):
        st.session_state.historial = []
        st.session_state.total_detectadas = 0
        st.session_state.ultima_placa = ""
        st.rerun()
