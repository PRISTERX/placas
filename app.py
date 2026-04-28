import streamlit as st
import cv2
import numpy as np
from PIL import Image
from ultralytics import YOLO
import easyocr
import tempfile
import os
import base64
import io
from datetime import datetime

# ─── CONFIG ───────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Detector de Placas",
    page_icon="🚗",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# ─── CSS PERSONALIZADO ────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=Syne:wght@400;700;800&display=swap');

:root {
    --bg: #0a0a0f;
    --surface: #12121a;
    --border: #1e1e2e;
    --accent: #00ff88;
    --accent2: #ff6b35;
    --text: #e8e8f0;
    --muted: #6b6b80;
}

html, body, [class*="css"] {
    background-color: var(--bg) !important;
    color: var(--text) !important;
    font-family: 'Syne', sans-serif !important;
}

.stApp {
    background: var(--bg) !important;
}

/* Header */
.hero {
    text-align: center;
    padding: 2.5rem 1rem 1.5rem;
    border-bottom: 1px solid var(--border);
    margin-bottom: 2rem;
}
.hero h1 {
    font-family: 'Syne', sans-serif;
    font-weight: 800;
    font-size: clamp(1.8rem, 6vw, 3rem);
    color: var(--accent);
    letter-spacing: -1px;
    margin: 0;
    line-height: 1.1;
}
.hero p {
    color: var(--muted);
    font-family: 'Space Mono', monospace;
    font-size: 0.75rem;
    margin-top: 0.5rem;
    letter-spacing: 2px;
    text-transform: uppercase;
}

/* Tarjeta resultado placa */
.plate-card {
    background: var(--surface);
    border: 1px solid var(--accent);
    border-radius: 12px;
    padding: 1.5rem 2rem;
    text-align: center;
    margin: 1.5rem 0;
    box-shadow: 0 0 30px rgba(0,255,136,0.08);
}
.plate-text {
    font-family: 'Space Mono', monospace;
    font-size: clamp(2rem, 8vw, 3.5rem);
    font-weight: 700;
    color: var(--accent);
    letter-spacing: 8px;
    text-transform: uppercase;
}
.plate-label {
    font-size: 0.7rem;
    color: var(--muted);
    letter-spacing: 3px;
    text-transform: uppercase;
    margin-bottom: 0.5rem;
}
.conf-badge {
    display: inline-block;
    background: rgba(0,255,136,0.1);
    border: 1px solid rgba(0,255,136,0.3);
    border-radius: 20px;
    padding: 0.2rem 0.8rem;
    font-family: 'Space Mono', monospace;
    font-size: 0.7rem;
    color: var(--accent);
    margin-top: 0.5rem;
}

/* Tarjeta sin detección */
.no-plate {
    background: var(--surface);
    border: 1px solid var(--accent2);
    border-radius: 12px;
    padding: 1.5rem;
    text-align: center;
    color: var(--accent2);
    font-family: 'Space Mono', monospace;
    font-size: 0.85rem;
    margin: 1rem 0;
}

/* Upload area */
[data-testid="stFileUploader"] {
    background: var(--surface) !important;
    border: 1px dashed var(--border) !important;
    border-radius: 12px !important;
    padding: 1rem !important;
}
[data-testid="stFileUploader"]:hover {
    border-color: var(--accent) !important;
}

/* Botones */
.stButton > button {
    background: var(--accent) !important;
    color: #000 !important;
    border: none !important;
    border-radius: 8px !important;
    font-family: 'Space Mono', monospace !important;
    font-weight: 700 !important;
    font-size: 0.85rem !important;
    letter-spacing: 1px !important;
    padding: 0.6rem 2rem !important;
    width: 100% !important;
    transition: all 0.2s !important;
}
.stButton > button:hover {
    background: #00cc6a !important;
    transform: translateY(-1px) !important;
}

/* Radio / tabs */
[data-testid="stRadio"] label {
    color: var(--text) !important;
    font-family: 'Space Mono', monospace !important;
    font-size: 0.8rem !important;
}

/* Métricas */
[data-testid="stMetric"] {
    background: var(--surface) !important;
    border: 1px solid var(--border) !important;
    border-radius: 10px !important;
    padding: 1rem !important;
}
[data-testid="stMetricLabel"] {
    color: var(--muted) !important;
    font-family: 'Space Mono', monospace !important;
    font-size: 0.7rem !important;
}
[data-testid="stMetricValue"] {
    color: var(--accent) !important;
    font-family: 'Space Mono', monospace !important;
}

/* Historial */
.history-item {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 0.8rem 1rem;
    margin-bottom: 0.5rem;
    display: flex;
    justify-content: space-between;
    align-items: center;
    font-family: 'Space Mono', monospace;
    font-size: 0.8rem;
}
.history-plate { color: var(--accent); font-weight: 700; letter-spacing: 3px; }
.history-time { color: var(--muted); font-size: 0.7rem; }

/* Spinner */
.stSpinner > div { border-color: var(--accent) transparent transparent transparent !important; }

/* Scrollbar */
::-webkit-scrollbar { width: 4px; }
::-webkit-scrollbar-track { background: var(--bg); }
::-webkit-scrollbar-thumb { background: var(--border); border-radius: 2px; }

/* Imagen resultado */
.result-img {
    border-radius: 10px;
    border: 1px solid var(--border);
    overflow: hidden;
}

/* Divider */
hr { border-color: var(--border) !important; }
</style>
""", unsafe_allow_html=True)

# ─── HEADER ──────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero">
    <h1>🚗 DETECTOR DE PLACAS</h1>
    <p>YOLOv8n · EasyOCR · Parqueadero Inteligente</p>
</div>
""", unsafe_allow_html=True)

# ─── CARGAR MODELOS ──────────────────────────────────────────────────────────
@st.cache_resource
def cargar_modelos():
    modelo = YOLO("best.pt")
    ocr = easyocr.Reader(['en'], gpu=False)
    return modelo, ocr

with st.spinner("Cargando modelos..."):
    modelo, ocr_reader = cargar_modelos()

# ─── ESTADO ──────────────────────────────────────────────────────────────────
if "historial" not in st.session_state:
    st.session_state.historial = []
if "total_detectadas" not in st.session_state:
    st.session_state.total_detectadas = 0

# ─── FUNCIÓN DETECCIÓN ───────────────────────────────────────────────────────
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

            # Recorte de la placa para OCR
            recorte = img_bgr[y1:y2, x1:x2]
            texto_placa = ""

            if recorte.size > 0:
                recorte_gray = cv2.cvtColor(recorte, cv2.COLOR_BGR2GRAY)
                recorte_up = cv2.resize(recorte_gray, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
                _, recorte_bin = cv2.threshold(recorte_up, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

                ocr_result = ocr_reader.readtext(recorte_bin, detail=0, allowlist='ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789')
                texto_placa = "".join(ocr_result).strip().upper()

            # Dibujar bounding box
            cv2.rectangle(img_anotada, (x1, y1), (x2, y2), (0, 255, 136), 2)
            label = f"{texto_placa} {conf:.0%}" if texto_placa else f"Placa {conf:.0%}"
            cv2.putText(img_anotada, label, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 136), 2)

            placas_detectadas.append({
                "texto": texto_placa,
                "confianza": conf,
                "bbox": (x1, y1, x2, y2)
            })

    img_resultado = Image.fromarray(cv2.cvtColor(img_anotada, cv2.COLOR_BGR2RGB))
    return img_resultado, placas_detectadas

# ─── MODO DE ENTRADA ─────────────────────────────────────────────────────────
modo = st.radio("Modo de entrada:", ["📁 Subir imagen", "📷 Cámara"], horizontal=True)

imagen_entrada = None

if modo == "📁 Subir imagen":
    archivo = st.file_uploader("Selecciona una imagen del vehículo",
                                type=["jpg", "jpeg", "png", "webp"],
                                label_visibility="collapsed")
    if archivo:
        imagen_entrada = Image.open(archivo)

elif modo == "📷 Cámara":
    foto = st.camera_input("Toma una foto del vehículo")
    if foto:
        imagen_entrada = Image.open(foto)

# ─── DETECCIÓN ───────────────────────────────────────────────────────────────
if imagen_entrada is not None:
    col1, col2 = st.columns(2)
    with col1:
        st.image(imagen_entrada, caption="Imagen original", use_column_width=True)

    with st.spinner("Analizando imagen..."):
        img_resultado, placas = detectar_placa(imagen_entrada)

    with col2:
        st.image(img_resultado, caption="Detección", use_column_width=True)

    st.markdown("---")

    if placas:
        for p in placas:
            st.session_state.total_detectadas += 1

            if p["texto"]:
                # Guardar en historial
                st.session_state.historial.insert(0, {
                    "placa": p["texto"],
                    "confianza": p["confianza"],
                    "hora": datetime.now().strftime("%H:%M:%S")
                })
                # Limitar historial a 20
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
                    <div class="plate-label">Placa detectada — OCR no pudo leer el texto</div>
                    <div class="plate-text">— — —</div>
                    <div class="conf-badge">Confianza detección: {p["confianza"]:.1%}</div>
                </div>
                """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div class="no-plate">
            ⚠️ No se detectó ninguna placa en la imagen.<br>
            Intenta con una imagen más clara o más cercana al vehículo.
        </div>
        """, unsafe_allow_html=True)

# ─── MÉTRICAS ────────────────────────────────────────────────────────────────
st.markdown("---")
c1, c2, c3 = st.columns(3)
c1.metric("Total detectadas", st.session_state.total_detectadas)
c2.metric("En historial", len(st.session_state.historial))
c3.metric("Modelo", "YOLOv8n")

# ─── HISTORIAL ───────────────────────────────────────────────────────────────
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
        st.rerun()
