import io
import json
import re
from datetime import datetime

import numpy as np
import streamlit as st
from PIL import Image, ImageOps
from tensorflow.keras.models import load_model

# Optional report readers
try:
    from pypdf import PdfReader
except ImportError:
    PdfReader = None

try:
    from docx import Document
except ImportError:
    Document = None


st.set_page_config(
    page_title="RetinaAI | Multimodal Analysis",
    page_icon="👁️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------- Styling ----------
st.markdown("""
<style>
/* Force a consistent light, high-contrast interface regardless of browser theme. */
.stApp, [data-testid="stAppViewContainer"], [data-testid="stMain"] {
    background: #f4f7fb !important;
    color: #17233b !important;
}
[data-testid="stHeader"] { background: rgba(244,247,251,.92) !important; }
[data-testid="stSidebar"] {
    background: #ffffff !important;
    border-right: 1px solid #e5eaf2 !important;
}
[data-testid="stSidebar"] * { color: #26344d !important; }
.block-container { padding-top: 1.6rem; padding-bottom: 2.5rem; max-width: 1380px; }
h1, h2, h3, h4, p, label, [data-testid="stMarkdownContainer"] {
    color: #17233b;
}
.hero {
    background: linear-gradient(120deg,#101d3a 0%,#244b86 100%);
    padding: 30px 34px; border-radius: 20px; color: #ffffff;
    margin: 4px 0 26px; box-shadow: 0 10px 28px rgba(26,54,93,.12);
}
.hero h1 { font-size: 2.15rem; margin: 0 0 8px; color: #ffffff !important; }
.hero p { color: #e2ebff !important; margin: 0; font-size: 1.02rem; }
.eyebrow { letter-spacing: .14em; text-transform: uppercase; font-size: .72rem;
    font-weight: 800; color: #a9c4ff !important; margin-bottom: 9px; }
.section-title { font-size: 1.15rem; font-weight: 800; color: #17233b !important; margin: 8px 0 12px; }
.metric {
    border: 1px solid #e3eaf3; border-radius: 15px; padding: 16px 18px;
    background: #ffffff; min-height: 94px;
}
.metric .label { font-size: .76rem; color: #64748b !important; font-weight: 700; letter-spacing: .04em; }
.metric .value { font-size: 1.5rem; color: #17233b !important; font-weight: 850; margin-top: 5px; }
.small-muted, [data-testid="stCaptionContainer"] { color: #64748b !important; }
[data-testid="stFileUploader"] {
    background: #ffffff; border: 1px dashed #bdcbe0; border-radius: 14px; padding: 10px;
}
[data-testid="stFileUploader"] * { color: #26344d !important; }
[data-testid="stFileUploaderDropzone"] { background: #f8faff !important; }
div.stButton > button[kind="primary"] {
    background: #315efb !important; color: #ffffff !important; border: 1px solid #315efb !important;
    border-radius: 11px; font-weight: 750; padding: .72rem 1.1rem;
}
div.stButton > button[kind="primary"]:disabled { background: #a8b8e8 !important; color: #ffffff !important; }
.stAlert { border-radius: 12px; }
[data-testid="stExpander"] { background: #ffffff; border: 1px solid #e5eaf2; border-radius: 12px; }
[data-testid="stProgress"] > div > div { background: #315efb; }
footer { visibility: hidden; }
</style>
""", unsafe_allow_html=True)

# ---------- Paths / model ----------
MODEL_PATH = "retinal_fusion_model_compatible.keras"
CONFIG_PATH = "model_config.json"

@st.cache_resource(show_spinner="Loading retinal AI model...")
def get_model():
    return load_model(MODEL_PATH, compile=False)

@st.cache_data
def get_config():
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        return json.load(f)

def extract_report_text(uploaded_file):
    name = uploaded_file.name.lower()
    raw = uploaded_file.getvalue()

    if name.endswith(".txt"):
        for encoding in ("utf-8", "utf-16", "latin-1"):
            try:
                return raw.decode(encoding)
            except UnicodeDecodeError:
                continue
        return raw.decode("utf-8", errors="replace")

    if name.endswith(".pdf"):
        if PdfReader is None:
            raise RuntimeError("PDF reader missing. Install pypdf using: pip install pypdf")
        reader = PdfReader(io.BytesIO(raw))
        return "\n".join(page.extract_text() or "" for page in reader.pages)

    if name.endswith(".docx"):
        if Document is None:
            raise RuntimeError("DOCX reader missing. Install python-docx using: pip install python-docx")
        doc = Document(io.BytesIO(raw))
        return "\n".join(p.text for p in doc.paragraphs)

    raise ValueError("Unsupported report format. Upload PDF, DOCX, or TXT.")

def report_features(report_text, columns):
    """Build the six binary report inputs expected by this repository's model."""
    text = re.sub(r"\s+", " ", report_text.lower())
    negations = ("no", "not", "without", "negative for", "absence of", "absent")
    features = []
    for keyword in columns:
        escaped = re.escape(keyword.lower())
        found = bool(re.search(r"\b" + escaped + r"\b", text))
        negated = False
        if found:
            # Check a short context immediately before each keyword occurrence.
            for match in re.finditer(r"\b" + escaped + r"\b", text):
                before = text[max(0, match.start() - 45):match.start()]
                if any(re.search(r"\b" + re.escape(n) + r"\b(?:\s+\w+){0,3}\s*$", before)
                       for n in negations):
                    negated = True
                    break
        features.append(0.0 if negated else float(found))
    return np.asarray(features, dtype=np.float32).reshape(1, len(columns))

def run_prediction(model, config, image):
    size = tuple(config["image_size"])
    image = ImageOps.exif_transpose(image).convert("RGB").resize(size)
    image_array = np.asarray(image, dtype=np.float32)[None, ...]
    report_array = report_features(st.session_state.report_text, config["report_columns"])

    prediction = model.predict([image_array, report_array], verbose=0)
    probs = np.asarray(prediction).reshape(-1)

    labels = config["findings"]
    if len(probs) != len(labels):
        raise ValueError(
            f"Model returned {len(probs)} outputs but config lists {len(labels)} findings."
        )
    threshold = float(config.get("threshold", 0.5))
    return [
        {"finding": label.replace("_", " ").title(),
         "probability": float(prob),
         "status": "Detected" if float(prob) >= threshold else "Not detected"}
        for label, prob in zip(labels, probs)
    ], image

# Initialize state used across Streamlit reruns.
if "report_text" not in st.session_state:
    st.session_state.report_text = ""
if "analysis_results" not in st.session_state:
    st.session_state.analysis_results = None

# ---------- Sidebar ----------
with st.sidebar:
    st.markdown("## 👁️ RetinaAI")
    st.caption("Multimodal Retinal Analysis")
    st.markdown("---")
    page = st.radio("NAVIGATION", ["Analysis", "About the Model"], label_visibility="visible")
    st.markdown("---")
    st.markdown("**Model inputs**")
    st.caption("1. Retinal fundus image")
    st.caption("2. Clinical report (PDF / DOCX / TXT)")
    st.markdown("---")
    st.caption("Research prototype • Not for clinical diagnosis")

# ---------- Header ----------
st.markdown("""
<div class="hero">
  <div class="eyebrow">MULTIMODAL DEEP LEARNING</div>
  <h1>RetinaAI Analysis Studio</h1>
  <p>Upload a retinal image and its clinical report to generate an AI-assisted research prediction.</p>
</div>
""", unsafe_allow_html=True)

if page == "About the Model":
    st.markdown("### Model overview")
    c1, c2, c3 = st.columns(3)
    c1.markdown('<div class="metric"><div class="label">IMAGE BRANCH</div><div class="value">EfficientNetB0</div></div>', unsafe_allow_html=True)
    c2.markdown('<div class="metric"><div class="label">REPORT BRANCH</div><div class="value">Dense Network</div></div>', unsafe_allow_html=True)
    c3.markdown('<div class="metric"><div class="label">FUSION</div><div class="value">Concatenation</div></div>', unsafe_allow_html=True)
    st.markdown("")
    st.info("The repository describes a multi-label model for Hemorrhage, Hard Exudates, Cotton Wool Spots, and Retinal Detachment.")
    st.warning("The repository reports preliminary results on a small test dataset. This interface is for educational and research purposes only and must not be used to diagnose, treat, or make medical decisions.")
    st.stop()

# ---------- Upload workflow ----------
left, right = st.columns([1, 1], gap="large")

with left:
    st.markdown('<div class="section-title">01 · Retinal image</div>', unsafe_allow_html=True)
    image_file = st.file_uploader(
        "Upload fundus image", type=["png", "jpg", "jpeg"],
        help="Supported formats: PNG, JPG, JPEG."
    )
    if image_file:
        try:
            preview = Image.open(image_file)
            st.image(preview, caption=image_file.name, use_container_width=True)
            st.caption(f"Image dimensions: {preview.width} × {preview.height}")
        except Exception:
            st.error("This image could not be opened. Please upload a valid image.")

with right:
    st.markdown('<div class="section-title">02 · Clinical report</div>', unsafe_allow_html=True)
    report_file = st.file_uploader(
        "Upload report", type=["pdf", "docx", "txt"],
        help="Upload a text-based PDF, DOCX, or TXT report."
    )
    report_text = ""
    if report_file:
        try:
            report_text = extract_report_text(report_file)
            st.session_state.report_text = report_text
            if report_text.strip():
                st.success(f"Report loaded · {len(report_text.split())} words extracted")
                with st.expander("Preview extracted report text"):
                    st.text_area("Extracted text", report_text, height=180, disabled=True)
            else:
                st.warning("No readable text was extracted. Scanned/image-only PDFs need OCR and are not supported in this version.")
        except Exception as exc:
            st.error(str(exc))

st.markdown("---")
st.markdown('<div class="section-title">03 · Run multimodal analysis</div>', unsafe_allow_html=True)
ready = bool(image_file and report_file and report_text.strip())

if not ready:
    st.info("Upload both a retinal image and a readable clinical report to enable analysis.")

if st.button("🔍  Analyze retinal image", type="primary", disabled=not ready, use_container_width=True):
    try:
        config = get_config()
        model = get_model()
        with st.spinner("Processing image and report through the multimodal model..."):
            results, processed_image = run_prediction(model, config, Image.open(image_file))
        st.session_state.analysis_results = results
        st.session_state.analysis_image = processed_image
        st.session_state.analysis_time = datetime.now().strftime("%d %b %Y, %I:%M %p")
        st.session_state.analysis_filename = image_file.name
    except FileNotFoundError as exc:
        st.error(f"Required project file not found: {exc}. Keep app.py, model_config.json, and retinal_fusion_model.keras in the same folder.")
    except Exception as exc:
        st.error(f"Analysis failed: {exc}")

# ---------- Results ----------
if st.session_state.get("analysis_results"):
    st.markdown("---")
    st.markdown("## Analysis results")
    st.caption(f"Completed {st.session_state.analysis_time} · {st.session_state.analysis_filename}")
    results = st.session_state.analysis_results

    detected = sum(r["status"] == "Detected" for r in results)
    a, b, c = st.columns(3)
    a.markdown(f'<div class="metric"><div class="label">FINDINGS ANALYSED</div><div class="value">{len(results)}</div></div>', unsafe_allow_html=True)
    b.markdown(f'<div class="metric"><div class="label">MODEL FLAGS</div><div class="value">{detected}</div></div>', unsafe_allow_html=True)
    c.markdown(f'<div class="metric"><div class="label">THRESHOLD</div><div class="value">{float(get_config().get("threshold", .5)):.0%}</div></div>', unsafe_allow_html=True)

    st.markdown("### Finding-wise probabilities")
    for item in results:
        col1, col2, col3 = st.columns([2, 1, 3])
        col1.markdown(f"**{item['finding']}**")
        col2.markdown("🟠 Detected" if item["status"] == "Detected" else "🟢 Not detected")
        col3.progress(min(max(item["probability"], 0.0), 1.0),
                      text=f"{item['probability']:.1%} model score")

    report_lines = [
        "RETINAAI — MULTIMODAL ANALYSIS REPORT",
        f"Date: {st.session_state.analysis_time}",
        f"Image: {st.session_state.analysis_filename}",
        "",
        "MODEL OUTPUTS (research use only):",
    ]
    report_lines += [
        f"- {r['finding']}: {r['status']} | Probability: {r['probability']:.4f}"
        for r in results
    ]
    report_lines += [
        "",
        "DISCLAIMER: Educational/research prototype only. Not a clinical diagnosis.",
        "A qualified eye-care professional must interpret clinical findings."
    ]
    st.download_button(
        "⬇️ Download analysis report (.txt)",
        data="\n".join(report_lines).encode("utf-8"),
        file_name="retinaai_analysis_report.txt",
        mime="text/plain",
    )

    st.warning("Important: These are model predictions, not a medical diagnosis. Do not use them to make treatment decisions. Consult a qualified eye-care professional.")
