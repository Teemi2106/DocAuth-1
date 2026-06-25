"""
DocVerify - AI-Based Document Verification and Forgery Detection System
Flask web application entry point.

Run with:
    python web_app.py

Stack (matching Chapter 3 of thesis):
  Backend  : Python 3.10 + Flask
  Frontend : HTML / CSS / JavaScript (templates/ and static/)
  AI Layer : ELA -> ResNet50 CNN + EasyOCR + Copy-Move detection
  Database : SQLite via db.py

Fusion formula (Section 3.6.1):
    Final Score = 0.5 * cnn_score + 0.3 * ocr_score + 0.2 * forensic_score
Decision threshold (Section 3.6.2):
    >= 0.65 -> Genuine   |   < 0.65 -> Forged / Suspicious
"""

from __future__ import annotations

import base64
import io
import os
import sys
import tempfile
import traceback
import uuid
from pathlib import Path

# Fix Windows console encoding
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")

from flask import Flask, render_template, request, redirect, url_for, flash, abort

from db import init_db, save_document, save_result, get_document_with_result, get_all_documents

app = Flask(__name__)
app.secret_key = os.environ.get("SECRET_KEY", "docverify-dev-key-change-in-production")

UPLOAD_FOLDER = Path("uploads")
UPLOAD_FOLDER.mkdir(exist_ok=True)

ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "tiff", "bmp", "webp", "pdf"}
MAX_UPLOAD_BYTES = 16 * 1024 * 1024  # 16 MB


# ── Helpers ───────────────────────────────────────────────────────────────────

def _allowed(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def _prepare_image(upload_path: Path) -> Path:
    """Convert PDF first page to PNG; return path unchanged for images."""
    if upload_path.suffix.lower() == ".pdf":
        try:
            import fitz  # PyMuPDF
            doc = fitz.open(str(upload_path))
            page = doc[0]
            pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))
            img_path = upload_path.with_suffix(".png")
            pix.save(str(img_path))
            return img_path
        except ImportError:
            raise ValueError(
                "PDF support requires PyMuPDF. Install it with: pip install PyMuPDF"
            )
    return upload_path


def _pil_to_b64(pil_image) -> str:
    buf = io.BytesIO()
    pil_image.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


def _run_analysis(image_path: Path) -> dict:
    """
    Run the three-layer analysis pipeline and return all scores.

    Mapping to thesis:
      cnn_score      <- ELA suspicion inverted (visual/CNN proxy, weight 0.5)
      ocr_score      <- EasyOCR avg confidence        (weight 0.3)
      forensic_score <- Copy-move genuineness score   (weight 0.2)
    """
    from PIL import Image
    from src.analysis.ela import generate_ela, ela_score
    from src.analysis.ocr import extract_text
    from src.copy_move.detector import detect_copy_move

    print(f"\n[DocVerify] Starting analysis: {image_path}", flush=True)
    img_pil = Image.open(image_path).convert("RGB")

    # ── ELA + CNN (visual/CNN score, weight 0.5) ───────────────────────────────
    # ELA requires a JPEG baseline. PNG/BMP/TIFF inputs are lossless, so the
    # first-ever JPEG compression changes every pixel - making the whole image
    # look "manipulated". Pre-converting to JPEG first gives ELA a fair baseline.
    print("[DocVerify] Running ELA...", flush=True)
    lossless_exts = {".png", ".bmp", ".tiff", ".tif", ".webp"}
    if image_path.suffix.lower() in lossless_exts:
        jpeg_buf = io.BytesIO()
        img_pil.save(jpeg_buf, "JPEG", quality=90)
        jpeg_buf.seek(0)
        ela_source = Image.open(jpeg_buf).copy()
        print("[DocVerify] PNG detected - pre-converted to JPEG for ELA", flush=True)
    else:
        ela_source = img_pil
    ela_img = generate_ela(ela_source, quality=95, scale=15)
    raw_ela = ela_score(ela_img)
    ela_b64 = _pil_to_b64(ela_img)
    print(f"[DocVerify] ELA done - raw={raw_ela:.4f}", flush=True)

    # CNN is blended with ELA so an out-of-domain CNN prediction cannot
    # condemn a document alone. Floor raised to 0.50 so ELA never fully
    # overrides the other two signals.
    ela_fallback = float(max(0.50, min(1.0, 1.0 - raw_ela / 0.25)))
    try:
        from src.cnn.inference import predict_genuine, model_ready
        if model_ready():
            raw_cnn = predict_genuine(ela_img)
            # 40 % CNN + 60 % ELA - limits damage when CNN training data
            # doesn't match the uploaded document type
            cnn_score = 0.4 * raw_cnn + 0.6 * ela_fallback
            print(f"[DocVerify] CNN (ResNet50)={raw_cnn:.4f}  ELA={ela_fallback:.4f}  blended={cnn_score:.4f}", flush=True)
        else:
            cnn_score = ela_fallback
            print(f"[DocVerify] CNN not trained - ELA score={cnn_score:.4f}", flush=True)
    except Exception:
        cnn_score = ela_fallback
        print(f"[DocVerify] CNN error - ELA fallback={cnn_score:.4f}\n{traceback.format_exc()}", flush=True)

    # ── OCR ───────────────────────────────────────────────────────────────────
    # If EasyOCR finds no words, avg_confidence returns 0.0, not 0.5.
    # Documents with no readable text should score neutrally (0.70), not zero.
    print("[DocVerify] Running OCR (may take a moment on first run)...", flush=True)
    ocr_text = ""
    ocr_score = 0.70  # neutral default
    try:
        ocr_result = extract_text(image_path)
        raw_conf = float(ocr_result.get("avg_confidence", 0.0))
        words = ocr_result.get("words", [])
        if words:
            # Floor at 0.40: very low confidence usually reflects image quality
            # (small text, screen rendering) rather than deliberate text manipulation.
            ocr_score = max(0.40, raw_conf)
        else:
            ocr_score = 0.70  # no text found -> neutral
        ocr_text = ocr_result.get("full_text", "")
        print(f"[DocVerify] OCR done - confidence={ocr_score:.4f}, words={len(words)}", flush=True)
    except Exception:
        print(f"[DocVerify] OCR failed (using neutral 0.70):\n{traceback.format_exc()}", flush=True)

    # ── Copy-Move Detection (forensic score) ──────────────────────────────────
    print("[DocVerify] Running copy-move detection...", flush=True)
    forensic_score = 0.5
    cm_method = "N/A"
    try:
        cm_result = detect_copy_move(image_path)
        forensic_score = float(max(0.0, 1.0 - cm_result["score"]))
        cm_method = cm_result.get("method", "orb_ransac")
        print(f"[DocVerify] Copy-move done - score={forensic_score:.4f}, method={cm_method}", flush=True)
    except Exception:
        print(f"[DocVerify] Copy-move failed (using neutral 0.5):\n{traceback.format_exc()}", flush=True)

    # ── Fusion (thesis formula, Section 3.6.1) ────────────────────────────────
    print(f"[DocVerify] Computing fusion score...", flush=True)
    final_score = 0.5 * cnn_score + 0.3 * ocr_score + 0.2 * forensic_score

    if final_score >= 0.65:
        verdict = "Genuine"
    elif final_score >= 0.40:
        verdict = "Suspicious"
    else:
        verdict = "Forged"

    print(f"[DocVerify] Final score={final_score:.4f} => {verdict}", flush=True)

    return {
        "cnn_score": round(cnn_score, 4),
        "ocr_score": round(ocr_score, 4),
        "forensic_score": round(forensic_score, 4),
        "final_score": round(final_score, 4),
        "verdict": verdict,
        "ela_image": ela_b64,
        "ocr_text": ocr_text,
        "raw_ela": round(raw_ela, 6),
        "cm_method": cm_method,
    }


# ── Routes ────────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/analyze", methods=["POST"])
def analyze():
    if "document" not in request.files:
        flash("No file selected.", "error")
        return redirect(url_for("index"))

    file = request.files["document"]

    if not file or file.filename == "":
        flash("No file selected.", "error")
        return redirect(url_for("index"))

    if not _allowed(file.filename):
        flash("Unsupported file type. Upload PNG, JPG, TIFF, BMP, or PDF.", "error")
        return redirect(url_for("index"))

    # Save upload
    safe_name = f"{uuid.uuid4().hex}_{Path(file.filename).name}"
    upload_path = UPLOAD_FOLDER / safe_name
    file.save(upload_path)

    # Convert PDF -> image if needed
    try:
        image_path = _prepare_image(upload_path)
    except ValueError as exc:
        flash(str(exc), "error")
        return redirect(url_for("index"))

    # Run analysis
    try:
        results = _run_analysis(image_path)
    except Exception as exc:
        print(f"[DocVerify] ANALYSIS CRASH:\n{traceback.format_exc()}", flush=True)
        flash(f"Analysis failed: {exc}", "error")
        return redirect(url_for("index"))

    # Persist to database
    doc_id = save_document(str(upload_path), file.filename, results["verdict"])
    save_result(
        document_id=doc_id,
        cnn_score=results["cnn_score"],
        ocr_score=results["ocr_score"],
        forensic_score=results["forensic_score"],
        final_score=results["final_score"],
        verdict=results["verdict"],
        ocr_text=results["ocr_text"],
        ela_image=results["ela_image"],
    )

    return redirect(url_for("result", doc_id=doc_id))


@app.route("/result/<int:doc_id>")
def result(doc_id: int):
    data = get_document_with_result(doc_id)
    if data is None:
        abort(404)
    return render_template("result.html", data=data)


@app.route("/history")
def history():
    docs = get_all_documents()
    return render_template("history.html", docs=docs)


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    init_db()
    print("[DocVerify] Server starting at http://127.0.0.1:5000", flush=True)
    # use_reloader=False prevents double-starts and connection resets on Windows
    app.run(debug=True, host="0.0.0.0", port=5000, use_reloader=False)
