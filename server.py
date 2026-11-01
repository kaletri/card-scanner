#!/usr/bin/env python3
"""
Card Scanner — Home-PC visual recognition backend (v1).

Serves DINOv2 visual-similarity search over the full TCGdex catalog.

Endpoints:
    GET  /health     -> service status + catalog size
    POST /identify   -> multipart form field "photo" (JPEG/PNG of the card,
                        ideally the same auto-crop the app already produces)
                        -> top-K visual matches with cosine scores, plus
                        an "ocr" object with the server-side EasyOCR reads
                        of the name band and collector-number strip
                        (null when easyocr is not installed).

The phone app treats these scores as the visual signal and fuses them with
its existing collector-number OCR + name fuzzy match (see SETUP.md).
When this backend is unreachable the app keeps its on-device pipeline.

Run:
    uvicorn server:app --host 0.0.0.0 --port 8765

Model: facebook/dinov2-base (Apache-2.0), CLS token, L2-normalized,
cosine similarity. CPU is fine for per-scan inference; set DEVICE=cuda
(or just let torch auto-pick) if the box has an NVIDIA GPU.
"""

import io
import json
import os
import re
import time
from contextlib import asynccontextmanager
from pathlib import Path

import numpy as np
import torch
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from PIL import Image

BASE_DIR = Path(__file__).resolve().parent
INDEX_PATH = Path(os.environ.get("INDEX_PATH", BASE_DIR / "index.npy"))
META_PATH = Path(os.environ.get("META_PATH", BASE_DIR / "index-meta.json"))
MODEL_ID = os.environ.get("DINOV2_MODEL", "facebook/dinov2-base")
TOP_K_DEFAULT = int(os.environ.get("TOP_K", "10"))
MAX_TOP_K = 25

# Lazy singletons (loaded once at startup).
_processor = None
_model = None
_device = None
_index = None        # (N, 768) float32, rows already L2-normalized
_meta = None         # list of {id, name, number, setId, setName} aligned to rows


def _embed(pil_image: Image.Image) -> np.ndarray:
    """DINOv2 CLS-token embedding, L2-normalized, as a (768,) float32 vector."""
    inputs = _processor(images=pil_image.convert("RGB"), return_tensors="pt")
    pixel_values = inputs["pixel_values"].to(_device)
    with torch.no_grad():
        outputs = _model(pixel_values=pixel_values)
        cls = outputs.last_hidden_state[:, 0]          # (1, 768) CLS token
        cls = torch.nn.functional.normalize(cls, p=2, dim=-1)
    return cls[0].cpu().numpy().astype(np.float32)


def _card_views(pil_image: Image.Image):
    """Framings of an upload to score against the index.

    Real scan photos are often loose — the card small in a wide frame, or
    edges slightly chopped — and scoring only the raw frame lets the
    background (or the missing edges) dominate the embedding. Each upload
    is scored under several framings and a card keeps its BEST score:
      - "full":     the image exactly as received
      - "card100" / "card92" / "card84": centered crops with a real
                    card's aspect ratio (63x88 mm) at 100% / 92% / 84%
                    of the frame's limiting dimension
      - "center75": the central 75% of the frame
      - "art":      the top ~62% of the card100 crop (the art window) —
                    holo/reverse-holo printings share the plain printing's
                    artwork but differ across the card body
    Returns a list of (name, image).
    """
    img = pil_image.convert("RGB")
    w, h = img.size
    views = [("full", img)]
    target = 63 / 88
    # Card-aspect crops at several scales: on a real scan the card may
    # fill the frame or sit smaller inside it, so judge it at the zoom
    # where it actually lives.
    for scale in (1.0, 0.92, 0.84):
        if w / h > target:
            ch = int(h * scale)
            cw = int(ch * target)
        else:
            cw = int(w * scale)
            ch = int(cw / target)
        x0, y0 = (w - cw) // 2, (h - ch) // 2
        views.append((f"card{int(scale * 100)}", img.crop((x0, y0, x0 + cw, y0 + ch))))
    tw, th = int(w * 0.75), int(h * 0.75)
    x0, y0 = (w - tw) // 2, (h - th) // 2
    views.append(("center75", img.crop((x0, y0, x0 + tw, y0 + th))))
    # Art window: the top ~62% of the card-aspect crop. Holo and
    # reverse-holo printings share identical artwork with the plain
    # printing but differ across most of the card body, which dilutes
    # every full-card view; the art region stays comparable.
    full_card = views[1][1]  # card100
    cw2, ch2 = full_card.size
    views.append(("art", full_card.crop((0, 0, cw2, int(ch2 * 0.62)))))
    return views


# --- Server-side OCR (EasyOCR) -------------------------------------------
# The phone's Tesseract reads kept fragmenting ("Basic Coba 4 Cn 0", "zc
# Cobalion 01 209"), so the server now reads the two text bands itself,
# on its GPU, from the card100 crop. The app (build 21j) will treat these
# reads as the primary name/number signal; Tesseract stays only as the
# backend-down fallback. If easyocr is not installed the server still
# starts and serves visual-only results (ocr: null) — never fail a scan
# over a missing OCR package.
_ocr_reader = None
_ocr_tried = False


def _get_ocr_reader():
    """Lazy EasyOCR reader (English, GPU when available). None if the
    package is missing or failed to load."""
    global _ocr_reader, _ocr_tried
    if _ocr_tried:
        return _ocr_reader
    _ocr_tried = True
    try:
        import easyocr

        _ocr_reader = easyocr.Reader(["en"], gpu=torch.cuda.is_available(), verbose=False)
        print("[ocr] easyocr reader ready")
    except Exception as e:
        print(f"[ocr] easyocr unavailable ({e}); visual-only mode")
        _ocr_reader = None
    return _ocr_reader


def _ocr_band_crops(card_image: Image.Image):
    """Name band (top strip) and collector-number strip (bottom-left) of
    a card-aspect crop, upscaled 2x for the reader. Pure PIL — safe to
    test without torch/easyocr installed."""
    w, h = card_image.size
    name = card_image.crop((0, 0, w, int(h * 0.12)))
    number = card_image.crop((0, int(h * 0.92), int(w * 0.45), h))
    return (
        name.resize((name.width * 2, name.height * 2), Image.LANCZOS),
        number.resize((number.width * 2, number.height * 2), Image.LANCZOS),
    )


def _pick_number_line(number_lines):
    """Pick the collector number from EasyOCR number-strip lines.

    The number is the leftmost text in the bottom-left corner. Returns
    (number, conf) where number is "123/214"-shaped, or (None, 0.0)
    when no line is a collector number — never a digit-soup guess.
    Pure function (no torch/easyocr) so it can be unit-tested alone.
    """
    cands = []
    for bbox, text, conf in number_lines:
        cleaned = "".join(ch for ch in text if ch.isdigit() or ch == "/")
        m = re.fullmatch(r"(\d{1,3})/(\d{1,3})", cleaned)
        if not m:
            continue
        xs = [p[0] for p in bbox]
        cands.append((min(xs), -conf, m.group(0), conf))
    if not cands:
        return None, 0.0
    cands.sort()
    return cands[0][2], cands[0][3]


def _read_ocr(card_image: Image.Image):
    """Read the name band and number strip. Returns
    {"name": str|None, "name_conf": float, "number": str|None,
     "number_conf": float}, or None when the reader is unavailable or
    the read fails. The raw strings are returned as-is (junk included);
    the app's matcher cleans them."""
    reader = _get_ocr_reader()
    if reader is None:
        return None
    try:
        name_crop, number_crop = _ocr_band_crops(card_image)
        name_lines = reader.readtext(np.array(name_crop))
        number_lines = reader.readtext(np.array(number_crop), allowlist="0123456789/")
        name = " ".join(t for _, t, _ in name_lines) or None
        name_conf = max((c for _, _, c in name_lines), default=0.0)
        # Number strip: the collector number is the leftmost text in the
        # bottom-left corner. Joining every detected line concatenates
        # it with copyright-year digits ("0010046080/0590008"), so
        # _pick_number_line keeps only a line that IS a collector
        # number — anything else means no number signal, never a guess.
        number, number_conf = _pick_number_line(number_lines)
        return {
            "name": name,
            "name_conf": round(float(name_conf), 3),
            "number": number,
            "number_conf": round(float(number_conf), 3),
        }
    except Exception as e:
        print(f"[ocr] read failed: {e}")
        return None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load model + index once at startup (lifespan API; on_event is deprecated)."""
    global _processor, _model, _device, _index, _meta
    if not INDEX_PATH.exists() or not META_PATH.exists():
        raise RuntimeError(
            f"Index not found. Run build_index.py first "
            f"(expected {INDEX_PATH} and {META_PATH})."
        )
    from transformers import AutoImageProcessor, Dinov2Model  # deferred: slow import

    _device = "cuda" if torch.cuda.is_available() else "cpu"
    _processor = AutoImageProcessor.from_pretrained(MODEL_ID)
    _model = Dinov2Model.from_pretrained(MODEL_ID).to(_device).eval()
    _index = np.load(INDEX_PATH).astype(np.float32)
    _meta = json.loads(META_PATH.read_text(encoding="utf-8"))
    assert len(_index) == len(_meta), "index.npy and index-meta.json out of sync"
    yield


app = FastAPI(title="Card Scanner visual backend", version="1.0.0", lifespan=lifespan)

# The web app runs on https://kaletri.github.io and calls this server
# cross-origin; without CORS headers the browser refuses to hand the
# response to the page. The server is only reachable inside Joseph's
# private tailnet, so a permissive policy is acceptable here.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health():
    return {
        "status": "ok",
        "model": MODEL_ID,
        "device": _device,
        "catalog_size": int(len(_index)) if _index is not None else 0,
        "embedding_dim": int(_index.shape[1]) if _index is not None else 0,
    }


@app.post("/identify")
async def identify(photo: UploadFile = File(...), top_k: int = TOP_K_DEFAULT):
    if top_k < 1 or top_k > MAX_TOP_K:
        raise HTTPException(status_code=400, detail=f"top_k must be 1..{MAX_TOP_K}")
    raw = await photo.read()
    if not raw:
        raise HTTPException(status_code=400, detail="empty photo")
    try:
        pil = Image.open(io.BytesIO(raw))
    except Exception:
        raise HTTPException(status_code=400, detail="could not decode image")

    if _index is None or len(_index) == 0:
        return JSONResponse({"results": []})
    # Timings are diagnostic (21j speed work): the log line carries the
    # visual vs OCR split so a slow scan can be blamed on evidence.
    t0 = time.perf_counter()
    views = _card_views(pil)
    vecs = np.stack([_embed(im) for _, im in views])  # (V, 768), unit norm
    all_scores = vecs @ _index.T                      # (V, N) cosine
    scores = all_scores.max(axis=0)                   # best framing per card
    best_view = all_scores.argmax(axis=0)             # which framing won
    k = min(top_k, len(scores))
    top = np.argpartition(scores, -k)[-k:]
    top = top[np.argsort(scores[top])[::-1]]  # descending

    results = [
        {
            "id": _meta[i]["id"],
            "name": _meta[i]["name"],
            "number": _meta[i]["number"],
            "setId": _meta[i]["setId"],
            "setName": _meta[i]["setName"],
            "score": float(scores[i]),  # cosine similarity, ~ -1..1
        }
        for i in top
    ]

    # Diary of what we answered (top 10), so a bad identification can be
    # diagnosed from evidence instead of guesswork. Console + file.
    from datetime import datetime

    stamp = datetime.now().isoformat(timespec="seconds")
    summary = ", ".join(f"{r['id']}={r['score']:.3f}" for r in results[:10])
    top_view = views[best_view[top[0]]][0] if len(top) else "?"
    # Server-side OCR read of the card100 crop (the app's primary
    # name/number signal from build 21j onward).
    t_visual = time.perf_counter() - t0
    ocr = _read_ocr(views[1][1])
    t_ocr = time.perf_counter() - t0 - t_visual
    t_total = time.perf_counter() - t0
    ocr_note = (
        f" ocr name={ocr['name']!r} num={ocr['number']!r}"
        if ocr
        else " ocr unavailable"
    )
    line = f"[{stamp}] identify -> {summary} (top via {top_view}){ocr_note} t={t_total:.2f}s (visual {t_visual:.2f}s, ocr {t_ocr:.2f}s)"
    print(line)
    try:
        with (BASE_DIR / "identify-log.txt").open("a", encoding="utf-8") as f:
            f.write(line + "\n")
    except Exception:
        pass  # logging must never break an identification

    return JSONResponse({
        "results": results,
        "ocr": ocr,
        "timings": {
            "total": round(t_total, 3),
            "visual": round(t_visual, 3),
            "ocr": round(t_ocr, 3),
        },
    })
