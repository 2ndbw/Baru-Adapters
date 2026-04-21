import subprocess, sys, os, io, base64, json
import numpy as np
from flask import Flask, render_template, jsonify, request
from PIL import Image
from sklearn.cluster import KMeans

sys.path.insert(0, os.path.dirname(__file__))
import land_classifier as lc

app = Flask(__name__)
BASE = os.path.dirname(__file__)

ADAPTERS = [
    {
        "id": "accounting",
        "name": "Accounting Period Close",
        "desc": "Books off by $1,200 at Q3. Baru finds the 2 wrong entries and fixes them.",
        "file": "demo_accounting.py",
        "runnable": True,
        "tags": ["finance", "demo"],
    },
    {
        "id": "staff",
        "name": "Staff Rotation Cycle",
        "desc": "12 nurses, 12 slots. After a rough quarter the rotation is off by 2 slots. Baru fixes it in 1 swap.",
        "file": "demo_staff_rotation.py",
        "runnable": True,
        "tags": ["scheduling", "demo"],
    },
    {
        "id": "robot",
        "name": "Warehouse Robot Path",
        "desc": "Robot stranded 3.6m from dock after a bad route. Baru corrects 2 moves and gets it home.",
        "file": "demo_warehouse_robot.py",
        "runnable": True,
        "tags": ["robotics", "demo"],
    },
    {
        "id": "base",
        "name": "BaruAdapter Base Class",
        "desc": "The easy way to build any Baru adapter. Answer 4 questions, Baru does the rest. Includes Counter, Clock, Toggle, Steps built-ins.",
        "file": "BaruAdapter.py",
        "runnable": True,
        "tags": ["core", "toolkit"],
    },
    {
        "id": "isodata",
        "name": "ISODATA Classifier Optimizer",
        "desc": "Real ISODATA on 2,000 synthetic Landsat 8 pixels — 9 bands + NDVI/NDWI/NDBI = 12 features, 5 land cover classes (Nevada mine site). Baru sits inside the parameter loop and finds minimum adjustments to reach publishable Kappa ≥ 0.85.",
        "file": "baru_isodata_classifier.py",
        "runnable": True,
        "tags": ["geospatial", "classification", "demo"],
    },
    {
        "id": "erdas",
        "name": "ERDAS IMAGINE Pipeline",
        "desc": "Landsat 8 satellite image processing for an open-pit mine site. Baru catches missing atmospheric correction and a duplicate step — 1 change each.",
        "file": "baru_erdas_pipeline.py",
        "runnable": True,
        "tags": ["geospatial", "demo"],
    },
    {
        "id": "catalog",
        "name": "Catalog OCR Extraction",
        "desc": "Fixes failed OCR scans in a grid-based catalog. Blurry text? Bad crop? Unknown model? Baru picks the right correction move.",
        "file": "G2_CatalogExtractionAdapter.py",
        "runnable": False,
        "tags": ["vision", "blueprint"],
    },
    {
        "id": "harmonic",
        "name": "Harmonic Cadence (Music)",
        "desc": "Resolves any chord back to the tonic using classical/jazz voice-leading rules. Circle of fifths, tritone subs, stepwise bass.",
        "file": "G3_HarmonicCadenceAdapter.py",
        "runnable": False,
        "tags": ["music", "blueprint"],
    },
    {
        "id": "g1base",
        "name": "Baru Base Interface (Gemini)",
        "desc": "Gemini's take on the Baru interface — 4 abstract methods defining any adapter domain.",
        "file": "G1_BaruAdapter_base.py",
        "runnable": False,
        "tags": ["core", "blueprint"],
    },
]


@app.route("/")
def index():
    return render_template("index.html", adapters=ADAPTERS)


@app.route("/run/<adapter_id>")
def run_adapter(adapter_id):
    adapter = next((a for a in ADAPTERS if a["id"] == adapter_id), None)
    if not adapter:
        return jsonify({"error": "Adapter not found"}), 404
    if not adapter["runnable"]:
        return jsonify({"error": "This is a blueprint — no runnable demo yet."}), 400

    script = os.path.join(BASE, adapter["file"])
    result = subprocess.run(
        [sys.executable, script],
        capture_output=True, text=True, timeout=15, cwd=BASE
    )
    output = result.stdout or result.stderr
    return jsonify({"output": output, "ok": result.returncode == 0})


@app.route("/source/<adapter_id>")
def source(adapter_id):
    adapter = next((a for a in ADAPTERS if a["id"] == adapter_id), None)
    if not adapter:
        return jsonify({"error": "Not found"}), 404
    path = os.path.join(BASE, adapter["file"])
    with open(path) as f:
        return jsonify({"source": f.read(), "file": adapter["file"]})


# ── Land Use Classifier ──────────────────────────────────────────────────────

# Distinct colors for up to 12 land use classes
CLASS_COLORS = [
    (70,  130, 180),   # steel blue     — water
    (34,  139,  34),   # forest green   — dense vegetation
    (144, 238, 144),   # light green    — grassland / crops
    (210, 180, 140),   # tan            — bare soil / sand
    (169, 169, 169),   # gray           — urban / rock / pavement
    (139,  90,  43),   # brown          — mine waste / disturbed ground
    (255, 215,   0),   # gold           — sparse scrub / dry grass
    ( 30,  30,  30),   # near-black     — shadow / deep water / pit
    (255, 160, 122),   # salmon         — exposed clay / iron oxide
    (135, 206, 235),   # sky blue       — shallow water / wetland
    (128,   0, 128),   # purple         — mixed urban
    (255, 255, 255),   # white          — cloud / snow / salt flat
]

CLASS_LABELS = [
    'Water', 'Dense vegetation', 'Grassland / crops',
    'Bare soil / sand', 'Urban / rock', 'Mine waste / disturbed',
    'Dry scrub', 'Shadow / deep pit', 'Clay / iron oxide',
    'Shallow water', 'Mixed urban', 'Cloud / snow',
]


@app.route("/classifier")
def classifier():
    return render_template("classifier.html")


@app.route("/api/classify", methods=["POST"])
def api_classify():
    if "image" not in request.files:
        return jsonify({"error": "No image uploaded"}), 400

    f   = request.files["image"]
    k   = max(2, min(12, int(request.form.get("k", 6))))
    sz  = int(request.form.get("size", 512))

    try:
        img = Image.open(f).convert("RGB")
    except Exception as e:
        return jsonify({"error": f"Could not read image: {e}"}), 400

    # Resize keeping aspect ratio
    img.thumbnail((sz, sz), Image.LANCZOS)
    w, h = img.size

    # Pixels → feature matrix
    pixels = np.array(img).reshape(-1, 3).astype(float) / 255.0

    # KMeans clustering in RGB space
    km = KMeans(n_clusters=k, n_init=10, max_iter=100, random_state=42)
    km.fit(pixels)
    labels   = km.labels_
    centers  = km.cluster_centers_   # (k, 3) in 0–1 range

    # Sort clusters by brightness so colors are assigned consistently
    brightness = centers.mean(axis=1)
    order = np.argsort(brightness)
    remap = {old: new for new, old in enumerate(order)}
    labels_sorted = np.array([remap[l] for l in labels])

    # Build classified image
    color_map = np.array(CLASS_COLORS[:k], dtype=np.uint8)
    classified_pixels = color_map[labels_sorted]
    classified_img = Image.fromarray(
        classified_pixels.reshape(h, w, 3).astype(np.uint8)
    )

    # Class statistics
    counts = np.bincount(labels_sorted, minlength=k)
    total  = counts.sum()
    classes_info = []
    for i in range(k):
        orig_center = centers[order[i]]
        classes_info.append({
            "label": CLASS_LABELS[i],
            "color": list(CLASS_COLORS[i]),
            "pct":   round(100 * counts[i] / total, 1),
            "center_rgb": [round(float(v) * 255) for v in orig_center],
        })

    # Encode both images as base64
    def img_to_b64(image):
        buf = io.BytesIO()
        image.save(buf, format="PNG")
        return base64.b64encode(buf.getvalue()).decode()

    return jsonify({
        "original":   img_to_b64(img),
        "classified": img_to_b64(classified_img),
        "n_classes":  k,
        "width":      w,
        "height":     h,
        "n_pixels":   int(total),
        "iterations": 100,
        "classes":    classes_info,
    })


@app.route("/api/classify_trained", methods=["POST"])
def classify_trained():
    if "image" not in request.files:
        return jsonify({"error": "No image uploaded"}), 400

    f = request.files["image"]
    try:
        raw_training = json.loads(request.form.get("training", "{}"))
        brush_radius = int(request.form.get("brush_radius", 8))
    except Exception as e:
        return jsonify({"error": f"Bad form data: {e}"}), 400

    try:
        img = lc.load_and_resize(f)
    except Exception as e:
        return jsonify({"error": f"Could not read image: {e}"}), 400

    img_array = np.array(img)

    # Build training dict: {class_id (int): ndarray (n, 3)}
    training = {}
    for cid_str, points in raw_training.items():
        cid = int(cid_str)
        if not points:
            continue
        pixels = lc.sample_pixels(img_array, points, brush_radius)
        if len(pixels) >= 3:
            training[cid] = pixels

    if len(training) < 2:
        return jsonify({"error": "Need at least 2 classes with painted samples"}), 400

    try:
        pred_map, clf, class_ids = lc.classify_image(img, training)
    except Exception as e:
        return jsonify({"error": f"Classification failed: {e}"}), 500

    classified_img = lc.render_classified(pred_map, class_ids)

    # Separability + Baru
    signatures, pairs, total_deficit = lc.compute_separability(training)
    optimizer, segments = lc.build_baru_optimizer(training, pairs, total_deficit, class_ids)

    baru_info = _build_baru_info(pairs, total_deficit, optimizer, class_ids)

    def img_to_b64(image):
        buf = io.BytesIO()
        image.save(buf, format="PNG")
        return base64.b64encode(buf.getvalue()).decode()

    legend = []
    for cid in class_ids:
        if cid < len(lc.CLASS_PALETTE):
            p = lc.CLASS_PALETTE[cid]
            legend.append({"id": cid, "label": p["label"], "hex": p["hex"]})

    return jsonify({
        "map":    img_to_b64(classified_img),
        "legend": legend,
        "baru":   baru_info,
    })


@app.route("/api/baru_fix", methods=["POST"])
def baru_fix_route():
    if "image" not in request.files:
        return jsonify({"error": "No image uploaded"}), 400

    f = request.files["image"]
    try:
        raw_training = json.loads(request.form.get("training", "{}"))
        brush_radius = int(request.form.get("brush_radius", 8))
    except Exception as e:
        return jsonify({"error": f"Bad form data: {e}"}), 400

    try:
        img = lc.load_and_resize(f)
    except Exception as e:
        return jsonify({"error": f"Could not read image: {e}"}), 400

    img_array = np.array(img)

    training = {}
    for cid_str, points in raw_training.items():
        cid = int(cid_str)
        if not points:
            continue
        pixels = lc.sample_pixels(img_array, points, brush_radius)
        if len(pixels) >= 3:
            training[cid] = pixels

    if len(training) < 2:
        return jsonify({"error": "Need at least 2 classes with painted samples"}), 400

    # Compute initial separability to get confused pairs
    _, pairs_before, deficit_before = lc.compute_separability(training)

    # Apply Baru auto-fix: Mahalanobis outlier removal
    cleaned_training, fix_summary = lc.apply_baru_fix(training, pairs_before)

    try:
        pred_map, clf, class_ids = lc.classify_image(img, cleaned_training)
    except Exception as e:
        return jsonify({"error": f"Classification failed: {e}"}), 500

    classified_img = lc.render_classified(pred_map, class_ids)

    signatures, pairs_after, deficit_after = lc.compute_separability(cleaned_training)
    optimizer, segments = lc.build_baru_optimizer(
        cleaned_training, pairs_after, deficit_after, class_ids
    )

    baru_info = _build_baru_info(pairs_after, deficit_after, optimizer, class_ids)
    baru_info["fix_summary"]    = fix_summary
    baru_info["deficit_before"] = deficit_before
    baru_info["deficit_after"]  = deficit_after

    def img_to_b64(image):
        buf = io.BytesIO()
        image.save(buf, format="PNG")
        return base64.b64encode(buf.getvalue()).decode()

    legend = []
    for cid in class_ids:
        if cid < len(lc.CLASS_PALETTE):
            p = lc.CLASS_PALETTE[cid]
            legend.append({"id": cid, "label": p["label"], "hex": p["hex"]})

    return jsonify({
        "map":    img_to_b64(classified_img),
        "legend": legend,
        "baru":   baru_info,
    })


def _build_baru_info(pairs, total_deficit, optimizer, class_ids):
    """Build the baru analysis dict returned to the frontend."""
    pairs_out = []
    for p in pairs:
        i_label = lc.CLASS_PALETTE[p["i"]]["label"] if p["i"] < len(lc.CLASS_PALETTE) else str(p["i"])
        j_label = lc.CLASS_PALETTE[p["j"]]["label"] if p["j"] < len(lc.CLASS_PALETTE) else str(p["j"])
        pairs_out.append({
            "i": p["i"], "j": p["j"],
            "i_label": i_label, "j_label": j_label,
            "jm": p["jm"], "jm_int": p["jm_int"],
            "deficit": p["deficit"], "confused": p["confused"],
        })

    recommendations = []
    if optimizer is not None and total_deficit > 0:
        try:
            gen = optimizer.generate(4)
            recommendations = list(gen.segments)
        except Exception:
            pass

    return {
        "state":           total_deficit,
        "perfect":         total_deficit == 0,
        "pairs":           pairs_out,
        "recommendations": recommendations,
        "n_confused":      sum(1 for p in pairs if p["confused"]),
    }


if __name__ == "__main__":
    app.run(debug=True, port=5050)
