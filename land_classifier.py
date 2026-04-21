"""
Supervised land use classifier + Baru separability optimizer.

Flow:
  1. User paints training areas on an aerial photo (frontend canvas)
  2. classify_image() extracts pixel values and trains GaussianNB
  3. compute_separability() measures JM distance between every class pair
  4. build_baru_optimizer() returns a Baru instance whose state is the
     total separability deficit — how far the training is from publishable
  5. baru_fix() removes Mahalanobis outliers from each confused class,
     re-trains, and returns the improved classification

Separability metric: Bhattacharyya coefficient → JM distance (0–2).
  JM ≥ 1.9  excellent   (classes clearly separated)
  JM ≥ 1.5  good        (classifier will work)
  JM < 1.5  poor        (classes confused — Baru flags this)

State encoding: deficit = sum of (150 - JM×100) for all pairs below threshold
  Perfect = 0 (all pairs above threshold)
"""

import numpy as np
from PIL import Image
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import cohen_kappa_score
import sys, os

sys.path.insert(0, os.path.dirname(__file__))
from baru import Baru


# ── Class palette — 100 standard aerial/satellite interpretation classes ──────
# Organized by category: Water (0-9), Natural Veg (10-24), Agriculture (25-34),
# Bare Ground (35-44), Urban/Built (45-64), Mining (65-74),
# Infrastructure (75-79), Geology/Terrain (80-89), Special (90-99)

CLASS_PALETTE = [
    # ── WATER (0–9) ──────────────────────────────────────────────────────────
    {"label": "Deep water / lake",          "color": [ 26, 111, 168], "hex": "#1a6fa8", "group": "Water"},
    {"label": "Shallow / nearshore water",  "color": [ 93, 173, 226], "hex": "#5dade2", "group": "Water"},
    {"label": "River / stream",             "color": [ 46, 134, 193], "hex": "#2e86c1", "group": "Water"},
    {"label": "Reservoir / dam",            "color": [ 31,  97, 141], "hex": "#1f618d", "group": "Water"},
    {"label": "Pond / small water body",    "color": [133, 193, 233], "hex": "#85c1e9", "group": "Water"},
    {"label": "Wetland / marsh water",      "color": [ 72, 201, 176], "hex": "#48c9b0", "group": "Water"},
    {"label": "Flooded field",              "color": [118, 215, 196], "hex": "#76d7c4", "group": "Water"},
    {"label": "Tidal flat / mudflat",       "color": [127, 179, 211], "hex": "#7fb3d3", "group": "Water"},
    {"label": "Irrigation canal / ditch",   "color": [169, 204, 227], "hex": "#a9cce3", "group": "Water"},
    {"label": "Swimming pool / artificial", "color": [174, 214, 241], "hex": "#aed6f1", "group": "Water"},
    # ── NATURAL VEGETATION (10–24) ────────────────────────────────────────────
    {"label": "Dense deciduous forest",     "color": [ 30, 132,  73], "hex": "#1e8449", "group": "Natural Veg"},
    {"label": "Dense coniferous forest",    "color": [ 20,  90,  50], "hex": "#145a32", "group": "Natural Veg"},
    {"label": "Mixed forest",               "color": [ 39, 174,  96], "hex": "#27ae60", "group": "Natural Veg"},
    {"label": "Open woodland / savanna",    "color": [ 82, 190, 128], "hex": "#52be80", "group": "Natural Veg"},
    {"label": "Shrubland / chaparral",      "color": [ 88, 214, 141], "hex": "#58d68d", "group": "Natural Veg"},
    {"label": "Grassland / prairie",        "color": [169, 223, 191], "hex": "#a9dfbf", "group": "Natural Veg"},
    {"label": "Dry scrub / desert shrub",   "color": [130, 180,  80], "hex": "#82b450", "group": "Natural Veg"},
    {"label": "Riparian vegetation",        "color": [ 14, 102,  85], "hex": "#0e6655", "group": "Natural Veg"},
    {"label": "Mangrove",                   "color": [ 17, 122, 101], "hex": "#117a65", "group": "Natural Veg"},
    {"label": "Peatland / bog",             "color": [ 26, 188, 156], "hex": "#1abc9c", "group": "Natural Veg"},
    {"label": "Alpine meadow",              "color": [118, 176,  65], "hex": "#76b041", "group": "Natural Veg"},
    {"label": "Tundra / Arctic heath",      "color": [157, 204, 101], "hex": "#9dcc65", "group": "Natural Veg"},
    {"label": "Emergent wetland veg",       "color": [  0, 180, 100], "hex": "#00b464", "group": "Natural Veg"},
    {"label": "Seagrass / aquatic veg",     "color": [  0, 184, 148], "hex": "#00b894", "group": "Natural Veg"},
    {"label": "Dead / stressed vegetation", "color": [160, 160,   0], "hex": "#a0a000", "group": "Natural Veg"},
    # ── AGRICULTURE (25–34) ───────────────────────────────────────────────────
    {"label": "Row crops (corn / soy)",     "color": [212, 230,  90], "hex": "#d4e65a", "group": "Agriculture"},
    {"label": "Orchards / fruit trees",     "color": [180, 210,  60], "hex": "#b4d23c", "group": "Agriculture"},
    {"label": "Vineyards",                  "color": [145, 185,  50], "hex": "#91b932", "group": "Agriculture"},
    {"label": "Rice paddy",                 "color": [100, 200, 130], "hex": "#64c882", "group": "Agriculture"},
    {"label": "Greenhouse / hoophouse",     "color": [240, 255, 200], "hex": "#f0ffc8", "group": "Agriculture"},
    {"label": "Hay / alfalfa field",        "color": [200, 225,  80], "hex": "#c8e150", "group": "Agriculture"},
    {"label": "Fallow / tilled field",      "color": [205, 180, 120], "hex": "#cdb478", "group": "Agriculture"},
    {"label": "Irrigated cropland",         "color": [ 80, 200, 160], "hex": "#50c8a0", "group": "Agriculture"},
    {"label": "Pasture / grazing land",     "color": [170, 215, 110], "hex": "#aad76e", "group": "Agriculture"},
    {"label": "Nursery / tree farm",        "color": [ 60, 160,  80], "hex": "#3ca050", "group": "Agriculture"},
    # ── BARE GROUND & SOIL (35–44) ────────────────────────────────────────────
    {"label": "Bare sandy soil",            "color": [234, 212, 163], "hex": "#ead4a3", "group": "Bare Ground"},
    {"label": "Bare clay / silt soil",      "color": [188, 143,  95], "hex": "#bc8f5f", "group": "Bare Ground"},
    {"label": "Bare rocky ground",          "color": [160, 140, 120], "hex": "#a08c78", "group": "Bare Ground"},
    {"label": "Gravel / cobble surface",    "color": [180, 165, 140], "hex": "#b4a58c", "group": "Bare Ground"},
    {"label": "Sand dune",                  "color": [240, 220, 140], "hex": "#f0dc8c", "group": "Bare Ground"},
    {"label": "Desert pavement / reg",      "color": [200, 175, 130], "hex": "#c8af82", "group": "Bare Ground"},
    {"label": "Exposed bedrock",            "color": [140, 120, 100], "hex": "#8c7864", "group": "Bare Ground"},
    {"label": "Eroded / gullied land",      "color": [175, 100,  60], "hex": "#af643c", "group": "Bare Ground"},
    {"label": "Beach / shoreline sand",     "color": [250, 235, 180], "hex": "#faebb4", "group": "Bare Ground"},
    {"label": "Alluvial fan / dry wash",    "color": [210, 190, 150], "hex": "#d2be96", "group": "Bare Ground"},
    # ── URBAN & BUILT (45–64) ─────────────────────────────────────────────────
    {"label": "High-density urban",         "color": [100, 100, 110], "hex": "#64646e", "group": "Urban"},
    {"label": "Low-density residential",    "color": [160, 160, 165], "hex": "#a0a0a5", "group": "Urban"},
    {"label": "Commercial / retail",        "color": [130, 130, 140], "hex": "#82828c", "group": "Urban"},
    {"label": "Industrial building",        "color": [110, 100, 100], "hex": "#6e6464", "group": "Urban"},
    {"label": "Warehouse / big box",        "color": [190, 185, 175], "hex": "#beb9af", "group": "Urban"},
    {"label": "Airport runway / tarmac",    "color": [ 60,  60,  65], "hex": "#3c3c41", "group": "Urban"},
    {"label": "Airport apron / taxiway",    "color": [ 80,  80,  85], "hex": "#505055", "group": "Urban"},
    {"label": "Railroad / rail yard",       "color": [ 50,  45,  45], "hex": "#322d2d", "group": "Urban"},
    {"label": "Paved road (major)",         "color": [ 90,  90,  95], "hex": "#5a5a5f", "group": "Urban"},
    {"label": "Unpaved road / track",       "color": [185, 155, 110], "hex": "#b99b6e", "group": "Urban"},
    {"label": "Parking lot",                "color": [120, 115, 115], "hex": "#787373", "group": "Urban"},
    {"label": "Bridge / overpass",          "color": [145, 140, 130], "hex": "#918c82", "group": "Urban"},
    {"label": "Solar panel array",          "color": [ 30,  50, 100], "hex": "#1e3264", "group": "Urban"},
    {"label": "Sports field / stadium",     "color": [ 50, 205,  50], "hex": "#32cd32", "group": "Urban"},
    {"label": "Golf course fairway",        "color": [ 70, 190,  80], "hex": "#46be50", "group": "Urban"},
    {"label": "Cemetery / memorial",        "color": [200, 220, 180], "hex": "#c8dcb4", "group": "Urban"},
    {"label": "Construction site",          "color": [210, 160,  80], "hex": "#d2a050", "group": "Urban"},
    {"label": "Rooftop — light / flat",     "color": [220, 218, 210], "hex": "#dcdad2", "group": "Urban"},
    {"label": "Rooftop — dark / pitched",   "color": [ 80,  75,  70], "hex": "#504b46", "group": "Urban"},
    {"label": "Concrete / white surface",   "color": [230, 230, 225], "hex": "#e6e6e1", "group": "Urban"},
    # ── MINING & EXTRACTION (65–74) ───────────────────────────────────────────
    {"label": "Open pit mine — active face","color": [192,  57,  43], "hex": "#c0392b", "group": "Mining"},
    {"label": "Mine waste dump",            "color": [165,  40,  30], "hex": "#a5281e", "group": "Mining"},
    {"label": "Heap leach pad",             "color": [210,  90,  50], "hex": "#d25a32", "group": "Mining"},
    {"label": "Tailings pond",              "color": [180, 100,  80], "hex": "#b46450", "group": "Mining"},
    {"label": "Quarry / gravel pit",        "color": [200, 130,  80], "hex": "#c88250", "group": "Mining"},
    {"label": "Borrow pit",                 "color": [190, 150, 100], "hex": "#be9664", "group": "Mining"},
    {"label": "Oil / gas well pad",         "color": [140,  60,  20], "hex": "#8c3c14", "group": "Mining"},
    {"label": "Pipeline corridor",          "color": [160,  80,  40], "hex": "#a05028", "group": "Mining"},
    {"label": "Mine access road",           "color": [175, 110,  60], "hex": "#af6e3c", "group": "Mining"},
    {"label": "Disturbed / reclaimed land", "color": [220, 160, 100], "hex": "#dca064", "group": "Mining"},
    # ── INFRASTRUCTURE (75–79) ────────────────────────────────────────────────
    {"label": "Power line corridor",        "color": [155,  89, 182], "hex": "#9b59b6", "group": "Infrastructure"},
    {"label": "Wind turbine / wind farm",   "color": [180, 120, 210], "hex": "#b478d2", "group": "Infrastructure"},
    {"label": "Water tower / storage tank", "color": [130,  80, 160], "hex": "#8250a0", "group": "Infrastructure"},
    {"label": "Landfill / waste disposal",  "color": [100,  60, 120], "hex": "#643c78", "group": "Infrastructure"},
    {"label": "Communication tower",        "color": [200, 150, 230], "hex": "#c896e6", "group": "Infrastructure"},
    # ── GEOLOGY & TERRAIN (80–89) ─────────────────────────────────────────────
    {"label": "Lava flow / volcanic rock",  "color": [ 30,  25,  25], "hex": "#1e1919", "group": "Geology"},
    {"label": "Salt flat / evaporite",      "color": [240, 240, 235], "hex": "#f0f0eb", "group": "Geology"},
    {"label": "Playa / dry lake bed",       "color": [215, 205, 185], "hex": "#d7cdb9", "group": "Geology"},
    {"label": "Snow / glacier ice",         "color": [245, 248, 255], "hex": "#f5f8ff", "group": "Geology"},
    {"label": "Landslide / mass movement",  "color": [150,  90,  50], "hex": "#965a32", "group": "Geology"},
    {"label": "River terrace / floodplain", "color": [185, 200, 160], "hex": "#b9c8a0", "group": "Geology"},
    {"label": "Karst / sinkhole terrain",   "color": [120, 100,  80], "hex": "#786450", "group": "Geology"},
    {"label": "Coastal cliff / escarpment", "color": [160, 130, 100], "hex": "#a08264", "group": "Geology"},
    {"label": "Braided river / gravel bar", "color": [195, 185, 160], "hex": "#c3b9a0", "group": "Geology"},
    {"label": "Cinder cone / tephra",       "color": [ 55,  40,  35], "hex": "#372823", "group": "Geology"},
    # ── SPECIAL / TRANSIENT (90–99) ───────────────────────────────────────────
    {"label": "Building shadow",            "color": [ 44,  62,  80], "hex": "#2c3e50", "group": "Special"},
    {"label": "Terrain / hill shadow",      "color": [ 30,  40,  55], "hex": "#1e2837", "group": "Special"},
    {"label": "Cloud",                      "color": [255, 255, 255], "hex": "#ffffff", "group": "Special"},
    {"label": "Fire scar / burn area",      "color": [ 40,  20,  10], "hex": "#28140a", "group": "Special"},
    {"label": "Algae bloom / turbid water", "color": [100, 180,  60], "hex": "#64b43c", "group": "Special"},
    {"label": "Aquaculture pond",           "color": [ 60, 140, 160], "hex": "#3c8ca0", "group": "Special"},
    {"label": "Boat / vessel",              "color": [255, 140,   0], "hex": "#ff8c00", "group": "Special"},
    {"label": "Vehicle / heavy equipment",  "color": [255, 200,   0], "hex": "#ffc800", "group": "Special"},
    {"label": "Flood / inundation",         "color": [ 70, 130, 180], "hex": "#4682b4", "group": "Special"},
    {"label": "Unknown / mixed",            "color": [128, 128, 128], "hex": "#808080", "group": "Special"},
]

JM_THRESHOLD = 150   # JM × 100 — pairs below this are flagged as confused


# ── Image utilities ───────────────────────────────────────────────────────────

def load_and_resize(file_obj, max_size=600):
    img = Image.open(file_obj).convert("RGB")
    img.thumbnail((max_size, max_size), Image.LANCZOS)
    return img


def sample_pixels(img_array, points, brush_radius):
    """
    Given a list of {x, y} paint points and a brush radius,
    return all unique pixel positions within the brush circles.
    img_array shape: (H, W, 3)
    """
    H, W = img_array.shape[:2]
    coords = set()
    for pt in points:
        cx, cy = int(pt["x"]), int(pt["y"])
        for dy in range(-brush_radius, brush_radius + 1):
            for dx in range(-brush_radius, brush_radius + 1):
                if dx * dx + dy * dy <= brush_radius * brush_radius:
                    nx, ny = cx + dx, cy + dy
                    if 0 <= nx < W and 0 <= ny < H:
                        coords.add((nx, ny))
    pixels = np.array([img_array[ny, nx] for nx, ny in coords], dtype=float) / 255.0
    return pixels


# ── Classification ────────────────────────────────────────────────────────────

def classify_image(img, training):
    """
    Train a Maximum Likelihood (GaussianNB) classifier and classify img.

    Parameters
    ----------
    img       : PIL.Image (RGB)
    training  : dict  {class_id (int): ndarray (n_samples, 3)}

    Returns
    -------
    pred_map   : ndarray (H, W)  — class id per pixel
    clf        : trained GaussianNB
    class_ids  : list of class ids present in training
    """
    img_array = np.array(img)

    X_train, y_train = [], []
    for cid, pixels in training.items():
        if len(pixels) < 3:
            continue
        X_train.append(pixels)
        y_train.extend([cid] * len(pixels))

    if not X_train:
        raise ValueError("No training data")

    X_train = np.vstack(X_train)
    y_train = np.array(y_train)

    clf = GaussianNB()
    clf.fit(X_train, y_train)

    H, W = img_array.shape[:2]
    all_pixels = img_array.reshape(-1, 3).astype(float) / 255.0
    predictions = clf.predict(all_pixels)
    pred_map = predictions.reshape(H, W)

    return pred_map, clf, sorted(training.keys())


def render_classified(pred_map, class_ids):
    """Turn a prediction map into an RGB PIL Image using the class palette."""
    H, W = pred_map.shape
    rgb = np.zeros((H, W, 3), dtype=np.uint8)
    for cid in class_ids:
        if cid < len(CLASS_PALETTE):
            color = CLASS_PALETTE[cid]["color"]
            mask = pred_map == cid
            rgb[mask] = color
    return Image.fromarray(rgb)


# ── Separability ─────────────────────────────────────────────────────────────

def bhattacharyya_distance(mu1, cov1, mu2, cov2):
    """
    Bhattacharyya distance between two multivariate Gaussians.
    Returns a non-negative float; higher = more separable.
    """
    eps = np.eye(len(mu1)) * 1e-6
    cov_mean = (cov1 + cov2) / 2 + eps
    diff = mu1 - mu2

    try:
        term1 = 0.125 * diff @ np.linalg.inv(cov_mean) @ diff
        sign1, logdet_mean = np.linalg.slogdet(cov_mean)
        sign2, logdet1     = np.linalg.slogdet(cov1 + eps)
        sign3, logdet2     = np.linalg.slogdet(cov2 + eps)
        term2 = 0.5 * (logdet_mean - 0.5 * (logdet1 + logdet2))
        return float(term1 + max(0, term2))
    except np.linalg.LinAlgError:
        return 0.0


def jm_distance(mu1, cov1, mu2, cov2):
    """JM distance = 2(1 − exp(−B)), range 0–2."""
    B = bhattacharyya_distance(mu1, cov1, mu2, cov2)
    return 2.0 * (1.0 - np.exp(-B))


def compute_separability(training):
    """
    Compute pairwise JM distances between all trained classes.

    Returns
    -------
    signatures : dict  {class_id: {"mean": ..., "cov": ...}}
    pairs      : list of {"i": cid_i, "j": cid_j, "jm": float,
                          "jm_int": int, "deficit": int, "confused": bool}
    total_deficit : int
    """
    signatures = {}
    for cid, pixels in training.items():
        if len(pixels) < 3:
            continue
        signatures[cid] = {
            "mean": pixels.mean(axis=0),
            "cov":  np.cov(pixels.T) if pixels.shape[1] > 1 else
                    np.array([[pixels.var()]]),
        }

    class_ids = sorted(signatures.keys())
    pairs = []
    total_deficit = 0

    for i in range(len(class_ids)):
        for j in range(i + 1, len(class_ids)):
            ci, cj = class_ids[i], class_ids[j]
            jm = jm_distance(
                signatures[ci]["mean"], signatures[ci]["cov"],
                signatures[cj]["mean"], signatures[cj]["cov"],
            )
            jm_int  = min(200, int(jm * 100))
            deficit = max(0, JM_THRESHOLD - jm_int)
            total_deficit += deficit
            pairs.append({
                "i": ci, "j": cj,
                "jm": round(jm, 3),
                "jm_int": jm_int,
                "deficit": deficit,
                "confused": deficit > 0,
            })

    pairs.sort(key=lambda p: p["jm_int"])
    return signatures, pairs, total_deficit


# ── Baru adapter ─────────────────────────────────────────────────────────────

def build_baru_optimizer(training, pairs, total_deficit, class_ids):
    """
    Build a Baru instance for the separability optimization problem.

    Segments are training adjustments:
      add_<class>         — add more samples to a class (reduces overlap)
      clean_<class>       — remove Mahalanobis outliers (tightens signature)
      merge_<i>_<j>       — merge a confused pair into one class

    Effect of each segment: estimated Kappa-equivalent gain.
    """
    if total_deficit == 0:
        return None, 0

    segments = {}

    # Per-class operations
    for cid in class_ids:
        name = CLASS_PALETTE[cid]["label"].replace(" ", "_").replace("/", "-").lower()
        # How much deficit involves this class?
        class_deficit = sum(
            p["deficit"] for p in pairs if (p["i"] == cid or p["j"] == cid)
        )
        if class_deficit > 0:
            segments[f"add_{name}"]   = min(class_deficit, 40)
            segments[f"clean_{name}"] = min(class_deficit, 30)

    # Per-confused-pair merge operations
    for p in pairs:
        if p["confused"]:
            n_i = CLASS_PALETTE[p["i"]]["label"].replace(" ", "_").replace("/", "-").lower()
            n_j = CLASS_PALETTE[p["j"]]["label"].replace(" ", "_").replace("/", "-").lower()
            segments[f"merge_{n_i}_{n_j}"] = p["deficit"]

    if not segments:
        return None, 0

    def baru_inverse(state):
        if state <= 0:
            return []
        pool = sorted(
            [(n, g) for n, g in segments.items() if g > 0],
            key=lambda x: x[1], reverse=True,
        )
        result, remaining = [], state
        while remaining > 0:
            for name, gain in pool:
                if gain <= remaining:
                    result.append(name)
                    remaining -= gain
                    break
            else:
                result.append(pool[-1][0])
                remaining -= pool[-1][1]
        return result

    optimizer = Baru(
        segments = list(segments.keys()),
        compose  = lambda s, adj: s - segments.get(adj, 0),
        perfect  = lambda s: s <= 0,
        inverse  = baru_inverse,
        start    = total_deficit,
    )
    return optimizer, segments


# ── Baru auto-fix ─────────────────────────────────────────────────────────────

def mahalanobis_filter(pixels, threshold=2.5):
    """
    Remove pixels more than `threshold` Mahalanobis distances from the
    class centroid. Returns the cleaned pixel array.
    """
    if len(pixels) < 6:
        return pixels
    mu  = pixels.mean(axis=0)
    cov = np.cov(pixels.T) + np.eye(pixels.shape[1]) * 1e-6
    try:
        inv_cov = np.linalg.inv(cov)
    except np.linalg.LinAlgError:
        return pixels
    diff = pixels - mu
    m_dist = np.sqrt(np.einsum("ij,jk,ik->i", diff, inv_cov, diff))
    keep = m_dist <= threshold
    cleaned = pixels[keep]
    return cleaned if len(cleaned) >= 3 else pixels


def apply_baru_fix(training, pairs):
    """
    Apply the auto-fix: Mahalanobis outlier removal on confused classes.
    Returns the cleaned training dict and a summary of what changed.
    """
    confused_classes = set()
    for p in pairs:
        if p["confused"]:
            confused_classes.add(p["i"])
            confused_classes.add(p["j"])

    cleaned = {}
    summary = []
    for cid, pixels in training.items():
        if cid in confused_classes:
            before = len(pixels)
            cleaned_px = mahalanobis_filter(pixels)
            after  = len(cleaned_px)
            cleaned[cid] = cleaned_px
            if before != after:
                label = CLASS_PALETTE[cid]["label"]
                summary.append(
                    f"  {label}: removed {before - after} outlier pixels "
                    f"({before} → {after})"
                )
        else:
            cleaned[cid] = pixels

    return cleaned, summary
