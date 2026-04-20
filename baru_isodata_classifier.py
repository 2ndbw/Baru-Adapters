"""
BARU DEMO — ISODATA Land Cover Classification Optimizer
========================================================

Scenario: Mapping land cover change at a Nevada open-pit copper mine
using a full Landsat 8 surface-reflectance composite.

9 spectral bands + 3 derived indices = 12 features per pixel:

    B1  Coastal/Aerosol  0.43–0.45 μm   coastal haze, aerosol depth
    B2  Blue             0.45–0.51 μm   water penetration, soil/veg separation
    B3  Green            0.53–0.59 μm   vegetation green peak
    B4  Red              0.64–0.67 μm   chlorophyll absorption, soil contrast
    B5  NIR              0.85–0.88 μm   biomass, vegetation structure
    B6  SWIR1            1.57–1.65 μm   soil moisture, mineral discrimination
    B7  SWIR2            2.11–2.29 μm   iron oxides, sulfides, rock alteration
    B10 TIRS1 (thermal)  10.6–11.2 μm   land surface temperature
    B11 TIRS2 (thermal)  11.5–12.5 μm   temperature (validation band)

    NDVI  = (B5−B4)/(B5+B4)  vegetation index
    NDWI  = (B3−B5)/(B3+B5)  water index
    NDBI  = (B6−B5)/(B6+B5)  built-up / bare rock index

Five land cover classes for the mine site:

    water       — tailings ponds, small reservoirs (dark, absorbs NIR/SWIR)
    vegetation  — desert scrub at site perimeter (high NIR spike)
    bare_soil   — undisturbed desert floor (rising SWIR slope)
    mine_waste  — waste rock dumps (very high SWIR2: iron oxides, pyrite)
    active_pit  — excavated faces + wet broken rock (dark + warm thermal)

BARU MODEL
    State    = Kappa shortfall from publishable threshold (TARGET=85)
    Segments = discrete ISODATA parameter adjustments
    Perfect  = shortfall ≤ 0  (Kappa ≥ 85)
    Inverse  = greedy: highest-gain adjustments until gap closes
"""

import sys, os
import numpy as np
from sklearn.metrics import cohen_kappa_score, accuracy_score, confusion_matrix

sys.path.insert(0, os.path.dirname(__file__))
from baru import Baru
from isodata import ISODATAParams, isodata

np.random.seed(42)

# ─────────────────────────────────────────────────────────────────────────────
# SPECTRAL LIBRARY — Landsat 8 Surface Reflectance (0.0–1.0)
# Columns: B1  B2   B3   B4   B5   B6   B7   B10  B11
# Thermal bands normalized: (T_K − 270) / 60
# ─────────────────────────────────────────────────────────────────────────────

BAND_NAMES = ['B1', 'B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B10', 'B11']
CLASSES    = ['water', 'vegetation', 'bare_soil', 'mine_waste', 'active_pit']

SIGNATURES = np.array([
    #  B1     B2     B3     B4     B5     B6     B7    B10   B11
    [0.055, 0.052, 0.054, 0.028, 0.038, 0.018, 0.010, 0.33, 0.30],  # water
    [0.042, 0.048, 0.072, 0.050, 0.485, 0.178, 0.098, 0.42, 0.40],  # vegetation
    [0.101, 0.118, 0.182, 0.248, 0.302, 0.352, 0.318, 0.63, 0.61],  # bare_soil
    [0.082, 0.096, 0.140, 0.180, 0.198, 0.295, 0.402, 0.67, 0.65],  # mine_waste
    [0.062, 0.072, 0.092, 0.098, 0.118, 0.118, 0.095, 0.58, 0.56],  # active_pit
])

# Per-band noise (realistic inter-class spectral variability)
NOISE = np.array([0.008, 0.008, 0.010, 0.012, 0.025, 0.022, 0.020, 0.020, 0.020])

N_PER_CLASS = 400   # 2,000 total pixels

def add_indices(X_bands):
    """Compute NDVI, NDWI, NDBI and append as additional features."""
    B3  = X_bands[:, 2]
    B4  = X_bands[:, 3]
    B5  = X_bands[:, 4]
    B6  = X_bands[:, 5]
    eps = 1e-9
    NDVI = (B5 - B4) / (B5 + B4 + eps)
    NDWI = (B3 - B5) / (B3 + B5 + eps)
    NDBI = (B6 - B5) / (B6 + B5 + eps)
    return np.column_stack([X_bands, NDVI, NDWI, NDBI])

def generate_scene():
    X_bands, y = [], []
    for cls_id, sig in enumerate(SIGNATURES):
        samples = sig + np.random.randn(N_PER_CLASS, 9) * NOISE
        X_bands.append(np.clip(samples, 0, 1))
        y.extend([cls_id] * N_PER_CLASS)
    return add_indices(np.vstack(X_bands)), np.array(y)

X, y_true = generate_scene()
FEATURE_NAMES = BAND_NAMES + ['NDVI', 'NDWI', 'NDBI']

print(f"  Scene: {X.shape[0]} pixels × {X.shape[1]} features "
      f"({len(BAND_NAMES)} bands + 3 indices)")

TARGET_KAPPA = 85


# ─────────────────────────────────────────────────────────────────────────────
# RUNNER — ISODATA on full 12-feature spectral space
# ─────────────────────────────────────────────────────────────────────────────

def run_and_score(params: ISODATAParams, verbose=False):
    labels, centers, n_iter, converged = isodata(X, params, verbose=False)

    n_clusters = int(labels.max()) + 1
    mapping = {}
    for c in range(n_clusters):
        mask = labels == c
        if mask.sum() > 0:
            mapping[c] = int(np.bincount(y_true[mask], minlength=len(CLASSES)).argmax())

    y_pred       = np.array([mapping.get(int(l), 0) for l in labels])
    overall_acc  = accuracy_score(y_true, y_pred) * 100
    kappa        = cohen_kappa_score(y_true, y_pred) * 100
    cm           = confusion_matrix(y_true, y_pred, labels=list(range(len(CLASSES))))

    if verbose:
        print(f"  Clusters found: {n_clusters}  Iterations: {n_iter}  "
              f"Converged: {converged}")
        print(f"  Overall accuracy: {overall_acc:.1f}%   Kappa: {kappa:.1f}")

    return overall_acc, kappa, cm, n_clusters, converged


def print_confusion(cm):
    header = f"  {'':14}" + "".join(f"{c[:6]:>9}" for c in CLASSES)
    print(header)
    for i, row in enumerate(cm):
        print(f"  {CLASSES[i]:<14}" + "".join(f"{int(v):>9}" for v in row))


# ─────────────────────────────────────────────────────────────────────────────
# BARU DOMAIN — ISODATA PARAMETER ADJUSTMENTS
# ─────────────────────────────────────────────────────────────────────────────

ADJUSTMENTS = {
    # Target cluster count
    'K_raise_5':       +14,
    'K_raise_2':       + 6,
    'K_lower_2':       - 8,
    'K_lower_5':       -18,

    # Split threshold THETA_S (lower = more splitting = more classes)
    'split_loosen':    + 8,   # lower THETA_S: splits happen more easily
    'split_tighten':   - 5,   # raise THETA_S: fewer splits

    # Merge distance THETA_C (higher = less merging = preserve classes)
    'merge_tighten':   +10,   # raise THETA_C: stop merging similar classes
    'merge_loosen':    -12,   # lower THETA_C: aggressive merging → class collapse

    # Min cluster size THETA_M
    'minpix_lower':    + 5,   # keep small clusters → rare classes survive
    'minpix_raise':    - 4,   # discard small clusters → rare classes lost

    # Iterations
    'iter_more':       + 4,
    'iter_fewer':      - 3,
}

ADJ_NAMES = list(ADJUSTMENTS.keys())


def baru_inverse(shortfall):
    if shortfall <= 0:
        return []
    result    = []
    remaining = shortfall
    pool = sorted(
        [(n, g) for n, g in ADJUSTMENTS.items() if g > 0],
        key=lambda x: x[1], reverse=True
    )
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


def make_optimizer(initial_kappa):
    return Baru(
        segments = ADJ_NAMES,
        compose  = lambda kappa, adj: kappa + ADJUSTMENTS[adj],
        perfect  = lambda kappa: kappa >= TARGET_KAPPA,
        inverse  = lambda kappa: baru_inverse(TARGET_KAPPA - kappa),
        start    = round(initial_kappa),
    )


# ─────────────────────────────────────────────────────────────────────────────
# STEP 1: Bad parameters — analyst used defaults on 12-band data
# ─────────────────────────────────────────────────────────────────────────────

bad_params = ISODATAParams(K=2, I=10, THETA_M=400, THETA_S=5.0, THETA_C=0.02, k=2)

print()
print("─" * 68)
print("ISODATA — BAD PARAMETERS  (K=2, 10 iterations, high discard threshold)")
print("─" * 68)
acc_bad, kappa_bad, cm_bad, k_bad, _ = run_and_score(bad_params, verbose=True)
print()
print_confusion(cm_bad)
print(f"\n  Shortfall from Kappa 85: {TARGET_KAPPA - kappa_bad:.1f} points")

# ─────────────────────────────────────────────────────────────────────────────
# STEP 2: Good parameters — properly tuned for 12 features / 5 classes
# ─────────────────────────────────────────────────────────────────────────────

good_params = ISODATAParams(K=5, I=100, THETA_M=40, THETA_S=0.04, THETA_C=0.35, k=7)

print()
print("─" * 68)
print("ISODATA — GOOD PARAMETERS  (K=5, 100 iterations, tuned thresholds)")
print("─" * 68)
acc_good, kappa_good, cm_good, k_good, _ = run_and_score(good_params, verbose=True)
print()
print_confusion(cm_good)
print(f"\n  Shortfall from Kappa 85: {TARGET_KAPPA - kappa_good:.1f} points")

# ─────────────────────────────────────────────────────────────────────────────
# STEP 3: Junior analyst — bad adjustment sequence
# ─────────────────────────────────────────────────────────────────────────────

optimizer = make_optimizer(round(kappa_bad))

junior_seq = [
    'K_raise_2',       # small K increase — not enough
    'iter_more',       # more iterations — helps a little
    'merge_loosen',    # WRONG: aggressive merging destroys mine_waste class
    'minpix_raise',    # WRONG: discards the rare active_pit cluster
    'split_tighten',   # WRONG: reduces splits, map becomes coarser
]

result_jr = optimizer.run(junior_seq)

print()
print("─" * 68)
print("JUNIOR ANALYST — ADJUSTMENT SEQUENCE ON 12-BAND DATA")
print("─" * 68)
print(f"  Starting Kappa: {round(kappa_bad)}")
print()
running = round(kappa_bad)
for adj in junior_seq:
    effect = ADJUSTMENTS[adj]
    running += effect
    sign = '+' if effect >= 0 else ''
    tag = '  ✓' if effect > 0 else '  ✗ HURTS'
    print(f"  {adj:<22} {sign}{effect:3d}  →  Kappa {running:3d}{tag}")
print(f"\n  Final Kappa: {result_jr.state}   Publishable: {result_jr.perfect}")
print(f"  Still {TARGET_KAPPA - result_jr.state} Kappa points below threshold.")

# ─────────────────────────────────────────────────────────────────────────────
# STEP 4: Baru correction
# ─────────────────────────────────────────────────────────────────────────────

fix = optimizer.correct(junior_seq)

print()
print("─" * 68)
print("BARU CORRECTION — MINIMUM CHANGES TO REACH KAPPA 85")
print("─" * 68)
print(f"  Changes: {fix.swaps}   Method: {fix.method}")
print()

changes = [(i, fix.before[i], fix.after[i])
           for i in range(min(len(fix.before), len(fix.after)))
           if fix.before[i] != fix.after[i]]
if len(fix.after) > len(fix.before):
    for i in range(len(fix.before), len(fix.after)):
        changes.append((i, '—', fix.after[i]))

for pos, old, new in changes:
    old_eff = ADJUSTMENTS.get(old, 0)
    new_eff = ADJUSTMENTS.get(new, 0)
    swing   = new_eff - old_eff
    print(f"  Step {pos+1}: '{old}' ({old_eff:+d})  →  '{new}' ({new_eff:+d})  "
          f"[{swing:+d} Kappa swing]")

verify = optimizer.run(fix.after)
print(f"\n  Final Kappa: {verify.state}   Publishable: {verify.perfect}")

# ─────────────────────────────────────────────────────────────────────────────
# STEP 5: Feature importance — which bands separate the classes best?
# ─────────────────────────────────────────────────────────────────────────────

print()
print("─" * 68)
print("SPECTRAL SEPARABILITY — band-by-band class contrast")
print("─" * 68)
print(f"  {'Feature':<8}  " +
      "  ".join(f"{c[:5]:>7}" for c in CLASSES))

# For each feature, show mean value per class
for fi, fname in enumerate(FEATURE_NAMES):
    vals = [X[y_true == ci, fi].mean() for ci in range(len(CLASSES))]
    spread = max(vals) - min(vals)
    bar = "█" * int(spread * 40)
    row = f"  {fname:<8}  " + "  ".join(f"{v:7.3f}" for v in vals)
    print(row + f"   spread={spread:.3f} {bar}")

# ─────────────────────────────────────────────────────────────────────────────
# STEP 6: Generate optimal adjustment sequence
# ─────────────────────────────────────────────────────────────────────────────

generated = optimizer.generate(6)

print()
print("─" * 68)
print("BARU-GENERATED OPTIMAL PARAMETER SEQUENCE")
print("─" * 68)
print(f"  Starting Kappa: {round(kappa_bad)}")
running = round(kappa_bad)
for adj in generated.segments:
    effect = ADJUSTMENTS[adj]
    running += effect
    sign = '+' if effect >= 0 else ''
    print(f"  {adj:<22} {sign}{effect:3d}  →  Kappa {running:3d}")
print(f"\n  Final Kappa: {generated.state}   Publishable: {generated.perfect}")
print("─" * 68)
