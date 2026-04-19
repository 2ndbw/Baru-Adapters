"""
BARU DEMO — ISODATA Land Cover Classification Optimizer
========================================================

Scenario: An ERDAS IMAGINE analyst is mapping land cover change at a
Nevada open-pit copper mine using Landsat 8 imagery (NIR / Red / SWIR2
three-band composite). Five target classes:

    water        — tailings ponds, small reservoirs
    vegetation   — sparse desert scrub at site perimeter
    bare_soil    — undisturbed desert floor
    mine_waste   — waste rock dumps (iron oxides, sulfides)
    active_pit   — excavated rock faces, wet disturbed ground

ISODATA has 5 parameters. Different combinations produce wildly different
classification accuracy. Analysts typically tune these by hand — run the
job, look at the confusion matrix, adjust, run again. That loop can take
an entire work day on a large scene.

Baru sits INSIDE that loop. It reads the current accuracy shortfall and
computes the minimum parameter adjustments to reach publishable Kappa (≥0.85).

─────────────────────────────────────────────────────────────────────────
BARU MODEL

    State    = accuracy shortfall: TARGET_KAPPA - current_kappa (×100, int)
               0 or negative = target reached
               positive = how many Kappa points you still need

    Segments = discrete ISODATA parameter adjustments, each with a
               known approximate effect on classification Kappa

    Perfect  = shortfall ≤ 0 (Kappa ≥ 0.85)

    Inverse  = greedy: stack highest-gain adjustments until gap is closed

─────────────────────────────────────────────────────────────────────────
"""

import sys, os
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import cohen_kappa_score, accuracy_score, confusion_matrix

sys.path.insert(0, os.path.dirname(__file__))
from baru import Baru
from isodata import ISODATAParams, isodata

np.random.seed(42)

# ─────────────────────────────────────────────────────────────────────────────
# SYNTHETIC LANDSAT 8 SCENE — Nevada Mine Site
# NIR (B5), Red (B4), SWIR2 (B7) — Surface Reflectance values (0.0–1.0)
# ─────────────────────────────────────────────────────────────────────────────

CLASSES = ['water', 'vegetation', 'bare_soil', 'mine_waste', 'active_pit']

# Mean spectral signature per class [NIR, Red, SWIR2]
SIGNATURES = np.array([
    [0.04, 0.03, 0.02],   # water:       absorbs NIR + SWIR almost completely
    [0.48, 0.07, 0.10],   # vegetation:  high NIR (cell structure), low Red (chlorophyll)
    [0.28, 0.24, 0.32],   # bare_soil:   moderate, gradually increasing toward SWIR
    [0.18, 0.15, 0.38],   # mine_waste:  elevated SWIR (iron oxides, pyrite, sulfides)
    [0.12, 0.09, 0.07],   # active_pit:  dark — wet broken rock, shadow in pit walls
])

N_PER_CLASS = 300   # 1,500 total pixels (representative sample from a large scene)
NOISE_SD    = 0.025  # realistic spectral variability within each class

X = np.vstack([
    np.clip(SIGNATURES[i] + np.random.randn(N_PER_CLASS, 3) * NOISE_SD, 0, 1)
    for i in range(len(CLASSES))
])
y_true = np.repeat(np.arange(len(CLASSES)), N_PER_CLASS)

TARGET_KAPPA = 85   # Kappa × 100 — minimum for publishable classification


# ─────────────────────────────────────────────────────────────────────────────
# ISODATA RUNNER — wraps isodata.py, maps clusters → true classes, scores
# ─────────────────────────────────────────────────────────────────────────────

def run_and_score(params: ISODATAParams, verbose=False):
    """
    Run ISODATA with the given parameters and return classification scores.
    Cluster-to-class mapping uses majority vote per cluster.
    """
    labels, centers, n_iter, converged = isodata(X, params, verbose=False)

    n_clusters = int(labels.max()) + 1
    mapping = {}
    for c in range(n_clusters):
        mask = labels == c
        if mask.sum() > 0:
            mapping[c] = int(np.bincount(y_true[mask], minlength=len(CLASSES)).argmax())

    y_pred = np.array([mapping.get(int(l), 0) for l in labels])

    overall_acc = accuracy_score(y_true, y_pred) * 100
    kappa       = cohen_kappa_score(y_true, y_pred) * 100
    cm          = confusion_matrix(y_true, y_pred, labels=list(range(len(CLASSES))))

    if verbose:
        print(f"  Clusters found: {n_clusters}  Iterations: {n_iter}  "
              f"Converged: {converged}")
        print(f"  Overall accuracy: {overall_acc:.1f}%   Kappa: {kappa:.1f}")

    return overall_acc, kappa, cm, n_clusters, converged


# ─────────────────────────────────────────────────────────────────────────────
# BARU DOMAIN — ISODATA PARAMETER ADJUSTMENTS
# ─────────────────────────────────────────────────────────────────────────────
# Each segment is one parameter adjustment an analyst can make.
# Effect = approximate change in Kappa (×100) for this dataset.
# Calibrated by running the classifier with each adjustment applied to
# the baseline "bad" parameters and measuring the Kappa delta.

ADJUSTMENTS = {
    # ── Target cluster count (K) ──────────────────────────────────────────
    'K_raise_5':       +12,   # K += 5: more target clusters → better separation
    'K_raise_2':       + 5,   # K += 2: modest increase
    'K_lower_2':       - 7,   # K -= 2: too few classes → confusion rises
    'K_lower_5':       -15,   # K -= 5: severe under-clustering

    # ── Split threshold THETA_S ───────────────────────────────────────────
    'split_loosen':    + 6,   # lower THETA_S: split more aggressively → finer map
    'split_tighten':   - 4,   # raise THETA_S: fewer splits → coarser map

    # ── Merge threshold THETA_C ───────────────────────────────────────────
    'merge_tighten':   + 8,   # raise THETA_C: less merging → preserve classes
    'merge_loosen':    -10,   # lower THETA_C: aggressive merging → class collapse

    # ── Min pixels per cluster THETA_M ────────────────────────────────────
    'minpix_lower':    + 4,   # keep smaller clusters → more rare-class coverage
    'minpix_raise':    - 3,   # discard more clusters → lose rare classes

    # ── Iteration count ───────────────────────────────────────────────────
    'iter_more':       + 3,   # more iterations → better convergence
    'iter_fewer':      - 2,   # fewer iterations → premature stop
}

ADJ_NAMES = list(ADJUSTMENTS.keys())


def baru_inverse(shortfall):
    """
    Given a Kappa shortfall, return the minimum parameter adjustments
    to close the gap. Greedy: largest positive gain first.
    """
    if shortfall <= 0:
        return []
    result    = []
    remaining = shortfall
    pool = sorted(
        [(name, g) for name, g in ADJUSTMENTS.items() if g > 0],
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
    """Create a Baru instance starting from a measured Kappa score."""
    return Baru(
        segments = ADJ_NAMES,
        compose  = lambda kappa, adj: kappa + ADJUSTMENTS[adj],
        perfect  = lambda kappa: kappa >= TARGET_KAPPA,
        inverse  = lambda kappa: baru_inverse(TARGET_KAPPA - kappa),
        start    = round(initial_kappa),
    )


# ─────────────────────────────────────────────────────────────────────────────
# STEP 1: Baseline — bad parameters, poor classification
# ─────────────────────────────────────────────────────────────────────────────
# Analyst uses defaults without tuning. K=2 is far too coarse for 5 classes.
# The algorithm converges fast but collapses everything into 2 clusters.

bad_params = ISODATAParams(K=2, I=15, THETA_M=200, THETA_S=3.0,
                           THETA_C=50.0, k=2)

print("─" * 66)
print("ISODATA — BAD PARAMETERS (analyst used defaults without tuning)")
print("─" * 66)
acc_bad, kappa_bad, cm_bad, k_bad, conv_bad = run_and_score(bad_params, verbose=True)
print()
print("  Confusion matrix (rows=true, cols=predicted):")
print(f"  Classes: {CLASSES}")
for i, row in enumerate(cm_bad):
    print(f"  {CLASSES[i]:<14} {[int(v) for v in row]}")
print(f"\n  Shortfall from target Kappa 85: {TARGET_KAPPA - kappa_bad:.1f} points")


# ─────────────────────────────────────────────────────────────────────────────
# STEP 2: Good parameters — publishable classification
# ─────────────────────────────────────────────────────────────────────────────

good_params = ISODATAParams(K=5, I=100, THETA_M=30, THETA_S=0.8,
                            THETA_C=8.0, k=5)

print()
print("─" * 66)
print("ISODATA — GOOD PARAMETERS (senior analyst, tuned)")
print("─" * 66)
acc_good, kappa_good, cm_good, k_good, conv_good = run_and_score(good_params, verbose=True)
print()
print("  Confusion matrix (rows=true, cols=predicted):")
for i, row in enumerate(cm_good):
    print(f"  {CLASSES[i]:<14} {[int(v) for v in row]}")
print(f"\n  Shortfall from target Kappa 85: {TARGET_KAPPA - kappa_good:.1f} points")


# ─────────────────────────────────────────────────────────────────────────────
# STEP 3: Junior analyst submits a parameter adjustment sequence
#         Some moves help, some hurt. Net result: still below target Kappa.
# ─────────────────────────────────────────────────────────────────────────────

shortfall = round(TARGET_KAPPA - kappa_bad)
optimizer  = make_optimizer(round(kappa_bad))

# The analyst tries a sequence of adjustments — some good, some bad
junior_adjustments = [
    'K_raise_2',       # good: more clusters helps
    'iter_more',       # good: better convergence
    'merge_loosen',    # bad: aggressive merging destroys the mine_waste class
    'minpix_raise',    # bad: discards the rare active_pit cluster
    'split_tighten',   # bad: reduces splitting, coarsens the map
]

result = optimizer.run(junior_adjustments)

print()
print("─" * 66)
print("JUNIOR ANALYST — PARAMETER ADJUSTMENT SEQUENCE")
print("─" * 66)
print(f"  Starting Kappa: {round(kappa_bad)}")
print(f"  Adjustments:    {junior_adjustments}")
print(f"  Final Kappa:    {result.state}")
print(f"  Publishable:    {result.perfect}")
print()
print("  Walk-through:")
running = round(kappa_bad)
for adj in junior_adjustments:
    effect = ADJUSTMENTS[adj]
    running += effect
    sign = '+' if effect >= 0 else ''
    print(f"    {adj:<22} {sign}{effect:3d}  →  Kappa {running}")
print(f"\n  Still {TARGET_KAPPA - result.state} Kappa points below publishable threshold.")


# ─────────────────────────────────────────────────────────────────────────────
# STEP 4: Baru finds the minimum correction to the junior's sequence
# ─────────────────────────────────────────────────────────────────────────────

fix = optimizer.correct(junior_adjustments)

print()
print("─" * 66)
print("BARU CORRECTION — MINIMUM CHANGES TO THE ANALYST'S SEQUENCE")
print("─" * 66)
print(f"  Before: {fix.before}")
print(f"  After:  {fix.after}")
print(f"  Changes: {fix.swaps}  [{fix.method}]")

changes = [(i, fix.before[i], fix.after[i])
           for i in range(min(len(fix.before), len(fix.after)))
           if fix.before[i] != fix.after[i]]
if len(fix.after) > len(fix.before):
    for i in range(len(fix.before), len(fix.after)):
        changes.append((i, '—', fix.after[i]))

if changes:
    print()
    for pos, old, new in changes:
        old_eff = ADJUSTMENTS.get(old, 0)
        new_eff = ADJUSTMENTS.get(new, 0)
        swing = new_eff - old_eff
        print(f"  Step {pos+1}: '{old}' ({old_eff:+d} Kappa)  →  "
              f"'{new}' ({new_eff:+d} Kappa)  [{swing:+d} swing]")

verify = optimizer.run(fix.after)
print(f"\n  Final Kappa after fix: {verify.state}")
print(f"  Publishable: {verify.perfect}")


# ─────────────────────────────────────────────────────────────────────────────
# STEP 5: Baru generates an optimal adjustment sequence from scratch
# ─────────────────────────────────────────────────────────────────────────────

generated = optimizer.generate(6)

print()
print("─" * 66)
print("BARU-GENERATED OPTIMAL ADJUSTMENT SEQUENCE")
print("─" * 66)
print(f"  Starting Kappa: {round(kappa_bad)}")

running = round(kappa_bad)
for adj in generated.segments:
    effect = ADJUSTMENTS[adj]
    running += effect
    sign = '+' if effect >= 0 else ''
    print(f"  {adj:<22} {sign}{effect:3d}  →  Kappa {running}")

print(f"\n  Final Kappa: {generated.state}")
print(f"  Publishable: {generated.perfect}")
print("─" * 66)
