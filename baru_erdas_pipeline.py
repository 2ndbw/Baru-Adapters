"""
BARU DEMO — ERDAS IMAGINE: Satellite Image Processing Pipeline
==============================================================

Scenario: A geologist at a mining company needs to produce a publishable
land cover change-detection map from Landsat 8 imagery of an open-pit
mine site in Nevada.

The analysis pipeline runs inside ERDAS IMAGINE. It is a sequence of
processing steps. The pipeline is "perfect" when all 8 required steps
have been applied — the image is analysis-ready and the classification
results are publishable (confusion matrix + Kappa coefficient on file).

Two junior analysts submit processing jobs. Each has a broken pipeline.
Baru finds the minimum fix for each with no domain knowledge beyond
the four adapter functions.

─────────────────────────────────────────────────────────────────────────
ERDAS IMAGINE LANDSAT 8 PIPELINE — STEP REFERENCE

  1. layer_stack             Combine individual band TIFs (B1–B11) into one
                             multi-layer file in spectral order. All downstream
                             steps require this. No quality gain — prerequisite.

  2. radiometric_calibration Convert raw Digital Numbers (DN) to Top-of-
                             Atmosphere (ToA) Reflectance using the rescaling
                             coefficients in the MTL metadata file.

  3. atmospheric_correction  Apply ATCOR to remove atmospheric absorption and
                             scattering, converting ToA → Surface Reflectance.
                             Without this, NDVI values are wrong; cross-date
                             change detection produces false positives.

  4. georeferencing          Assign geographic coordinates. Validate with
                             Ground Control Points (GCPs); RMS error must be
                             < 0.5 pixels or all spatial analysis is invalid.

  5. orthorectification      Apply RPC coefficients + DEM to remove terrain
                             distortions. A 1000m peak without orthorectification
                             can be displaced 3+ pixels (~90m). Required for
                             accurate overlay with GIS layers.

  6. band_combination        Select the optimal band composite for the target
                             feature. For mine site mapping: bands 7-5-3 (SWIR2-
                             NIR-Green) highlights bare rock and disturbed soil.
                             Generate NDVI for vegetation boundary detection.

  7. isodata_classification  Run ISODATA unsupervised classification. Standard
                             params for mine mapping: InitialClusters=20,
                             MinPixels=200, MaxIterations=30, Convergence=0.95.
                             Output: preliminary land cover classes.

  8. accuracy_assessment     Build confusion matrix from independent validation
                             points. Report producer's accuracy, user's accuracy,
                             and Kappa coefficient. Required for publication.
                             Without this, results cannot be peer-reviewed.

─────────────────────────────────────────────────────────────────────────

State:  bitmask (integer) — each bit marks one completed step
Home:   all 8 required bits set (state & REQUIRED_MASK) == REQUIRED_MASK
"""

import sys, os
sys.path.insert(0, os.path.dirname(__file__))
from baru import Baru

# ─────────────────────────────────────────────────────────────────────────────
# STEP 1: Define the domain
# ─────────────────────────────────────────────────────────────────────────────

# Each ERDAS processing step maps to a unique bit.
# Applying a step ORs its bit into the state (idempotent: apply twice = same result).

STEP_BIT = {
    'layer_stack':              1 << 0,   #   1  — prerequisite for all processing
    'radiometric_calibration':  1 << 1,   #   2  — DN → ToA Reflectance
    'atmospheric_correction':   1 << 2,   #   4  — ToA → Surface Reflectance (ATCOR)
    'georeferencing':           1 << 3,   #   8  — GCP-based spatial alignment
    'orthorectification':       1 << 4,   #  16  — terrain distortion removal (DEM + RPC)
    'band_combination':         1 << 5,   #  32  — SWIR/NIR composite + NDVI generation
    'isodata_classification':   1 << 6,   #  64  — unsupervised land cover clustering
    'accuracy_assessment':      1 << 7,   # 128  — confusion matrix + Kappa (publishable)
    'noise_filter':             1 << 8,   # 256  — optional: median filter to remove speckle
    'pan_sharpening':           1 << 9,   # 512  — optional: 30m → 15m resolution enhancement
}

STEP_NAMES = list(STEP_BIT.keys())

# These 8 steps are required for a publishable classification map.
REQUIRED_STEPS = [
    'layer_stack',
    'radiometric_calibration',
    'atmospheric_correction',
    'georeferencing',
    'orthorectification',
    'band_combination',
    'isodata_classification',
    'accuracy_assessment',
]

REQUIRED_MASK = 0
for step in REQUIRED_STEPS:
    REQUIRED_MASK |= STEP_BIT[step]
# REQUIRED_MASK = 1+2+4+8+16+32+64+128 = 255

def erdas_inverse(state):
    """
    Given the current pipeline state, return the list of required steps
    that have not yet been applied. Baru appends these to close the pipeline.
    """
    return [step for step in REQUIRED_STEPS if not (state & STEP_BIT[step])]

erdas = Baru(
    segments = STEP_NAMES,
    compose  = lambda state, step: state | STEP_BIT[step],
    perfect  = lambda state: (state & REQUIRED_MASK) == REQUIRED_MASK,
    inverse  = erdas_inverse,
    start    = 0,
)

def fmt_state(state):
    done    = [s for s in REQUIRED_STEPS if state & STEP_BIT[s]]
    missing = [s for s in REQUIRED_STEPS if not (state & STEP_BIT[s])]
    return done, missing

# ─────────────────────────────────────────────────────────────────────────────
# STEP 2: Senior analyst — complete publishable pipeline
# ─────────────────────────────────────────────────────────────────────────────

senior_pipeline = [
    'layer_stack',
    'radiometric_calibration',
    'atmospheric_correction',
    'georeferencing',
    'orthorectification',
    'band_combination',
    'noise_filter',              # optional — removes sensor speckle
    'isodata_classification',
    'accuracy_assessment',
]

result = erdas.run(senior_pipeline)
done, missing = fmt_state(result.state)

print("─" * 65)
print("SENIOR ANALYST — COMPLETE PIPELINE (Landsat 8, Nevada Mine Site)")
print("─" * 65)
print(f"  Steps submitted:  {len(senior_pipeline)}")
print(f"  Required done:    {len(done)}/8")
print(f"  Missing:          {'none' if not missing else missing}")
print(f"  Publishable:      {result.perfect}")
print()
for step in senior_pipeline:
    tag = " (required)" if step in REQUIRED_STEPS else " (optional)"
    print(f"  ✓  {step:<32}{tag}")

# ─────────────────────────────────────────────────────────────────────────────
# STEP 3: Junior analyst #1 — skipped atmospheric correction
# ─────────────────────────────────────────────────────────────────────────────
# The most common ERDAS mistake. The imagery looks plausible —
# it still classifies — but all spectral values are ToA Radiance,
# not Surface Reflectance. NDVI scores are systematically wrong.
# Cross-date change detection will show false changes where none exist.

junior1_pipeline = [
    'layer_stack',
    'radiometric_calibration',
    # atmospheric_correction MISSING
    'georeferencing',
    'orthorectification',
    'band_combination',
    'isodata_classification',
    'accuracy_assessment',
]

result1 = erdas.run(junior1_pipeline)
done1, missing1 = fmt_state(result1.state)

print()
print("─" * 65)
print("JUNIOR ANALYST #1 — MISSING ATMOSPHERIC CORRECTION")
print("─" * 65)
print(f'  "The imagery looks fine. Atmospheric correction takes 45 minutes."')
print()
print(f"  Steps submitted:  {len(junior1_pipeline)}")
print(f"  Required done:    {len(done1)}/8")
print(f"  Missing:          {missing1}")
print(f"  Publishable:      {result1.perfect}")
print()
print(f"  Real impact: ISODATA ran on ToA Radiance values, not Surface")
print(f"  Reflectance. Band 5 (NIR) and Band 7 (SWIR2) values inflated")
print(f"  by ~15-20%. Mine waste rock classified as vegetation. The")
print(f"  accuracy assessment passed because validation also used the")
print(f"  same bad imagery. Silent failure.")

fix1 = erdas.correct(junior1_pipeline)

print()
print(f"  Baru fix: {fix1.swaps} change(s)  [{fix1.method}]")
added1 = [s for s in fix1.after if s not in fix1.before]
print(f"  Appended: {added1}")

verify1 = erdas.run(fix1.after)
print(f"  Publishable after fix: {verify1.perfect}")

# ─────────────────────────────────────────────────────────────────────────────
# STEP 4: Junior analyst #2 — duplicate optional step, missing accuracy
# ─────────────────────────────────────────────────────────────────────────────
# Pan-sharpening (optional) was accidentally run twice.
# The second run is completely wasted — bitmask OR is idempotent.
# Meanwhile, accuracy_assessment was never submitted.
# The classification exists but cannot be published: no confusion matrix.

junior2_pipeline = [
    'layer_stack',
    'radiometric_calibration',
    'atmospheric_correction',
    'georeferencing',
    'orthorectification',
    'pan_sharpening',            # optional — ok
    'band_combination',
    'isodata_classification',
    'pan_sharpening',            # DUPLICATE — wasted slot (bit already set, no effect)
    # accuracy_assessment MISSING
]

result2 = erdas.run(junior2_pipeline)
done2, missing2 = fmt_state(result2.state)

print()
print("─" * 65)
print("JUNIOR ANALYST #2 — DUPLICATE STEP, MISSING ACCURACY ASSESSMENT")
print("─" * 65)
print(f'  "I accidentally queued pan_sharpening twice. Forgot accuracy."')
print()
print(f"  Steps submitted:  {len(junior2_pipeline)}")
print(f"  Required done:    {len(done2)}/8")
print(f"  Missing:          {missing2}")
print(f"  Publishable:      {result2.perfect}")
print()
print(f"  Real impact: Classification map exists and looks correct.")
print(f"  But without a confusion matrix, the mine's environmental")
print(f"  compliance report cannot cite it. Job must be resubmitted.")

fix2 = erdas.correct(junior2_pipeline)

print()
print(f"  Baru fix: {fix2.swaps} change(s)  [{fix2.method}]")

changes2 = [(i, fix2.before[i], fix2.after[i])
            for i in range(min(len(fix2.before), len(fix2.after)))
            if fix2.before[i] != fix2.after[i]]
if changes2:
    print()
    for pos, old, new in changes2:
        print(f"  Position {pos+1}: '{old}'  →  '{new}'")
    print(f"  Baru swapped the wasted duplicate for the missing required step.")

verify2 = erdas.run(fix2.after)
print(f"\n  Publishable after fix: {verify2.perfect}")

# ─────────────────────────────────────────────────────────────────────────────
# STEP 5: Generate a complete pipeline from scratch
# ─────────────────────────────────────────────────────────────────────────────

generated = erdas.generate(11)
done_gen, _ = fmt_state(generated.state)

print()
print("─" * 65)
print("GENERATED COMPLETE PIPELINE (Baru builds it from scratch)")
print("─" * 65)
print(f"  Steps: {len(generated.segments)}")
print(f"  Required steps covered: {len(done_gen)}/8")
print(f"  Publishable: {generated.perfect}")
print()
for step in generated.segments:
    tag = " (required)" if step in REQUIRED_STEPS else " (optional)"
    print(f"  ✓  {step:<32}{tag}")
print("─" * 65)
