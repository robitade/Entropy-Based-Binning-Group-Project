"""
validation_report.py  —  Student 3, Task 3.3
=============================================
Validates the programmatic entropy-based binning output for the Age
attribute against Student 2's hand-computed boundaries.

This script:
  1. Runs the algorithm programmatically on Age.
  2. Loads Student 2's manual boundaries (edit MANUAL_BOUNDARIES below).
  3. Compares the two sets of boundaries step by step.
  4. Prints a formatted validation report suitable for the PDF appendix.

Usage
-----
    python validation_report.py

Edit the MANUAL_BOUNDARIES list with the actual values from Student 2's
worksheet before running.
"""

import pandas as pd
import numpy as np
import math
from entropy_binning import (
    entropy_bin, compute_entropy, compute_information_gain,
    find_best_split, mdlp_criterion, get_candidate_splits
)

# ─────────────────────────────────────────────────────────────────────────────
#  CONFIGURATION  — Fill in Student 2's hand-computed values here
# ─────────────────────────────────────────────────────────────────────────────

# Replace these with the actual boundaries Student 2 found manually.
# Example: if S2 found splits at Age 39.5 and 46.5, write [39.5, 46.5]
MANUAL_BOUNDARIES = [39.5, 46.5]   # <-- update with S2's real values

ATTR  = "Age"                        
LABEL = "Risk_Level"
TOLERANCE = 0.01   # boundaries within ±0.01 are considered matching


# ─────────────────────────────────────────────────────────────────────────────
#  Dataset (inline — replace with CSV load if available)
# ─────────────────────────────────────────────────────────────────────────────

records = [
    (1,  22, 18.5,  72,  "Low"),    (2,  24, 20.1,  80,  "Low"),
    (3,  26, 21.4,  85,  "Low"),    (4,  28, 24.2, 110,  "Medium"),
    (5,  30, 22.8,  95,  "Low"),    (6,  31, 26.5, 130,  "Medium"),
    (7,  33, 23.0, 100,  "Low"),    (8,  35, 27.8, 140,  "Medium"),
    (9,  36, 25.5, 115,  "Low"),    (10, 38, 28.3, 150,  "High"),
    (11, 39, 24.7, 125,  "Medium"), (12, 41, 29.1, 158,  "High"),
    (13, 42, 26.2, 120,  "Medium"), (14, 43, 30.5, 165,  "High"),
    (15, 45, 25.0, 110,  "Medium"), (16, 46, 27.3, 145,  "Medium"),
    (17, 48, 31.0, 155,  "High"),   (18, 49, 28.5, 130,  "Medium"),
    (19, 51, 33.4, 170,  "High"),   (20, 53, 29.6, 148,  "Medium"),
    (21, 55, 34.1, 182,  "High"),   (22, 57, 30.7, 160,  "High"),
    (23, 59, 32.2, 175,  "High"),   (24, 61, 35.5, 190,  "High"),
    (25, 63, 33.8, 185,  "High"),   (26, 65, 36.0, 195,  "High"),
    (27, 67, 31.5, 165,  "High"),   (28, 70, 37.2, 200,  "High"),
    (29, 72, 38.0, 205,  "High"),   (30, 75, 42.0, 210,  "High"),
]
df = pd.DataFrame(records,
                  columns=["PatientID", "Age", "BMI", "Glucose", LABEL])
df = df.sort_values(ATTR).reset_index(drop=True)


# ─────────────────────────────────────────────────────────────────────────────
#  SECTION 1 — Overall Dataset Entropy (Sanity Check)
# ─────────────────────────────────────────────────────────────────────────────

print("=" * 70)
print("  VALIDATION REPORT  —  Entropy-Based Binning vs Manual Calculations")
print(f"  Attribute : {ATTR}")
print("=" * 70)

h_overall = compute_entropy(df[LABEL])
print(f"\n[1] Overall Dataset Entropy H(S)")
print(f"    Programmatic : {h_overall:.4f} bits")
print(f"    Brief ref.   : 1.5135 bits")
match = abs(h_overall - 1.5135) < 0.001
print(f"    Match        : {'✓ YES' if match else '✗ NO — check compute_entropy()'}")


# ─────────────────────────────────────────────────────────────────────────────
#  SECTION 2 — Candidate Split IG Table for Age (first 10 + best)
# ─────────────────────────────────────────────────────────────────────────────

print(f"\n[2] Information Gain for Candidate Split Points (Age)")
print(f"    {'Candidate t':>12}  {'nL':>4}  {'nR':>4}  {'H(SL)':>7}  {'H(SR)':>7}  {'IG(t)':>8}")
print(f"    {'─'*12}  {'─'*4}  {'─'*4}  {'─'*7}  {'─'*7}  {'─'*8}")

candidates = get_candidate_splits(df[ATTR])
ig_table   = []

for t in candidates:
    left   = df[df[ATTR] <= t]
    right  = df[df[ATTR] >  t]
    n_l, n_r = len(left), len(right)
    h_l    = compute_entropy(left[LABEL])
    h_r    = compute_entropy(right[LABEL])
    ig     = compute_information_gain(df, t, ATTR, LABEL)
    ig_table.append((t, n_l, n_r, h_l, h_r, ig))

# Print first 5 and last 5 candidates to keep output concise
to_print = ig_table[:5] + [None] + ig_table[-5:]
for row in to_print:
    if row is None:
        print(f"    {'  ... (middle rows omitted)':>55}")
        continue
    t, n_l, n_r, h_l, h_r, ig = row
    print(f"    {t:>12.1f}  {n_l:>4}  {n_r:>4}  {h_l:>7.4f}  {h_r:>7.4f}  {ig:>8.4f}")

best_t, best_ig = find_best_split(df, ATTR, LABEL)
print(f"\n    ── Best split t* = {best_t}  (IG = {best_ig:.4f} bits)")
print(f"    ── Expected range from brief: Age 38–42")
if 38 <= best_t <= 43:
    print(f"    ✓ t* = {best_t} is within the expected range.")
else:
    print(f"    ✗ t* = {best_t} is outside expected range — investigate.")


# ─────────────────────────────────────────────────────────────────────────────
#  SECTION 3 — MDLP Test at Best Split
# ─────────────────────────────────────────────────────────────────────────────

print(f"\n[3] MDLP Stopping Criterion at t* = {best_t}")

n        = len(df)
left_df  = df[df[ATTR] <= best_t]
right_df = df[df[ATTR] >  best_t]
k        = df[LABEL].nunique()
k_l      = left_df[LABEL].nunique()
k_r      = right_df[LABEL].nunique()
h_s      = compute_entropy(df[LABEL])
h_sl     = compute_entropy(left_df[LABEL])
h_sr     = compute_entropy(right_df[LABEL])
delta    = math.log2(3**k - 2) - k*h_s + k_l*h_sl + k_r*h_sr
rhs      = (math.log2(n - 1) + delta) / n

print(f"    n                 = {n}")
print(f"    k (classes in S)  = {k}")
print(f"    kL                = {k_l}   kR = {k_r}")
print(f"    H(S)              = {h_s:.4f}")
print(f"    H(SL)             = {h_sl:.4f}   H(SR) = {h_sr:.4f}")
print(f"    Δ(S,t)            = log2(3^{k}-2) - {k}×{h_s:.4f} + {k_l}×{h_sl:.4f} + {k_r}×{h_sr:.4f}")
print(f"                      = {delta:.4f}")
print(f"    RHS threshold     = (log2({n}-1) + {delta:.4f}) / {n} = {rhs:.4f}")
print(f"    IG(t*)            = {best_ig:.4f}")
accepted = mdlp_criterion(df, best_t, ATTR, LABEL)
print(f"    MDLP result       : {'✓ ACCEPTED  (IG > RHS)' if accepted else '✗ REJECTED  (IG ≤ RHS)'}")


# ─────────────────────────────────────────────────────────────────────────────
#  SECTION 4 — Full Programmatic Boundaries
# ─────────────────────────────────────────────────────────────────────────────

programmatic_boundaries = entropy_bin(df, ATTR, LABEL)

print(f"\n[4] Final Boundaries — Programmatic Output")
print(f"    Boundaries : {programmatic_boundaries}")
print(f"    Bins       : {len(programmatic_boundaries) + 1}")


# ─────────────────────────────────────────────────────────────────────────────
#  SECTION 5 — Comparison with Student 2 Manual Calculations
# ─────────────────────────────────────────────────────────────────────────────

print(f"\n[5] Comparison with Student 2 Manual Calculations")
print(f"    Manual boundaries     : {MANUAL_BOUNDARIES}")
print(f"    Programmatic boundaries: {programmatic_boundaries}")

# Match each manual boundary to the nearest programmatic one
unmatched_manual  = list(MANUAL_BOUNDARIES)
unmatched_prog    = list(programmatic_boundaries)
matched_pairs     = []
discrepancies     = []

for mb in MANUAL_BOUNDARIES:
    # Find the closest programmatic boundary
    if unmatched_prog:
        closest = min(unmatched_prog, key=lambda x: abs(x - mb))
        diff    = abs(closest - mb)
        if diff <= TOLERANCE:
            matched_pairs.append((mb, closest, diff))
            unmatched_prog.remove(closest)
        else:
            discrepancies.append(
                f"Manual boundary {mb} has no programmatic match "
                f"(closest: {closest}, diff = {diff:.4f})"
            )
    else:
        discrepancies.append(
            f"Manual boundary {mb} has no programmatic match (list exhausted)"
        )

if unmatched_prog:
    for pb in unmatched_prog:
        discrepancies.append(
            f"Programmatic boundary {pb} has no manual counterpart"
        )

print(f"\n    Matched pairs (within ±{TOLERANCE}):")
if matched_pairs:
    for mb, pb, diff in matched_pairs:
        print(f"      Manual {mb:>6}  ↔  Programmatic {pb:>6}  (diff = {diff:.4f})  ✓")
else:
    print("      None")

print(f"\n    Discrepancies:")
if discrepancies:
    for d in discrepancies:
        print(f"      ✗ {d}")
else:
    print("      None — full agreement between manual and programmatic results. ✓")


# ─────────────────────────────────────────────────────────────────────────────
#  SECTION 6 — Conclusion
# ─────────────────────────────────────────────────────────────────────────────

print(f"\n[6] Validation Conclusion")
print(f"    {'─'*60}")
if not discrepancies:
    print("    The programmatic implementation fully agrees with Student 2's")
    print("    manual calculations. No bugs detected.")
else:
    print(f"    {len(discrepancies)} discrepancy/discrepancies found.")
    print("    Recommended next steps:")
    print("      1. Re-check Student 2's worksheet for the flagged boundaries.")
    print("      2. Add a print statement inside entropy_bin() at each recursion")
    print("         level to trace the exact split chosen and the MDLP outcome.")
    print("      3. Verify that both student and code use midpoints (not raw values)")
    print("         as split thresholds.")
    print("      4. Confirm both use the same k (distinct classes per partition,")
    print("         not the global k=3) in the MDLP delta formula.")

print(f"\n{'='*70}\n")