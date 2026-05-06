"""
run_binning.py  —  Student 3, Week 2 Driver Script
===================================================
Applies the entropy-based binning algorithm to all three continuous
attributes (Age, BMI, Glucose Level) in the shared patient dataset,
prints results, and produces a summary table.

Usage
-----
    python run_binning.py

Assumes patient_data.csv is in the same directory.
If the CSV is not present, the script builds the dataset inline.
"""

import pandas as pd
import numpy as np
from entropy_binning import entropy_bin, apply_bins, compute_entropy

# ─────────────────────────────────────────────
#  1.  Load / Build the Dataset
# ─────────────────────────────────────────────

def build_dataset():
    """Inline recreation of the 30-patient dataset from Section 2 of the brief."""
    records = [
        (1,  22, 18.5,  72,  "Low"),
        (2,  24, 20.1,  80,  "Low"),
        (3,  26, 21.4,  85,  "Low"),
        (4,  28, 24.2, 110,  "Medium"),
        (5,  30, 22.8,  95,  "Low"),
        (6,  31, 26.5, 130,  "Medium"),
        (7,  33, 23.0, 100,  "Low"),
        (8,  35, 27.8, 140,  "Medium"),
        (9,  36, 25.5, 115,  "Low"),
        (10, 38, 28.3, 150,  "High"),
        (11, 39, 24.7, 125,  "Medium"),
        (12, 41, 29.1, 158,  "High"),
        (13, 42, 26.2, 120,  "Medium"),
        (14, 43, 30.5, 165,  "High"),
        (15, 45, 25.0, 110,  "Medium"),
        (16, 46, 27.3, 145,  "Medium"),
        (17, 48, 31.0, 155,  "High"),
        (18, 49, 28.5, 130,  "Medium"),
        (19, 51, 33.4, 170,  "High"),
        (20, 53, 29.6, 148,  "Medium"),
        (21, 55, 34.1, 182,  "High"),
        (22, 57, 30.7, 160,  "High"),
        (23, 59, 32.2, 175,  "High"),
        (24, 61, 35.5, 190,  "High"),
        (25, 63, 33.8, 185,  "High"),
        (26, 65, 36.0, 195,  "High"),
        (27, 67, 31.5, 165,  "High"),
        (28, 70, 37.2, 200,  "High"),
        (29, 72, 38.0, 205,  "High"),
        (30, 75, 42.0, 210,  "High"),
    ]
    df = pd.DataFrame(records,
                      columns=["PatientID", "Age", "BMI", "Glucose", "Risk_Level"])
    return df


try:
    df = pd.read_csv("patient_data.csv")
    print("✓ Loaded patient_data.csv\n")
except FileNotFoundError:
    df = build_dataset()
    print("ℹ  patient_data.csv not found — using inline dataset.\n")

LABEL = "Risk_Level"
ATTRS = ["Age", "BMI", "Glucose"]


# ─────────────────────────────────────────────
#  2.  Helper: Pretty Print Results
# ─────────────────────────────────────────────

def print_bin_summary(df_binned, attr, boundaries):
    """
    Prints a formatted summary table for one binned attribute showing
    boundaries, instance count, class distribution, and bin entropy.
    """
    bin_col   = attr + "_bin"
    n_bins    = len(boundaries) + 1
    edges     = [-np.inf] + sorted(boundaries) + [np.inf]

    print(f"\n{'─'*65}")
    print(f"  Attribute : {attr}")
    print(f"  Boundaries: {[round(b, 2) for b in boundaries]}")
    print(f"  Bins found: {n_bins}")
    print(f"{'─'*65}")
    print(f"  {'Bin':<8} {'Range':<22} {'n':>4}  {'Low':>5} {'Med':>5} {'High':>5}  {'H(bin)':>7}")
    print(f"  {'─'*8} {'─'*22} {'─'*4}  {'─'*5} {'─'*5} {'─'*5}  {'─'*7}")

    for i, bin_label in enumerate(df_binned[bin_col].cat.categories):
        subset   = df_binned[df_binned[bin_col] == bin_label]
        counts   = subset[LABEL].value_counts()
        n        = len(subset)
        low      = counts.get("Low",    0)
        med      = counts.get("Medium", 0)
        high     = counts.get("High",   0)
        h        = compute_entropy(subset[LABEL])

        lo_edge  = edges[i]
        hi_edge  = edges[i + 1]
        lo_str   = f"{lo_edge:.1f}" if lo_edge != -np.inf else "-∞"
        hi_str   = f"{hi_edge:.1f}" if hi_edge !=  np.inf else "+∞"
        rng      = f"({lo_str}, {hi_str}]"

        print(f"  {str(bin_label):<8} {rng:<22} {n:>4}  {low:>5} {med:>5} {high:>5}  {h:>7.4f}")

    print()


# ─────────────────────────────────────────────
#  3.  Run Algorithm on All Three Attributes
# ─────────────────────────────────────────────

print("=" * 65)
print("  ENTROPY-BASED BINNING  —  Student 3 Week 2 Results")
print("=" * 65)

results = {}   # stores {attr: boundaries} for validation section

for attr in ATTRS:
    # Ensure sorted by attribute, reset index before recursion
    df_sorted = df.sort_values(attr).reset_index(drop=True)

    boundaries = entropy_bin(df_sorted, attr=attr, label=LABEL)
    results[attr] = boundaries

    df_binned  = apply_bins(df_sorted, attr, boundaries)
    print_bin_summary(df_binned, attr, boundaries)


# ─────────────────────────────────────────────
#  4.  Brief Sanity Check vs. Brief Hint
# ─────────────────────────────────────────────
#
#  The brief (Section 7.2) hints: "the optimal first split for Age
#  is somewhere in the Age 38–42 range."
#
age_boundaries = results["Age"]
print("Sanity check — Age boundaries:", age_boundaries)
if age_boundaries:
    first_split = age_boundaries[0]
    if 38 <= first_split <= 43:
        print(f"  ✓ First split {first_split} is within the expected 38–42 range.\n")
    else:
        print(f"  ✗ First split {first_split} is OUTSIDE the expected range — recheck.\n")
else:
    print("  ✗ No boundaries found for Age — check MDLP parameters.\n")


# ─────────────────────────────────────────────
#  5.  Summary Table
# ─────────────────────────────────────────────

print("=" * 65)
print("  SUMMARY TABLE")
print("=" * 65)
print(f"  {'Attribute':<12} {'# Bins':>7}  Boundaries")
print(f"  {'─'*12} {'─'*7}  {'─'*35}")
for attr, bds in results.items():
    bds_str = str([round(b, 2) for b in bds]) if bds else "No split (pure or tiny)"
    print(f"  {attr:<12} {len(bds)+1:>7}  {bds_str}")
print()

