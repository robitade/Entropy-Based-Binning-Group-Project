import pandas as pd
import numpy as np
import math

# ─────────────────────────────────────────────
#  WEEK 1 FUNCTIONS (kept here for completeness)
# ─────────────────────────────────────────────

def compute_entropy(labels):
    """
    Calculates the Shannon Entropy H(S) of a list/Series of class labels.
    Handles the 0·log2(0) = 0 convention by skipping zero-probability classes.

    Parameters
    ----------
    labels : list or pd.Series
        Class labels for a set of instances.

    Returns
    -------
    float
        Entropy in bits.

    Example
    -------
    >>> compute_entropy(['L', 'L', 'H'])
    0.9182958340544896
    """
    if len(labels) == 0:
        return 0.0

    counts = pd.Series(labels).value_counts()
    probs  = counts / len(labels)
    return -sum(p * math.log2(p) for p in probs if p > 0)


def get_candidate_splits(values):
    """
    Returns all midpoint candidate split thresholds for a 1-D attribute array.
    Only generates midpoints between *distinct* consecutive values, as required
    by Fayyad & Irani (1993) — splitting between identical values is meaningless.

    Parameters
    ----------
    values : array-like
        Raw (possibly unsorted, possibly duplicate) attribute values.

    Returns
    -------
    list of float
        Sorted list of candidate thresholds.

    Example
    -------
    >>> get_candidate_splits([22, 24, 26])
    [23.0, 25.0]
    """
    sorted_unique = sorted(np.unique(values))
    return [(sorted_unique[i] + sorted_unique[i + 1]) / 2
            for i in range(len(sorted_unique) - 1)]


def compute_information_gain(data, threshold, attr, label):
    """
    Calculates Information Gain IG(t) for a single candidate split threshold t.

    IG(t) = H(S) - (nL/n)·H(SL) - (nR/n)·H(SR)

    Parameters
    ----------
    data      : pd.DataFrame  — current partition
    threshold : float         — candidate split point
    attr      : str           — name of the continuous attribute column
    label     : str           — name of the class label column

    Returns
    -------
    float : Information Gain in bits (always >= 0)
    """
    n       = len(data)
    h_total = compute_entropy(data[label])

    left  = data[data[attr] <= threshold][label]
    right = data[data[attr] >  threshold][label]

    n_l, n_r = len(left), len(right)

    # Guard: a split that puts everything on one side gives IG = 0
    if n_l == 0 or n_r == 0:
        return 0.0

    h_weighted = (n_l / n) * compute_entropy(left) + \
                 (n_r / n) * compute_entropy(right)

    return h_total - h_weighted


def find_best_split(data, attr, label):
    """
    Evaluates every candidate split and returns the one with the highest IG.

    Parameters
    ----------
    data  : pd.DataFrame
    attr  : str  — continuous attribute column name
    label : str  — class label column name

    Returns
    -------
    (best_threshold, best_ig) : (float | None, float)
    """
    candidates = get_candidate_splits(data[attr])
    if not candidates:
        return None, 0.0

    best_ig        = -float('inf')
    best_threshold = None

    for t in candidates:
        ig = compute_information_gain(data, t, attr, label)
        if ig > best_ig:
            best_ig        = ig
            best_threshold = t

    return best_threshold, max(best_ig, 0.0)


# ─────────────────────────────────────────────
#  WEEK 2  —  TASK 3.2
#  MDLP Stopping Criterion & Recursive Engine
# ─────────────────────────────────────────────

def mdlp_criterion(data, threshold, attr, label):
    """
    Fayyad–Irani Minimum Description Length Principle (MDLP) stopping test.

    A split at *threshold* is accepted only when:

        IG(S, t) > [log2(n - 1) + Δ(S, t)] / n

    where:
        Δ(S, t) = log2(3^k - 2) - k·H(S) + kL·H(SL) + kR·H(SR)
        k  = number of distinct classes in S
        kL = number of distinct classes in SL  (left partition)
        kR = number of distinct classes in SR  (right partition)

    Parameters
    ----------
    data      : pd.DataFrame  — current partition (already subset + index-reset)
    threshold : float
    attr      : str
    label     : str

    Returns
    -------
    bool : True if the split passes MDLP (i.e., the split should be accepted).
    """
    n  = len(data)
    if n <= 1:
        return False

    left  = data[data[attr] <= threshold]
    right = data[data[attr] >  threshold]

    # Class label sets for each partition
    s_labels  = data[label]
    sl_labels = left[label]
    sr_labels = right[label]

    # Number of *distinct* classes present in each partition
    k  = s_labels.nunique()
    k_l = sl_labels.nunique()
    k_r = sr_labels.nunique()

    h_s  = compute_entropy(s_labels)
    h_sl = compute_entropy(sl_labels)
    h_sr = compute_entropy(sr_labels)

    ig = compute_information_gain(data, threshold, attr, label)

    # MDLP delta term  (Equation 1 in the project brief)
    delta = math.log2(3 ** k - 2) - k * h_s + k_l * h_sl + k_r * h_sr

    # Right-hand side threshold
    rhs = (math.log2(n - 1) + delta) / n

    return ig > rhs


def entropy_bin(data, attr, label, min_size=2, depth=0, max_depth=10):
    """
    Main recursive Fayyad–Irani entropy-based discretization function.

    Algorithm
    ---------
    1. Compute H(S) of the current partition.
    2. Find the best split threshold t* (highest IG).
    3. Test the MDLP stopping criterion.
       • If STOP  → return [] (no boundaries; this partition is a leaf).
       • If CONTINUE → split into SL and SR, recurse on both, combine results.

    Parameters
    ----------
    data      : pd.DataFrame  — current partition
    attr      : str           — continuous attribute being discretized
    label     : str           — class label column
    min_size  : int           — minimum instances required in a partition to split
    depth     : int           — current recursion depth (internal use)
    max_depth : int           — hard limit on recursion depth (safety guard)

    Returns
    -------
    list of float
        Sorted list of split boundary thresholds found within this partition.

    Notes
    -----
    - data must be sorted by attr and have a reset integer index before the call.
    - The function never modifies the input DataFrame.
    """
    # ── Base cases ──────────────────────────────────────────────────────────
    if depth >= max_depth:
        return []
    if len(data) < 2 * min_size:         # too few instances to split sensibly
        return []
    if data[label].nunique() == 1:       # pure partition — no entropy to reduce
        return []

    # ── Step 1 & 2: find best split ─────────────────────────────────────────
    t_star, ig_star = find_best_split(data, attr, label)

    if t_star is None or ig_star <= 0:
        return []

    # ── Step 3: MDLP test ───────────────────────────────────────────────────
    if not mdlp_criterion(data, t_star, attr, label):
        return []                         # MDLP says: don't split here

    # ── Split accepted: partition and recurse ────────────────────────────────
    left  = data[data[attr] <= t_star].reset_index(drop=True)
    right = data[data[attr] >  t_star].reset_index(drop=True)

    boundaries_left  = entropy_bin(left,  attr, label,
                                   min_size=min_size,
                                   depth=depth + 1,
                                   max_depth=max_depth)
    boundaries_right = entropy_bin(right, attr, label,
                                   min_size=min_size,
                                   depth=depth + 1,
                                   max_depth=max_depth)

    return sorted(boundaries_left + [t_star] + boundaries_right)


def apply_bins(data, attr, boundaries, labels=None):
    """
    Replaces the continuous *attr* column with a discrete bin label.

    Parameters
    ----------
    data       : pd.DataFrame  — original dataset (not modified in-place)
    attr       : str           — attribute to discretize
    boundaries : list of float — sorted split thresholds from entropy_bin()
    labels     : list of str | None
                 Optional human-readable bin names.
                 If None, bins are labelled "Bin 1", "Bin 2", …
                 Must have length = len(boundaries) + 1 if provided.

    Returns
    -------
    pd.DataFrame
        Copy of *data* with *attr* replaced by a categorical bin column.

    Example
    -------
    >>> result = apply_bins(df, 'Age', [38.5, 50.0])
    # Adds column 'Age_bin' with values 'Bin 1', 'Bin 2', 'Bin 3'
    """
    df      = data.copy()
    n_bins  = len(boundaries) + 1

    if labels is None:
        labels = [f"Bin {i + 1}" for i in range(n_bins)]

    if len(labels) != n_bins:
        raise ValueError(
            f"Expected {n_bins} labels for {len(boundaries)} boundaries, "
            f"got {len(labels)}."
        )

    # Build edges: -inf … b1 … b2 … +inf
    edges = [-np.inf] + sorted(boundaries) + [np.inf]

    bin_col = attr + "_bin"
    df[bin_col] = pd.cut(
        df[attr],
        bins=edges,
        labels=labels,
        right=True          # intervals are (lower, upper]  i.e. attr <= threshold → left bin
    )

    return df