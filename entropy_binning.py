import pandas as pd
import numpy as np
import math

def compute_entropy(labels):
    """
    Task 3.1.1: Calculates the Shannon Entropy of a list of class labels.
    I used a Series to count occurrences and handled the 0*log(0) case 
    by skipping any class with zero instances.
    """
    if len(labels) == 0:
        return 0
    
    # Get the counts and turn them into probabilities
    counts = pd.Series(labels).value_counts()
    probs = counts / len(labels)
    
    # Formula: H(S) = -sum(p * log2(p))
    entropy = -sum(p * math.log2(p) for p in probs if p > 0)
    return entropy

def get_candidate_splits(values):
    """
    Task 3.1.3: Finds midpoints between unique consecutive values.
    I made sure to use np.unique first so we don't try to split 
    between identical values (a common error mentioned in the brief).
    """
    sorted_unique = sorted(np.unique(values))
    # Calculate the average between each pair of sorted values
    midpoints = [(sorted_unique[i] + sorted_unique[i+1]) / 2 
                 for i in range(len(sorted_unique) - 1)]
    return midpoints

def compute_information_gain(data, threshold, attr, label):
    """
    Task 3.1.2: Calculates the Information Gain for a specific split point.
    Calculates the entropy of the current group and subtracts the 
    weighted entropy of the left and right bins.
    """
    n = len(data)
    h_total = compute_entropy(data[label])
    
    # Divide the data into two groups based on the threshold
    left_bin = data[data[attr] <= threshold][label]
    right_bin = data[data[attr] > threshold][label]
    
    # Calculate weights and bin entropies
    n_l, n_r = len(left_bin), len(right_bin)
    h_left = compute_entropy(left_bin)
    h_right = compute_entropy(right_bin)
    
    # Weighted average entropy
    h_weighted = (n_l / n) * h_left + (n_r / n) * h_right
    
    return h_total - h_weighted

def find_best_split(data, attr, label):
    """
    Task 3.1.4: Loops through all midpoints to find the one with the highest IG.
    This tells us exactly where the most informative cut is.
    """
    candidates = get_candidate_splits(data[attr])
    if not candidates:
        return None, 0
    
    best_ig = -1
    best_threshold = None
    
    for t in candidates:
        current_ig = compute_information_gain(data, t, attr, label)
        if current_ig > best_ig:
            best_ig = current_ig
            best_threshold = t
            
    return best_threshold, best_ig