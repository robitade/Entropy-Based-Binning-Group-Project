from entropy_binning import compute_entropy, get_candidate_splits

def run_tests():
    print("--- Running Week 1 Unit Tests ---")
    
    # Test 1: Pure set (Page 10 requirement)
    h1 = compute_entropy(['L', 'L', 'L', 'L'])
    print(f"Test 1 [L,L,L,L]: H = {h1} (Expected: 0)")
    
    # Test 2: Balanced binary (Page 10 requirement)
    h2 = compute_entropy(['L', 'H'])
    print(f"Test 2 [L,H]: H = {h2} (Expected: 1.0)")
    
    # Test 3: Mixed set (Page 10 requirement)
    h3 = compute_entropy(['L', 'L', 'H'])
    print(f"Test 3 [L,L,H]: H = {h3:.4f} (Expected: ~0.9183)")

    # Test 4: Candidate Splits (Check for midpoint logic)
    cands = get_candidate_splits([10, 20, 20, 30])
    print(f"Test 4 Midpoints: {cands} (Expected: [15.0, 25.0])")

    print("\n--- Tests Complete ---")

if __name__ == "__main__":
    run_tests()