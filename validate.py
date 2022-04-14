#!/usr/bin/env python

"""
    sinkhorn_wmd/validate.py
"""

import json
import numpy as np

MAX_DIFF_THRESH = 1e-6

if __name__ == "__main__":
    
    # --
    # Load scores
    
    scores = open('results/scores').read().splitlines()
    scores = np.array([float(xx) for xx in scores])
    
    # --
    # Load targets
    
    target = open('./scores').read().splitlines()
    target = np.array([float(xx) for xx in target])
    
    # --
    # Check correctness
    
    max_diff = float(np.abs(target - scores).max())
    
    # --
    # Log
    
    print(json.dumps({
        "max_diff" : max_diff,
        "status"   : "PASS" if max_diff < MAX_DIFF_THRESH else "FAIL",
    }))
    
    does_pass = "PASS" if max_diff < MAX_DIFF_THRESH else "FAIL"
    open('results/.pass', 'w').write(str(does_pass))
