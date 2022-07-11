#!/usr/bin/env python

"""
    sinkhorn_wmd/main.py
    
    !! Note to program performers  !! 
    
    There are minor differences between 
        this implementation and the one that was previously released as an program workflow.
        
        1) We just run for a fixed number of iterations 
            here, for ease of reproducibility
        2) The `reg` parameter in the previous implementation 
            is equivilant to `1 / lamb` here to match pseudocode
    
    The computation of each iteration is the same, though we're using standard
    elementwise multiplication and matrix multiplication instead of the custom
    numba kernels from the previous version, for ease of explanation.  If you find
    edge cases where this is producing different results from the program workflow, 
    please let me know.
    
    ~ Ben Johnson (bkj.322@gmail.com)
"""

import os
import sys
import argparse
import numpy as np
from time import time
from scipy import sparse
from scipy.spatial.distance import cdist

# --
# Sinkhorn

#@profile
def sinkhorn_wmd(r, c, vecs, lamb, max_iter):
    """
        r (np.array):          query vector (sparse, but represented as dense)
        c (sparse.csr_matrix): data vectors, in CSR format.  shape is `(dim, num_observations)`
        vecs (np.array):       embedding vectors, from which we'll compute a distance matrix
        lamb (float):          regularization parameter -- note this is (1 / reg) from previous original implementation
        max_iter (int):        maximum number of iterations
        
        Inline comments reference pseudocode from Alg. 1 in paper
            https://arxiv.org/pdf/1306.0895.pdf
    """
    # I=(r > 0)
    sel = r.squeeze() > 0
    
    # r=r(I)
    r = r[sel].reshape(-1, 1).astype(np.float64)
    
    print(r.shape[0])

    # M=M(I,:)
    M = cdist(vecs[sel], vecs).astype(np.float64)
    
    # x=ones(lenth(r), size(c,2)) / length(r)
    a_dim  = r.shape[0]
    b_nobs = c.shape[1]
    x      = np.ones((a_dim, b_nobs)) / a_dim 
    
    # K=exp(-lambda * M)
    K = np.exp(M * lamb)
    p=(1 / r) * K
    KT=K.T
    KM=(K * M)
    # while x changes: x=diag(1./r)*K*(c.*(1./(K’*(1./x))))
    
    t = time()
    it = 0
    while it < max_iter:
        
        u = 1.0 / x
        v = c.multiply(1 / (KT @ u))
        x = p @ v.tocsc()
        
        it += 1
        
        #print('it=%d | %f' % (it, time() - t), file=sys.stderr)
    
    # d_lamba_M(r,c) = sum(u.*((K.*M)*v)
    u = 1.0 / x
    v = c.multiply(1 / (KT @ u))
    return (u * ((KM) @ v)).sum(axis=0)


# --
# CLI

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--inpath', type=str, default='data/cache')
    parser.add_argument('--n-docs', type=int, default=5000)
    parser.add_argument('--query-idx', type=int, default=100)
    parser.add_argument('--lamb', type=float, default=-1)
    parser.add_argument('--max_iter', type=int, default=15)
    args = parser.parse_args()
    
    # !! In order to check accuracy, you _must_ use these parameters !!
    assert args.inpath == 'data/cache'
    assert args.n_docs == 5000
    assert args.query_idx == 100
    
    return args


if __name__ == "__main__":
    args = parse_args()
    
    # --
    # IO
    
    vecs = np.load(args.inpath + '-vecs.npy')
    mat  = sparse.load_npz(args.inpath + '-mat.npz')
    
    # --
    # Prep
    
    # Maybe subset docs
    if args.n_docs:
        mat  = mat[:,:args.n_docs]
    
    # --
    # Run
    
    # Get query vector
    r = np.asarray(mat[:,args.query_idx].todense()).squeeze()
   
    t = time()
    scores = sinkhorn_wmd(r, mat, vecs, lamb=args.lamb, max_iter=args.max_iter)
 
    elapsed = time() - t
    print('elapsed=%f' % elapsed, file=sys.stderr)
    
    # --
    # Write output
    
    os.makedirs('results', exist_ok=True)
    
    np.savetxt('results/scores', scores, fmt='%.8f')
    open('results/elapsed', 'w').write(str(elapsed))
