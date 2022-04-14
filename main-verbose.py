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

from logs.loggers import log_header, log_array, log_sparse_vector, log_sparse_matrix

np.set_printoptions(precision=4)

# --
# Sinkhorn

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
    
    log_header('INIT sinkhorn_wmd', n=50)
    
    log_sparse_vector('INPUT r:   ', r)
    log_sparse_matrix('INPUT c:   ', c)
    log_array('INPUT vecs:', vecs)
    print('INPUT lamb:    ', lamb)
    print('INPUT max_iter:', max_iter)
    
    # I=(r > 0)
    sel = r.squeeze() > 0
    log_sparse_vector('INIT I:', sel)
    
    # r=r(I)
    r = r[sel].reshape(-1, 1).astype(np.float64)
    log_array('FILTER r:', r.squeeze())
    
    # M=M(I,:)
    M = cdist(vecs[sel], vecs).astype(np.float64)
    log_array('INIT M:', M)
    
    # K=exp(-lambda * M)
    K = np.exp(- M * lamb)
    log_array('INIT K:', K)
    
    # x=ones(lenth(r), size(c,2)) / length(r)
    a_dim  = r.shape[0]
    b_nobs = c.shape[1]
    x      = np.ones((a_dim, b_nobs)) / a_dim 
    log_array('INIT x:', x)
    
    # while x changes: x=diag(1./r)*K*(c.*(1./(K’*(1./x))))
    
    t = time()
    it = 0
    while it < max_iter:
        log_header('START ITERATION:', it, n=25)
        
        u = 1.0 / x
        v = c.multiply(1 / (K.T @ u))
        x = (1 / r) * K @ v.tocsc()
        
        log_array('x:', x)
        
        it += 1
    
    # d_lamba_M(r,c) = sum(u.*((K.*M)*v)
    u = 1.0 / x
    v = c.multiply(1 / (K.T @ u))
    
    log_array('u:', u)
    log_sparse_matrix('v:', v)
    
    out = (u * ((K * M) @ v)).sum(axis=0)
    log_array('out:', out)
    return out


# --
# CLI

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--inpath', type=str, default='data/cache')
    parser.add_argument('--n-docs', type=int, default=5000)
    parser.add_argument('--query-idx', type=int, default=100)
    parser.add_argument('--lamb', type=float, default=1)
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
    
    docs = open(args.inpath + '-docs').read().splitlines()
    vecs = np.load(args.inpath + '-vecs.npy')
    mat  = sparse.load_npz(args.inpath + '-mat.npz')
    
    # --
    # Prep
    
    # Maybe subset docs
    if args.n_docs:
        docs = docs[:args.n_docs]
        mat  = mat[:,:args.n_docs]
    
    # --
    # Run
    
    # Get query vector
    r = np.asarray(mat[:,args.query_idx].todense()).squeeze()
    
    # t = time()
    scores = sinkhorn_wmd(r, mat, vecs, lamb=args.lamb, max_iter=args.max_iter)
    # elapsed = time() - t
    # print('elapsed=%f' % elapsed, file=sys.stderr)
    
    # # --
    # # Write output
    
    # os.makedirs('results', exist_ok=True)
    
    # np.savetxt('results/scores', scores, fmt='%.8e')
    # open('results/elapsed', 'w').write(str(elapsed))