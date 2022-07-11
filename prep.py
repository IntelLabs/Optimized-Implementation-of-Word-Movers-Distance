#!/usr/bin/env python

"""
    prep.py
"""

import io
import sys
import argparse
import numpy as np
import pandas as pd
from scipy import sparse
from tqdm import trange, tqdm

# --
# IO Helpers

def load_vecs(path, n_toks=100000):
    fin   = io.open(path, 'r', encoding='utf-8', newline='\n', errors='ignore')
    _, d  = map(int, fin.readline().split())
    
    toks = [None] * n_toks
    vecs = np.zeros((n_toks, d))
    
    for idx in trange(n_toks):
        x         = next(fin).rstrip().split(' ')
        toks[idx] = x[0]
        vecs[idx] = x[1:]
    
    return np.array(toks), vecs


def docs2mat(docs, tok_lookup, min_toks_in_doc=5):
    indptr        = [0]
    indices       = []
    filtered_docs = []
    for doc in tqdm(docs):
        toks = sorted(set([tok_lookup[tok] for tok in doc.split(' ') if tok in tok_lookup]))
        if len(toks) >= min_toks_in_doc:
            indptr.append(indptr[-1] + len(toks))
            indices += toks
            filtered_docs.append(doc)
        
    vals = np.ones(len(indices))
    
    mat = sparse.csr_matrix((vals, indices, indptr), shape=(len(filtered_docs), vecs.shape[0]))
    mat = mat.T.tocsr()
    mat = mat.multiply(1 / mat.sum(axis=0)).tocsr() # Normalize documents
    
    return mat, filtered_docs


# --
# Run

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--outpath', type=str, default='data/cache')
    args = parser.parse_args()
    
    assert args.outpath == 'data/cache'
    
    return args


if __name__ == "__main__":
    
    word2vec_path = 'data/crawl-300d-2M.vec'
    text_path     = 'data/dbpedia.train'
    
    args = parse_args()
    
    # --
    # Load + prep
    
    print('loading word embeddings from %s' % word2vec_path)
    toks, vecs = load_vecs(path=word2vec_path)
    tok_lookup = dict(zip(toks, range(len(toks))))
    
    print('loading text data from %s' % text_path, file=sys.stderr)
    docs = pd.read_csv(text_path, header=None)
    docs = (docs[1] + ' ' + docs[2]).values
    mat, docs = docs2mat(docs, tok_lookup)
    
    # --
    # Save
    
    print('saving to %s' % args.outpath, file=sys.stderr)
    sparse.save_npz(args.outpath + '-mat', mat)
    open(args.outpath + '-docs', 'w').write('\n'.join(docs))
    np.save(args.outpath + '-toks', toks)
    np.save(args.outpath + '-vecs', vecs)


