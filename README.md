# Sinkhorn Word Movers Distance

This repository hosts the source code for an efficient implementation of "Word Mover's Distance" (WMD) using the Sinkhorn-Knopp algorithm. 
Paper reference will be added upon publication.

# To Compile

* REQUIREMENT: gcc version gcc-7.1.0 or higher
* source your_icc_compiler 
* source compile

# To Run
* Download the embedding file from https://www.kaggle.com/datasets/yekenot/fasttext-crawl-300d-2m. We do not provide the file, since it is large.

* Then perform the following steps to prepare the input file.
*   * take first 100001 lines: head -n100001 crawl-300d-2M.vec >test.out
*   * remove first line: sed '1d' test.out > test2.out
*   * remove first column of each line: cut -d" " -f2- test2.out > data/vecs.out
*   * discard temporary files: rm test.out test2.out 


* set KMP AFFINITY. For example: export KMP_AFFINITY=compact,1,0,granularity=fine
* ./name_of_executable

* There is also a small input in data (v2, r2, sample.mat, set the input in the program to run, set word2vec size to 3).

### Please cite this work as:
@article{tithi2020efficient,
  title={An Efficient Shared-memory Parallel Sinkhorn-Knopp Algorithm to Compute the Word Mover's Distance},
  author={Tithi, Jesmin Jahan and Petrini, Fabrizio},
  journal={arXiv preprint arXiv:2005.06727},
  year={2020}
}

## References

- [1] [From Word Embeddings To Document Distances](http://proceedings.mlr.press/v37/kusnerb15.pdf)
- [2] [Sinkhorn Distances: Lightspeed Computation of Optimal Transportation Distances](https://arxiv.org/pdf/1306.0895.pdf)
- [3] [Beginner's Guide to Word2Vec and Neural Word Embeddings](https://skymind.ai/wiki/word2vec)
- [4] [Notes on Optimal Transport](https://michielstock.github.io/OptimalTransport/)

<!--  Reviewed 5/1/23 MRB -->
