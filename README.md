# Sinkhorn Word Movers Distance (sinkhorn_wmd)

This repository hosts the source code for an efficient implementation of "Word Mover's Distance" (WMD) using the Sinkhorn-Knopp algorithm. 
Paper reference will be added upon publication.

## Background

Word Mover's Distance is a way to measure the dissimilarity between two text documents.  

It is described fairly accessibly in [1] -- we attempt to motivate and summarize the paper here, but please read [1] up until Section 4.1.  

We'll use a different method to overcome computational inefficiencies (described in [2]), so reading Sections 4.1 and later is optional.

### Motivation

Suppose that we have two documents `A` and `B`:

 - A) `Trump speaks to the media in Illinois`
 - B) `The President greets the press in Chicago`.

If we throw away information about word ordering and capitalization, we can represent the documents in the so-called "bag-of-words" format:

 - A) `['illinois', 'in', 'media', 'speaks', 'the', 'to', 'trump']`
 - B) `['chicago', 'greets', 'in', 'president', 'press', 'the', 'the']`

If we further throw away very frequent and uninformative words (so-called "stop words"), we're left with:

 - A) `['illinois', 'media', 'speaks', 'trump']`
 - B) `['chicago', 'greets', 'president', 'press']`

How might we be able to compute the similarity or dissimilarity between these two documents? 

`A` and `B` don't contain any of the same words, so we can't look at things like set intersection.  However, we can see that the two sentences have basically the same meaning. Intuitively, we'd think that these sentences should be "more similar" to each other than to, for example, the sentence "The band gave a concert in Japan".  Word Mover's Distance is an algorithm that attempts to capture that intuition.

### Word2Vec

`word2vec` is a famous algorithm that computes a `d`-dimensional vector called a "word embedding" for every word in a set of documents.  If you're unfamiliar with word embeddings, you can find a whole bunch of explainer blog posts on Google that explain what they are and how they're estimated (for example, [3]).  For this programming task, you'll be given precalculated word embeddings as an input -- these were trained by Google on a very large number of documents scraped from the internet, and capture a lot of information about the meanings of words.  In particular, using the `word2vec` embeddings, it's natural to measure the dissimilarity between two words via the Euclidean distance between their embeddings in `d`-dimensional space:
```
distance(a, b) = sqrt(sum(pow(embeddings[a] - embeddings[b], 2)))
```
For example, if
```
distance("media", "press") = sqrt(sum(pow(embeddings["media"] - embeddings["press"], 2)))
distance("media", "japan") = sqrt(sum(pow(embeddings["media"] - embeddings["japan"], 2)))
```
we'd expect `distance("media", "press") < distance("media", "japan")`.

But remember, we want to be able to compute distances between _documents_, not just words.  If we think about the word vectors as signifying a point in `d`-dimensional space, then we can interpret the distance between words `a` and `b` as the "cost" of transporting a unit of "mass" from point `embeddings[a]` to point `embeddings[b]`.  Then, if we think of sentences `A` and `B` as _sets_ of points in space, with a unit of mass on each point in `A`, we can define the Word Mover's Distance to be the _minimum_ cumulative cost of transporting all of the mass from points `[embeddings[a] for a in A]` to points `[embeddings[b] for b in B]`.  Note that mass can and might flow from a single point in `A` to multiple points in `B` and visa versa.  The `num_words(A) x num_words(B)` matrix that specifies the flow of mass from `A` to `B` is called the "transportation plan".

__Note on terminology:__ Our distance metric is called "Word Mover's Distance" because of it's similarity to the "Earth Mover's Distance" metrics (EMD).  EMD is also sometimes called "optimal transportation distance".  These names make intuitive sense: EMD is the cost of the optimal way that you could transport dirt from a set of source piles to a set of destination piles.  As such, this family of optimization problems are well-studied in operations research, where people are sometimes interested in _literally_ moving dirt around.

## Algorithm Overview

Notice that computing the distance between documents `A` and `B` involves _solving an optimization problem_.  This might make you worry that WMD will be slow: if we have a database of 1M documents, running a single query involves solving 1M optimization problems!  [1] proposes a number of optimizations to speed up document retrieval using WMD.  However, in this task you're going to implement a different method, using the algorithm described in [2].  (Roughly, [1] proposes methods that reduce the _number_ of expensive WMD evaluations per query, while [2] devlops an approximation to WMD that (dramatically) reduces the _cost_ per query.)

[2] proposes a method that we can use to compute an approximation of WMD very cheaply.  Roughly, they add an entropy penalty to the optimization problem that encourages the solution to lie "close" to the (trivial) transportation plan that sends equal mass from each point in `A` to each point in `B`. ("Closeness" is measured by KL-divergence.  More details available in Section 3 of [2].)  This added penalty actually makes the problem substantially easier to solve.  Specifically, we can use the fast Sinkhorn-Knopp algorithm, and reduce the algorithmic complexity of the problem from `O(d ** 3 * log(d))` to (an easily parallelized) `O(d ** 2)`

Paper [2] has lots of math and proofs.  However, Algorithm 1 shows the relatively concise algorithm that we'll use to compute the approximate WMD distance:

This skips over some details like convergence checking, so we provide the following pseudocode for clarification (we'll just run for a fixed number of iterations for simplicity).  We also modify the algorithm slightly to lazily compute the distance matrix `M` based on the nonzero entries of the query `r`, instead of computing the full `vocab_size x vocab_size` distance matrix.

__Note:__ There are several opportunities for optimization here.  For instance, if you see something like `A * (B @ C)` where `A` is a sparse matrix, you might not actually want to compute all of `B @ C`...

__Another note:__ This version of the Sinkhorn distance algorithm is "vectorized" -- instead of computing the distance between query `r` and a single vector `c`, we're computing `db_size` distances all in one shot.  Thus, looks a little different from descriptions of the non-vectorized algorithm that you see elsewhere (eg, in [4]).

### Parameters

 - lambda = 1
 - max_iter = 15

All parameters are shown in `main-redacted.py`.

## Evaluation

__For an implementation to be considered "correct", the maximum absolute difference between your score and the reference score must be less than `1e-6`.__  Eg:
```
assert np.abs(target - output).max() < 1e-6
```

__Notes:__ On 8 threads, an unoptimized version of this code written in numpy/scipy runs in about a minute.  A more optimized version written in numpy/scipy/numba runs in < 1 second.

## References

- [1] [From Word Embeddings To Document Distances](http://proceedings.mlr.press/v37/kusnerb15.pdf)
- [2] [Sinkhorn Distances: Lightspeed Computation of Optimal Transportation Distances](https://arxiv.org/pdf/1306.0895.pdf)
- [3] [Beginner's Guide to Word2Vec and Neural Word Embeddings](https://skymind.ai/wiki/word2vec)
- [4] [Notes on Optimal Transport](https://michielstock.github.io/OptimalTransport/)

