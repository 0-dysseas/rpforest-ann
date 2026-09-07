# rpforest-ann

Random Projection Forest for Approximate Nearest Neighbor (ANN) search, implemented from first principles in C.

## Status

Phsae 1 complete: vector/dataset representation and synthetic dataset generation, verified through tests.
Phase 2 complete: random projection tree, hyperplane splits, recursive build and leaf buckets. 
Phase 3 complete: margin-based priority-queue search over a single tree.
Phase 4 () in progress. See commit history and [DESIGN.md](DESIGN.md) for details.

## Motivation

Implemented while studying the ideas behind Spotify's [Annoy](https://github.com/spotify/annoy) library.

## Build

```
make
./build/rpforest-ann
```

## Approach

### Data representation

A vector is D floats stored next to each other in memory. A dataset of N vectors is one single block of N * D floats, not N seperate allocations. This facilitates leaf-bucket scans and brute-force baseline when reading many consequetive vectors. Dimentionality D is set at runtime for easier comparison of different dimentionalities later on.

### Synthetic dataset generation

Real MFCC audio features were not available, so the project imitates this data with the generation of synthetic vectors with identical stattistical shape.

Each generatedd vector starts from a small number of shared random values, called latent factors. Every output dimention is built as a weighted combination of those same factors plus its own independent noise. Because different dimentions share the same factors, they end up correlated with each other, similar to how real dimensions usually are.

The dimensions also do not all carry the same amount of signal: later dimensions are given smaller variance, following the way real MFCC coefficients shrink in magnitude further down the coefficient index.

### Random projection tree

Each tree recursively partitions the dataset with random hyperplane splits on the equidistant of two randomly chosen points of each internal node's set. Every point goes to the closest pivot, distributing the points of the dataset in two spaces accordingly. This reduces algebraically to a single dot product and threshold comparison per point, the same approach used by Annoy. Recursion stops either once a node's point count drops to a fixed leaf sixe or a maximum depth is reached.

Verified against two properties, partition correctness (every point ends up in exactly one leaf) and split self-consistency (independently re-applying the tree's own split rule from the root agrees with were each point was actually placed).

### Search on single tree

A query descends one tree using dot product and threshold tests from the tree's creation, always continuing into the side the query falls on. Whenever a split is passed, instead of dicarding the non-taken side, it is pushed onto a priority queue with a priority equal to the smallest _margin_ (distance from split's threshold) seen anywhere along the path so far.
- A large margin means the decision was confident, so that untaken side stays low priority.
- A small margin means the decision was close and worth revisiting later.

The descend continues from the best entry left in the queue once a leaf is reached, collecting candidate points from every leaf visited, until either: a set _search budget_ of candidates has been collected or the queue runs out. Every collected candidate is scored by its actual distance to the query and the closest k are returned.

Verified with three checks: querying with a point already in the dataset always returns that same point as its own nearest neighbor at distance 0, returned results come back sorted by distance, and, over 50 fresh query points compared against an exact brute-force search, the tree matched about 61% of the true nearest 5 neighbors with a search budget covering 10% of the dataset. A single tree finding a majority but not all of the true neighbors at that budget is expected, this is exactly what combining multiple trees is meant to improve on.

### Random projection forest

TBD...

## Benchmarks

TBD. Brute force vs. tree and forest latency and recall@k, once implemented.

## Sources

- Erik Bernhardsson, "Nearest neighbor methods and vector models", [part 1](https://erikbern.com/2015/09/24/nearest-neighbor-methods-vector-models-part-1.html) and [part 2](https://erikbern.com/2015/10/01/nearest-neighbors-and-vector-models-part-2-how-to-search-in-high-dimensional-spaces.html)
- Dasgupta and Freund, ["Random projection trees and low dimensional manifolds"](https://cseweb.ucsd.edu/~dasgupta/papers/rptree-stoc.pdf) (STOC 2008)
- [spotify/annoy](https://github.com/spotify/annoy)
- [MFCC tutorial](http://practicalcryptography.com/miscellaneous/machine-learning/guide-mel-frequency-cepstral-coefficients-mfccs/), Practical Cryptography
- [Box-Muller transform](https://en.wikipedia.org/wiki/Box%E2%80%93Muller_transform), Wikipedia
- [Factor analysis](https://en.wikipedia.org/wiki/Factor_analysis), Wikipedia

## License

MIT. See [LICENSE](LICENSE).
