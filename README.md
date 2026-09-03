# rpforest-ann

Random Projection Forest for Approximate Nearest Neighbor (ANN) search, implemented from first principles in C.

## Status

Phsae 1 complete: vector/dataset representation and synthetic dataset generation, verified through tests. Phase 2 (random projection tree) in progress. See commit history and [DESIGN.md](DESIGN.md) for details.

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

### Random projection tree and forest

TBD, filled in once Phase 2 and Phase 3 are implemented.

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
