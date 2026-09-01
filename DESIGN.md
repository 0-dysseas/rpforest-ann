# Design Notes

Rationale for design decisions, updated per phase. Sources cited inline; full explanations happened
in the working sessions, this file is the durable record of what was decided and why.

## Phase 1 — Vector & dataset representation

- Each vector is D contiguous floats. Distance computation and hyperplane-projection both need to
  walk all D dimensions of one vector together, so the vector itself is the unit that must stay
  contiguous (not, e.g., one array per dimension across the dataset).
- The dataset is stored as a single contiguous `malloc`'d block of `N * D` floats (row-major: vector
  `i` occupies floats `[i*D, (i+1)*D)`), not as N separately-allocated vectors behind an array of
  pointers. Reasoning: the dataset is built once by the generator and only read afterward, so the
  per-vector-malloc approach's main advantage (independent resize/free) isn't needed, while its costs
  are real — pointer-chasing to reach each vector, unpredictable placement in memory, extra
  malloc/free overhead. One contiguous block gives cache-friendly sequential access (relevant every
  time a leaf bucket or the brute-force baseline scans multiple vectors) at effectively zero extra
  implementation cost over the alternative.
- Dimensionality D is a runtime parameter, not a compile-time constant — Phase 6 benchmarks latency
  across multiple dimensionalities, so it can't be fixed at build time.

## Phase 1 — Synthetic dataset generation

- Chose a low-rank shared-latent-factor model over generating a full covariance matrix and applying
  a Cholesky decomposition. Without real MFCC statistics to calibrate a covariance matrix against,
  Cholesky's extra statistical rigor doesn't translate into more realism here, while it adds real
  implementation and numerical-stability risk (an arbitrarily chosen covariance matrix isn't
  guaranteed positive semi-definite) to a supporting component rather than the core deliverable.
- Mechanism: draw a small number of shared Gaussian latent factors (via a Box-Muller uniform-to-
  Gaussian transform), combine them through weights into each of the D output dimensions, add
  independent per-dimension noise. Because multiple output dimensions share the same underlying
  factors, they come out correlated — this gives the dataset the kind of low-dimensional manifold
  structure that random projection trees are theoretically built to exploit (see Dasgupta & Freund
  below), rather than being uncorrelated high-dimensional noise.
- Per-dimension variance decay (mimicking how real MFCC coefficient magnitude falls off with index)
  applied via a scaling factor decreasing with dimension index.
- Exact factor count, weighting scheme, and decay schedule are implementation choices made while
  writing the generator.

## Sources consulted (Phase 1)

- Erik Bernhardsson, "Nearest neighbor methods and vector models" — part 1:
  https://erikbern.com/2015/09/24/nearest-neighbor-methods-vector-models-part-1.html
  and part 2: https://erikbern.com/2015/10/01/nearest-neighbors-and-vector-models-part-2-how-to-search-in-high-dimensional-spaces.html
- Dasgupta & Freund, "Random projection trees and low dimensional manifolds" (STOC 2008):
  https://cseweb.ucsd.edu/~dasgupta/papers/rptree-stoc.pdf
- spotify/annoy source: https://github.com/spotify/annoy
- MFCC tutorial (Practical Cryptography):
  http://practicalcryptography.com/miscellaneous/machine-learning/guide-mel-frequency-cepstral-coefficients-mfccs/
- Cepstrum — Wikipedia: https://en.wikipedia.org/wiki/Cepstrum
- Array of Structs and Struct of Arrays: https://hwisnu.bearblog.dev/array-of-structs-and-struct-of-arrays/
- FAISS IndexFlatL2 docs: https://faiss.ai/cpp_api/struct/structfaiss_1_1IndexFlatL2.html
- Pinecone, "Nearest Neighbor Indexes for Similarity Search": https://www.pinecone.io/learn/series/faiss/vector-indexes/
- Box-Muller transform — Wikipedia: https://en.wikipedia.org/wiki/Box%E2%80%93Muller_transform
- Box Muller Transform — Algorithm Archive: https://www.algorithm-archive.org/contents/box_muller/box_muller.html
- Cholesky decomposition intuition: https://alexander-pastukhov.github.io/notes-on-statistics/advanced-04-cholesky.html
- The Math Behind "The Curse of Dimensionality": https://towardsdatascience.com/the-math-behind-the-curse-of-dimensionality-cf8780307d74/
