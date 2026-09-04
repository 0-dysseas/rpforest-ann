# Design Notes

Design rationale per phase, updated as decisions are made.

## Phase 1: Vector and dataset representation

Each vector is D contiguous floats. Distance computation and hyperplane projection both need to walk all D dimensions of one vector together, so the vector itself is the unit that must stay contiguous, not one array per dimension across the dataset.

The dataset is stored as a single contiguous `malloc`'d block of `N * D` floats (row major: vector `i` occupies floats `[i*D, (i+1)*D)`), not as N separately allocated vectors behind an array of pointers. The dataset is built once by the generator and only read afterward, so the per-vector-malloc approach's main advantage (independent resize/free) is not needed. Its costs are real: pointer chasing to reach each vector, unpredictable placement in memory, extra malloc/free overhead. One contiguous block gives cache-friendly sequential access, relevant whenever a leaf bucket or the brute-force baseline scans multiple vectors, at no extra implementation cost over the alternative.

Dimensionality D is a runtime parameter, not a compile-time constant. Phase 6 benchmarks latency across multiple dimensionalities, so it cannot be fixed at build time.

## Phase 1: Synthetic dataset generation

Chose a low-rank shared-latent-factor model over generating a full covariance matrix and applying a Cholesky decomposition. Without real MFCC statistics to calibrate a covariance matrix against, Cholesky's extra statistical rigor does not translate into more realism here, while it adds implementation and numerical-stability risk (an arbitrarily chosen covariance matrix is not guaranteed positive semi-definite) to a supporting component rather than the core deliverable.

Mechanism: draw a small number of shared Gaussian latent factors (via a Box-Muller uniform-to-Gaussian transform), combine them through weights into each of the D output dimensions, add independent per-dimension noise. Because multiple output dimensions share the same underlying factors, they come out correlated. This gives the dataset the kind of low-dimensional manifold structure that random projection trees are built to exploit (Dasgupta and Freund, below), rather than uncorrelated high-dimensional noise.

Per-dimension variance decay (mimicking how real MFCC coefficient magnitude falls off with index) applied via a scaling factor decreasing with dimension index.

Noise is non-isotropic: each output dimension gets its own variance from the decay schedule rather than one shared variance across all dimensions. The isotropic case (equal variance in every dimension) is the special condition under which this model's maximum-likelihood solution reduces to PCA, and using it would also conflict with the per-dimension variance decay already decided above.

Exact factor count, weighting scheme, and decay schedule are implementation choices made while writing the generator.

## Phase 1: Implementation (Dataset struct)

`Dataset` holds a `float *data` pointer to the contiguous buffer, plus `size_t n` and `size_t dim`. `size_t` is used instead of `int`: it is unsigned (a negative count is not representable) and matches the type `malloc` and `sizeof` use.

`dataset_create` allocates `n * dim` floats and returns the struct by value. It does not check for integer overflow in `n * dim * sizeof(float)`. Not a real risk at this project's scale, left unchecked deliberately.

`dataset_free` frees the buffer and zeroes the struct's fields, so a stale `Dataset` fails loudly (via `dataset_at`'s assert, or a safe no-op `free(NULL)`) rather than causing silent memory corruption.

`dataset_at` returns a pointer to vector `i`'s data via `data + i * dim`, with an `assert(i < n)` bounds check. Asserts are compiled out if `NDEBUG` is defined. The project's Makefile does not define it in either build target, so the check stays active in both.

## Phase 1: Implementation (random utilities)

`include/random_utils.h` and `src/random_utils.c` provide `uniform_random` and `gaussian_random`, used by the generator for both the factor draws and the per-dimension noise.

`gaussian_random` uses the Box-Muller transform: two independent uniform values in (0, 1) produce two independent standard normal values. Only one of the two is used per call, the second discarded. A known inefficiency, acceptable at this project's scale.

Both functions return `double`, not `float`. The intermediate math (`log`, `sqrt`, `cos`) is more numerically stable in double precision. The generator casts down to `float` only when writing the final value into the `Dataset` buffer.

`PI` is defined locally rather than using `M_PI`. `M_PI` is a POSIX/glibc extension, not part of ISO C, and is not guaranteed to be declared under the project's `-std=c11` strict mode.

The Makefile's `LDFLAGS` now includes `-lm`, required to link `sqrt`, `log`, and `cos` from the math library.

## Phase 1: Implementation (Generator)

`include/generator.h` and `src/generator.c` provide `generate_dataset(n, dim, k)`, implementing the shared-latent-factor model above.

The loading matrix L (dim by k) is stored the same way the Dataset itself is: one flat `malloc`'d block of `dim * k` doubles, indexed by hand as `loadings[j * k + f]`, rather than an array of pointers. Same reasoning as the Dataset struct: one allocation, contiguous, no pointer chasing.

L and the per-dimension decay schedule are computed once, before any vector is generated, and shared by all n vectors. For each vector: a fresh k-length factor draw, a matrix-vector product against L giving the raw dim-length signal, fresh independent noise added to it, and the sum scaled by the decay schedule.

The decay multiplication is a single trick doing two jobs at once. Rather than tracking a separate noise-variance schedule in addition to the magnitude-decay schedule, the same per-dimension decay factor is applied at the end, to the signal and the noise together. Because both are scaled by the same shrinking factor for a given dimension, the noise ends up smaller in exactly the dimensions where the signal is smaller, which is the non-isotropic property decided above, without a second array to maintain. This final multiplication is an addition on top of the textbook factor analysis equation X = LF + ε, not part of the original model.

The nested loops that build each vector run in O(n * dim * k) total: n vectors, each needing dim output values, each of which requires summing k terms. This is the minimum work the computation can take, not an accidental complexity blowup; producing n * dim numbers that each require a k-term sum cannot be done in less than n * dim * k operations.

Neither this module nor random_utils calls `srand()`. Seeding is left to the caller (main, once wired up), since a library function reseeding on every call would be wrong, and calling it more than once per program run would defeat its purpose.

## Phase 1: Empirical verification

`tests/test_generator.c` checks the two structural properties the shared-latent-factor design claims to produce: decaying per-dimension variance and nonzero correlation between dimensions.

Variance measured across a generated dataset (D = 20) fell from 3.455 at dimension 0 to 0.013 at dimension 19, a drop of about 270x. This matches the decay schedule's prediction: variance scales with the square of the per-dimension decay factor, and the ratio of the last decay factor to the first, squared, comes out close to the measured ratio. The trend is not strictly monotonic (dimension 2 measured higher than dimensions 0 and 1, for instance), which is expected: each dimension's row in the loading matrix L is an independent random draw, so the smooth decay envelope carries per-dimension noise on top of it.

The correlation matrix over the first six dimensions showed a diagonal of exactly 1.000 (each dimension perfectly correlated with itself, confirming the correlation computation itself) and clearly nonzero off-diagonal entries in both directions, for example 0.632 between dimensions 0 and 3, and -0.605 between dimensions 1 and 2. Negative correlations are expected under this model: two dimensions sharing a latent factor with opposite-signed weights on it move in opposite directions, and L's entries are zero-centered, so this happens routinely. A few pairs (dimension 0 and 5, at 0.027) came out close to zero, also expected since not every dimension pair shares much factor weight by chance.

Both properties together confirm the generator produces data with genuine low-dimensional structure to exploit, rather than uncorrelated high-dimensional noise. This is the premise the rest of the project, tree splits that find structure-aligned hyperplanes, depends on. Phase 1 is closed out on this basis.

## Sources consulted (Phase 1)

- Erik Bernhardsson, "Nearest neighbor methods and vector models", part 1: https://erikbern.com/2015/09/24/nearest-neighbor-methods-vector-models-part-1.html
- Erik Bernhardsson, part 2: https://erikbern.com/2015/10/01/nearest-neighbors-and-vector-models-part-2-how-to-search-in-high-dimensional-spaces.html
- Dasgupta and Freund, "Random projection trees and low dimensional manifolds" (STOC 2008): https://cseweb.ucsd.edu/~dasgupta/papers/rptree-stoc.pdf
- spotify/annoy source: https://github.com/spotify/annoy
- MFCC tutorial (Practical Cryptography): http://practicalcryptography.com/miscellaneous/machine-learning/guide-mel-frequency-cepstral-coefficients-mfccs/
- Cepstrum, Wikipedia: https://en.wikipedia.org/wiki/Cepstrum
- Array of Structs and Struct of Arrays: https://hwisnu.bearblog.dev/array-of-structs-and-struct-of-arrays/
- FAISS IndexFlatL2 docs: https://faiss.ai/cpp_api/struct/structfaiss_1_1IndexFlatL2.html
- Pinecone, "Nearest Neighbor Indexes for Similarity Search": https://www.pinecone.io/learn/series/faiss/vector-indexes/
- Box-Muller transform, Wikipedia: https://en.wikipedia.org/wiki/Box%E2%80%93Muller_transform
- Box Muller Transform, Algorithm Archive: https://www.algorithm-archive.org/contents/box_muller/box_muller.html
- Cholesky decomposition intuition: https://alexander-pastukhov.github.io/notes-on-statistics/advanced-04-cholesky.html
- The Math Behind the Curse of Dimensionality: https://towardsdatascience.com/the-math-behind-the-curse-of-dimensionality-cf8780307d74/
- Factor analysis, Wikipedia: https://en.wikipedia.org/wiki/Factor_analysis
- FactorAnalysis, scikit-learn documentation: https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.FactorAnalysis.html
- Random Projection, scikit-learn user guide: https://scikit-learn.org/stable/modules/random_projection.html

## Phase 2: Node representation

A random projection tree node is one of two kinds: an internal node, which only decides which of two children a point belongs to, and a leaf, which holds the actual points that landed in that region. `RPNode` represents both with a single struct and an `is_leaf` tag rather than two separate struct types (one for internal nodes, one for leaves) behind a discriminated pointer or a union. All fields for both cases exist on every node, wasting some memory (a leaf carries unused `normal`/`threshold`/`left`/`right` fields, an internal node carries unused `indices`/`count`), which a union would avoid. That waste is accepted for now: compact, cache-aware node layout is its own later phase (roadmap item 7), and doing it now would mean optimizing a data structure whose access patterns have not been measured yet.

## Phase 2: Implementation (RPNode struct)

`is_leaf` is a plain `int`, not `bool` from `<stdbool.h>`. Neither `Dataset`, `random_utils`, nor `generator` use `<stdbool.h>` anywhere in the project so far, so `int` keeps the new struct consistent with the rest of the codebase's existing style rather than introducing a second convention for the same 0/1 flag.

`RPNode`'s internal-node fields are `float *normal` and `float threshold` (the hyperplane's coefficients, derived from two random pivot points, see the split-rule algebra worked out in chat) plus `struct RPNode *left` and `struct RPNode *right`. Its leaf fields are `size_t *indices` and `size_t count`.

The struct is declared as `typedef struct RPNode { ... } RPNode;`, naming the struct twice: `RPNode` immediately after `struct` is the struct tag, `RPNode` after the closing brace is the typedef alias. The tag is required here specifically because the struct is self-referential (`left` and `right` are pointers to the same struct type), and inside the struct body the typedef name does not exist yet, only the tag does. This is the standard C idiom for any self-referential struct (tree nodes, linked list nodes).

## Phase 2: Implementation (RPTree struct)

The recursive build (not yet written) partitions one flat array of point indices in place: the root call gets the full array, and each split hands its two children pointers into two contiguous sub-ranges of that same array, not separately allocated copies. A leaf's `indices` pointer is therefore frequently not the address `malloc` originally returned, it can be offset into the middle of that block. Freeing such a pointer directly is undefined behavior.

`RPTree` exists to make this safe: it wraps the tree's `root` together with the single flat `indices` array the whole tree is built from, and owns that array exclusively. The array is allocated once, when the tree is built, and freed exactly once, when the tree is freed. Individual `RPNode` leaves never own or free their `indices` pointer, they only reference into the array `RPTree` owns.
