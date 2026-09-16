// Phase 6 benchmark driver.
//
// Measures per-query latency (all three search methods: brute force,
// single tree, forest) and recall@k (tree/forest against brute force's
// exact result) across four sweeps:
//   latency_vs_n         - dataset size N varies, dimensionality fixed
//   latency_vs_dim       - dimensionality varies, N fixed
//   recall_vs_budget     - single tree's search_budget varies, N/dim/num_trees fixed
//   recall_vs_num_trees  - forest's num_trees varies, N/dim/search_budget fixed
//
// Design points (settled in chat before this was written):
//   - One "measurement" is many different held-out queries (drawn from
//     the same generator/distribution as the dataset, never inserted
//     into it), not the same query repeated. This matches the query
//     seeding convention already used in tests/test_search.c and
//     tests/test_forest.c (corpus initseq 1, tree-build initseq 2,
//     query initseq 3), reused here so results stay comparable.
//   - The same query set is used across all three methods for a given
//     (n, dim) pair, so the comparison is apples to apples.
//   - A short untimed warm-up runs before the timed loop, so CPU
//     frequency scaling / cold caches don't bias the first timed
//     queries (see CppCon 2015, Carruth, "Tuning C++").
//   - Timing uses clock_gettime(CLOCK_MONOTONIC, ...): immune to
//     wall-clock adjustments, appropriate for measuring elapsed
//     intervals (see Strange Loop, Tene, "How NOT to Measure Latency").
//   - No DoNotOptimize-style trick is needed: brute_force_knn,
//     rptree_search, and rpforest_search are defined in separately
//     compiled translation units and this file is built without LTO
//     (see Makefile), so the compiler cannot see into them or prove
//     their results are unused; each result is also fed directly into
//     the CSV write immediately after it's timed.
//   - Raw per-query results are written to data/benchmark_results.csv;
//     scripts/plot_results.py reads that file and produces the plots,
//     so re-plotting doesn't require re-running the benchmark.

#define _POSIX_C_SOURCE 200809L

#include "dataset.h"
#include "generator.h"
#include "tree.h"
#include "forest.h"
#include "brute_force.h"

#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define SEED 42u
#define CORPUS_SEQ 1u
#define TREE_SEQ 2u
#define QUERY_SEQ 3u

#define FACTORS 4
#define MAX_LEAF_SIZE 20
#define MAX_DEPTH 20
#define K 5
#define NUM_QUERIES 250
#define WARMUP_QUERIES 10

#define N_BASE 20000
#define DIM_BASE 32
#define NUM_TREES_BASE 8
#define SEARCH_BUDGET_BASE 200

static FILE *csv;

static double elapsed_ns(struct timespec start, struct timespec end) {
    return (double)(end.tv_sec - start.tv_sec) * 1e9 +
           (double)(end.tv_nsec - start.tv_nsec);
}

// Builds a fresh corpus and a fresh held-out query set at the given
// (n, dim), builds a single tree and a forest over the corpus, then
// times NUM_QUERIES queries against all three methods (brute force
// doubles as each query's exact ground truth) and appends one CSV row
// per (method, query) pair. sweep names which experiment this call
// belongs to, for the plotting script to select on.
static void run_config(const char *sweep, size_t n, size_t dim,
                        size_t num_trees, size_t search_budget) {
    Dataset ds = generate_dataset(n, dim, FACTORS, SEED, CORPUS_SEQ);
    Dataset queries = generate_dataset(NUM_QUERIES, dim, FACTORS, SEED, QUERY_SEQ);
    if (ds.data == NULL || queries.data == NULL) {
        fprintf(stderr, "run_config: generate_dataset failed (n=%zu dim=%zu)\n", n, dim);
        exit(1);
    }

    RPTree tree = rptree_build(&ds, MAX_LEAF_SIZE, MAX_DEPTH, SEED, TREE_SEQ);
    RPForest forest = rpforest_build(&ds, num_trees, MAX_LEAF_SIZE, MAX_DEPTH, SEED);
    if (tree.root == NULL || forest.trees == NULL) {
        fprintf(stderr, "run_config: build failed (n=%zu dim=%zu)\n", n, dim);
        exit(1);
    }

    for (size_t w = 0; w < WARMUP_QUERIES && w < NUM_QUERIES; w++) {
        const float *query = dataset_at(&queries, w);
        RPSearchResult r1 = brute_force_knn(&ds, query, K);
        RPSearchResult r2 = rptree_search(&tree, &ds, query, K, search_budget);
        RPSearchResult r3 = rpforest_search(&forest, &ds, query, K, search_budget);
        rptree_search_free(&r1);
        rptree_search_free(&r2);
        rptree_search_free(&r3);
    }

    for (size_t q = 0; q < NUM_QUERIES; q++) {
        const float *query = dataset_at(&queries, q);
        struct timespec t0, t1;

        clock_gettime(CLOCK_MONOTONIC, &t0);
        RPSearchResult exact = brute_force_knn(&ds, query, K);
        clock_gettime(CLOCK_MONOTONIC, &t1);
        fprintf(csv, "%s,brute_force,%zu,%zu,,,%zu,%.1f,1.000\n",
                sweep, n, dim, q, elapsed_ns(t0, t1));

        clock_gettime(CLOCK_MONOTONIC, &t0);
        RPSearchResult tree_result = rptree_search(&tree, &ds, query, K, search_budget);
        clock_gettime(CLOCK_MONOTONIC, &t1);
        double tree_recall = recall_at_k(&tree_result, &exact, K);
        fprintf(csv, "%s,single_tree,%zu,%zu,,%zu,%zu,%.1f,%.3f\n",
                sweep, n, dim, search_budget, q, elapsed_ns(t0, t1), tree_recall);

        clock_gettime(CLOCK_MONOTONIC, &t0);
        RPSearchResult forest_result = rpforest_search(&forest, &ds, query, K, search_budget);
        clock_gettime(CLOCK_MONOTONIC, &t1);
        double forest_recall = recall_at_k(&forest_result, &exact, K);
        fprintf(csv, "%s,forest,%zu,%zu,%zu,%zu,%zu,%.1f,%.3f\n",
                sweep, n, dim, num_trees, search_budget, q, elapsed_ns(t0, t1), forest_recall);

        rptree_search_free(&exact);
        rptree_search_free(&tree_result);
        rptree_search_free(&forest_result);
    }

    dataset_free(&queries);
    rpforest_free(&forest);
    rptree_free(&tree);
    dataset_free(&ds);
}

int main(void) {
    csv = fopen("data/benchmark_results.csv", "w");
    if (csv == NULL) {
        perror("fopen data/benchmark_results.csv");
        return 1;
    }
    fprintf(csv, "sweep,method,n,dim,num_trees,search_budget,query_index,latency_ns,recall_at_5\n");

    size_t n_values[] = {1000, 2000, 5000, 10000, 20000, 50000, 100000};
    for (size_t i = 0; i < sizeof(n_values) / sizeof(n_values[0]); i++) {
        printf("latency_vs_n: n=%zu\n", n_values[i]);
        run_config("latency_vs_n", n_values[i], DIM_BASE, NUM_TREES_BASE, SEARCH_BUDGET_BASE);
    }

    size_t dim_values[] = {8, 16, 32, 64, 128, 256};
    for (size_t i = 0; i < sizeof(dim_values) / sizeof(dim_values[0]); i++) {
        printf("latency_vs_dim: dim=%zu\n", dim_values[i]);
        run_config("latency_vs_dim", N_BASE, dim_values[i], NUM_TREES_BASE, SEARCH_BUDGET_BASE);
    }

    size_t budget_values[] = {20, 50, 100, 200, 500, 1000};
    for (size_t i = 0; i < sizeof(budget_values) / sizeof(budget_values[0]); i++) {
        printf("recall_vs_budget: search_budget=%zu\n", budget_values[i]);
        run_config("recall_vs_budget", N_BASE, DIM_BASE, NUM_TREES_BASE, budget_values[i]);
    }

    size_t tree_values[] = {1, 2, 4, 8, 16, 32};
    for (size_t i = 0; i < sizeof(tree_values) / sizeof(tree_values[0]); i++) {
        printf("recall_vs_num_trees: num_trees=%zu\n", tree_values[i]);
        run_config("recall_vs_num_trees", N_BASE, DIM_BASE, tree_values[i], SEARCH_BUDGET_BASE);
    }

    fclose(csv);
    printf("done, results written to data/benchmark_results.csv\n");
    return 0;
}
