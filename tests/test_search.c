#include "dataset.h"
#include "generator.h"
#include "tree.h"
#include "brute_force.h"

#include <stdio.h>

int main(void) {
    size_t n = 2000;
    size_t dim = 20;
    size_t factors = 4;
    size_t max_leaf_size = 20;
    size_t max_depth = 20;

    Dataset ds = generate_dataset(n, dim, factors, 42, 1);
    if (ds.data == NULL) {
        fprintf(stderr, "generate_dataset failed\n");
        return 1;
    }

    RPTree tree = rptree_build(&ds, max_leaf_size, max_depth, 42, 2);
    if (tree.root == NULL) {
        fprintf(stderr, "rptree_build failed\n");
        dataset_free(&ds);
        return 1;
    }

    // Self-match check: querying with a point's own vector must return that
    // same point as its own nearest neighbor, at distance 0. The routing
    // test search uses (dot_product against the stored normal, compared to
    // the stored threshold) is bit-identical to the one build used to place
    // the point, so the greedy walk always reaches the point's own leaf on
    // the very first descent, regardless of search_budget.
    size_t self_mismatches = 0;
    for (size_t i = 0; i < n; i++) {
        const float *query = dataset_at(&ds, i);
        RPSearchResult result = rptree_search(&tree, &ds, query, 1, max_leaf_size);
        if (result.count == 0 || result.indices[0] != i || result.distances[0] != 0.0f) {
            self_mismatches++;
        }
        rptree_search_free(&result);
    }
    printf("self-match check: %zu points did not retrieve themselves as their own nearest neighbor at distance 0 (expect 0)\n", self_mismatches);

    // Sorted-order check: a multi-result query's distances must be
    // non-decreasing.
    size_t out_of_order = 0;
    {
        const float *query = dataset_at(&ds, 0);
        RPSearchResult result = rptree_search(&tree, &ds, query, 10, 200);
        for (size_t i = 1; i < result.count; i++) {
            if (result.distances[i] < result.distances[i - 1]) {
                out_of_order++;
            }
        }
        rptree_search_free(&result);
    }
    printf("sorted-order check: %zu adjacent result pairs were out of order (expect 0)\n", out_of_order);

    // Recall@k check: for a batch of fresh query points drawn from the same
    // distribution as the dataset (not points already in it), compare the
    // tree's approximate top-k against the true top-k found by brute-force
    // linear scan.
    size_t num_queries = 50;
    size_t k = 5;
    size_t search_budget = 200;

    Dataset queries = generate_dataset(num_queries, dim, factors, 42, 3);
    if (queries.data == NULL) {
        fprintf(stderr, "generate_dataset (queries) failed\n");
        rptree_free(&tree);
        dataset_free(&ds);
        return 1;
    }

    double total_recall = 0.0;

    for (size_t q = 0; q < num_queries; q++) {
        const float *query = dataset_at(&queries, q);
        RPSearchResult exact = brute_force_knn(&ds, query, k);

        RPSearchResult result = rptree_search(&tree, &ds, query, k, search_budget);

        total_recall += recall_at_k(&result, &exact, k);

        rptree_search_free(&result);
        rptree_search_free(&exact);
    }

    printf("recall@%zu over %zu fresh queries (search_budget = %zu): %.3f average\n",
           k, num_queries, search_budget, total_recall / (double)num_queries);

    dataset_free(&queries);
    rptree_free(&tree);
    dataset_free(&ds);
    return 0;
}
