#include "dataset.h"
#include "generator.h"
#include "tree.h"

#include <stdio.h>
#include <stdlib.h>

typedef struct {
    size_t leaf_count;
    size_t max_leaf_size_seen;
    size_t min_leaf_size_seen;
    size_t max_depth_seen;
    size_t total_points_in_leaves;
} TreeStats;

static void walk(const RPNode *node, size_t depth, size_t *seen, TreeStats *stats) {
    if (node == NULL) {
        return;
    }

    if (node->is_leaf) {
        stats->leaf_count++;
        stats->total_points_in_leaves += node->count;
        if (node->count > stats->max_leaf_size_seen) {
            stats->max_leaf_size_seen = node->count;
        }
        if (stats->leaf_count == 1 || node->count < stats->min_leaf_size_seen) {
            stats->min_leaf_size_seen = node->count;
        }
        if (depth > stats->max_depth_seen) {
            stats->max_depth_seen = depth;
        }
        for (size_t i = 0; i < node->count; i++) {
            seen[node->indices[i]]++;
        }
        return;
    }

    walk(node->left, depth + 1, seen, stats);
    walk(node->right, depth + 1, seen, stats);
}

// walks from the root using only the stored normal/threshold at each internal
// node, independent of how the tree was built, to see which leaf this point's
// own coordinates say it belongs in.
static const RPNode *find_leaf(const Dataset *ds, const RPNode *node, const float *point) {
    while (!node->is_leaf) {
        float dot = 0.0f;
        for (size_t d = 0; d < ds->dim; d++) {
            dot += point[d] * node->normal[d];
        }
        node = (dot < node->threshold) ? node->left : node->right;
    }
    return node;
}

static int leaf_contains(const RPNode *leaf, size_t target) {
    for (size_t i = 0; i < leaf->count; i++) {
        if (leaf->indices[i] == target) {
            return 1;
        }
    }
    return 0;
}

int main(void) {
    srand(42);

    size_t n = 2000;
    size_t dim = 20;
    size_t k = 4;
    size_t max_leaf_size = 20;
    size_t max_depth = 20;

    Dataset ds = generate_dataset(n, dim, k);
    if (ds.data == NULL) {
        fprintf(stderr, "generate_dataset failed\n");
        return 1;
    }

    RPTree tree = rptree_build(&ds, max_leaf_size, max_depth);
    if (tree.root == NULL) {
        fprintf(stderr, "rptree_build failed\n");
        dataset_free(&ds);
        return 1;
    }

    size_t *seen = calloc(n, sizeof(size_t));
    if (seen == NULL) {
        fprintf(stderr, "allocation failed\n");
        rptree_free(&tree);
        dataset_free(&ds);
        return 1;
    }

    TreeStats stats = {0};
    walk(tree.root, 0, seen, &stats);

    printf("tree structure:\n");
    printf("  leaves: %zu\n", stats.leaf_count);
    printf("  leaf size: min = %zu, max = %zu, average = %.2f (max_leaf_size = %zu)\n",
           stats.min_leaf_size_seen, stats.max_leaf_size_seen,
           (double)stats.total_points_in_leaves / (double)stats.leaf_count, max_leaf_size);
    printf("  max depth reached: %zu (max_depth = %zu)\n", stats.max_depth_seen, max_depth);
    printf("  total points across leaves: %zu (expect exactly %zu)\n", stats.total_points_in_leaves, n);

    size_t missing = 0, duplicated = 0;
    for (size_t i = 0; i < n; i++) {
        if (seen[i] == 0) {
            missing++;
        } else if (seen[i] > 1) {
            duplicated++;
        }
    }
    printf("\npartition check: %zu points missing from all leaves, %zu points duplicated across leaves (expect 0 and 0)\n",
           missing, duplicated);

    size_t mismatches = 0;
    for (size_t i = 0; i < n; i++) {
        const float *point = dataset_at(&ds, i);
        const RPNode *leaf = find_leaf(&ds, tree.root, point);
        if (!leaf_contains(leaf, i)) {
            mismatches++;
        }
    }
    printf("split consistency check: %zu points landed in a leaf inconsistent with their own tree's split decisions (expect 0)\n",
           mismatches);

    free(seen);
    rptree_free(&tree);
    dataset_free(&ds);
    return 0;
}
