#ifndef TREE_H
#define TREE_H

#include "dataset.h"

typedef struct RPNode {
    int is_leaf;

    // when node is not leaf (internal node)
    float *normal;
    float threshold;
    struct RPNode *left;
    struct RPNode *right;

    // when node is leaf
    size_t *indices;
    size_t count;
} RPNode;

typedef struct {
    RPNode *root;
    size_t *indices;
} RPTree;

typedef struct {
    size_t *indices;
    float *distances;
    size_t count;
} RPSearchResult;

RPTree rptree_build(const Dataset *ds, size_t max_leaf_size, size_t max_depth);

RPSearchResult rptree_search(const RPTree *tree, const Dataset *ds, const float *query, size_t k, size_t search_budget);

void rptree_free(RPTree *tree);

void rptree_search_free(RPSearchResult *result);

#endif