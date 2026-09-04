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

RPTree rptree_build(const Dataset *ds, size_t max_leaf_size, size_t max_depth);

void rptree_free(RPTree *tree);

#endif