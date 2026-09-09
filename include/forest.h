#ifndef FOREST_H
#define FOREST_H

#include "dataset.h"
#include "tree.h"
#include <stdint.h>

typedef struct {
    RPTree *trees;
    size_t count;
} RPForest;

RPForest rpforest_build(const Dataset *ds, size_t num_trees, size_t max_leaf_size, size_t max_depth, uint64_t initstate);

RPSearchResult rpforest_search(const RPForest *forest, const Dataset *ds, const float *query, size_t k, size_t search_budget);

void rpforest_free(RPForest *forest);

#endif