#include "forest.h"
#include <stdlib.h>

RPForest rpforest_build(const Dataset *ds, size_t num_trees, size_t max_leaf_size, size_t max_depth, uint64_t initstate) {
    RPForest forest = {NULL, 0};

    if (num_trees == 0) {
        return forest;
    }

    forest.trees = malloc(num_trees * sizeof(RPTree));
    if (forest.trees == NULL) {
        return forest;
    }

    for (size_t i = 0; i < num_trees; i++) {
        forest.trees[i] = rptree_build(ds, max_leaf_size, max_depth, initstate, i);
        if (forest.trees[i].root == NULL) {
            forest.count = i;
            rpforest_free(&forest);
            return (RPForest){NULL, 0};
        }
    }

    forest.count = num_trees;
    return forest;
}

void rpforest_free(RPForest *forest) {
    if (forest->trees == NULL) {
        return;
    }
    for (size_t i = 0; i < forest->count; i++) {
        rptree_free(&forest->trees[i]);
    }
    free(forest->trees);
    forest->trees = NULL;
    forest->count = 0;
}