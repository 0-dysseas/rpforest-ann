#include "tree.h"
#include "random_utils.h"
#include "pqueue.h"
#include <math.h>
#include <float.h>
#include <assert.h>
#include <stdlib.h>

static float dot_product(const float *vec, const float *normal, size_t dim) {
    float dot = 0.0f;
    for (size_t d = 0; d < dim; d++) {
        dot += vec[d] * normal[d];
    }
    return dot;
}

// Threshold = 1/2 * (||B||^2 - ||A||^2)
static void choose_split(const Dataset *ds, const size_t *indices, size_t count, float *normal, float *threshold) {
    assert(count >= 2);

    size_t pa = (size_t)(uniform_random() * count);
    size_t pb = (size_t)(uniform_random() * count);
    while ( pb == pa) {
        pb = (size_t)(uniform_random() * count);
    }

    const float *a = dataset_at(ds, indices[pa]);
    const float *b = dataset_at(ds, indices[pb]);

    double a_sq = 0.0;
    double b_sq = 0.0;
    for (size_t d = 0; d < ds->dim; d++) {
        normal[d] = b[d] - a[d];
        a_sq += (double)a[d] * a[d];
        b_sq += (double)b[d] * b[d];
    }

    *threshold = (float)((b_sq - a_sq) / 2.0);

}

static size_t partition_indices(const Dataset *ds, size_t *indices, size_t count, const float *normal, float threshold) {
    size_t left = 0;
    size_t right = count;

    while (left < right) {
        const float *vec = dataset_at(ds, indices[left]);
        float dot = dot_product(vec, normal, ds->dim);

        if (dot < threshold) {
            left++;
        } else {
            right--;
            size_t tmp = indices[left];
            indices[left] = indices[right];
            indices[right] = tmp;
        }
    }

    return left;
}

static RPNode *build_recursive(const Dataset *ds, size_t *indices, size_t count, size_t depth, size_t max_leaf_size, size_t max_depth) {
    RPNode *node = malloc(sizeof(RPNode));
    if (node == NULL) {
        return NULL;
    }

    if (count <= max_leaf_size || depth >= max_depth) {
        node->is_leaf = 1;
        node->indices = indices;
        node->count = count;
        node->normal = NULL;
        node->left = NULL;
        node->right = NULL;
        return node;
    }

    float *normal = malloc(ds->dim * sizeof(float));
    if (normal  == NULL) {
        free(node);
        return NULL;
    }

    float threshold;
    choose_split(ds, indices, count, normal, &threshold);
    size_t split = partition_indices(ds, indices, count, normal, threshold);

    node->is_leaf = 0;
    node->normal = normal;
    node->threshold = threshold;
    node->indices = NULL;
    node->count = 0;
    node->left = build_recursive(ds, indices, split, depth + 1, max_leaf_size, max_depth);
    node->right = build_recursive(ds, indices + split, count - split, depth + 1, max_leaf_size, max_depth);

    return node;
}

RPTree rptree_build(const Dataset *ds, size_t max_leaf_size, size_t max_depth) {
    RPTree tree;
    tree.indices = malloc(ds->n * sizeof(size_t));
    if (tree.indices == NULL) {
        tree.root = NULL;
        return tree;
    }

    for (size_t i = 0; i < ds->n; i++) {
        tree.indices[i] = i;
    }

    tree.root = build_recursive(ds, tree.indices, ds->n, 0, max_leaf_size, max_depth);
    return tree;
}

static void free_node(RPNode *node) {
    if (node == NULL) {
        return;
    }
    if (!node->is_leaf) {
        free(node->normal);
        free_node(node->left);
        free_node(node->right);
    }
    free(node);
}

void rptree_free(RPTree *tree) {
    free_node(tree->root);
    free(tree->indices);
    tree->root = NULL;
    tree->indices = NULL;
}

