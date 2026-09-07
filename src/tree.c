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

static float squared_distance(const Dataset *ds, const float *query, size_t idx) {
    const float *point = dataset_at(ds, idx);
    float sum = 0.0f;
    for (size_t d = 0; d < ds->dim; d++) {
        float diff = query[d] - point[d];
        sum += diff * diff;
    }
    return sum;
}

typedef struct {
    size_t index;
    float distance;
} Candidate;

static int compare_candidates(const void *a, const void *b) {
    float da = ((const Candidate *)a)->distance;
    float db = ((const Candidate *)b)->distance;
    if (da < db) return -1;
    if (da > db) return 1;
    return 0;
}

static size_t *collect_candidates(const RPTree *tree, const Dataset *ds, const float *query, size_t search_budget, size_t *out_count) {
    *out_count = 0;

    if (tree->root == NULL) {
        return NULL;
    }

    size_t capacity = (search_budget > 0) ? search_budget : 1;
    size_t *candidates = malloc(capacity * sizeof(size_t));
    if (candidates == NULL) {
        return NULL;
    }

    PQueue pq = pqueue_create(8);
    if (pq.entries == NULL) {
        free(candidates);
        return NULL;
    }

    RPNode *node = tree->root;
    float priority = FLT_MAX;
    size_t count = 0;

    for (;;) {
        while (!node->is_leaf) {
            float dot = dot_product(query, node->normal, ds->dim);
            float margin = dot - node->threshold;
            float abs_margin = fabsf(margin);
            if (abs_margin < priority) {
                priority = abs_margin;
            }

            RPNode *near = (margin < 0.0f) ? node->left : node->right;
            RPNode *far = (margin < 0.0f) ? node->right : node->left;

            if (!pqueue_push(&pq, far, priority)) {
                pqueue_free(&pq);
                free(candidates);
                return NULL;
            }
            node = near;
        }

        if (count + node->count > capacity) {
            size_t new_capacity = capacity * 2;
            while (new_capacity < count + node->count) {
                new_capacity *= 2;
            }
            size_t *grown = realloc(candidates, new_capacity * sizeof(size_t));
            if (grown == NULL) {
                pqueue_free(&pq);
                free(candidates);
                return NULL;
            }
            candidates = grown;
            capacity = new_capacity;
        }
        for (size_t i = 0; i < node->count; i++) {
            candidates[count++] = node->indices[i];
        }

        if (count >= search_budget || pqueue_is_empty(&pq)) {
            break;
        }
        pqueue_pop(&pq, &node, &priority);
    }

    pqueue_free(&pq);
    *out_count = count;
    return candidates;
}

RPSearchResult rptree_search(const RPTree *tree, const Dataset *ds, const float *query, size_t k, size_t search_budget) {
    RPSearchResult failure = {NULL, NULL, 0};

    size_t candidate_count = 0;
    size_t *candidates = collect_candidates(tree, ds, query, search_budget, &candidate_count);
    if (candidates == NULL) {
        return failure;
    }

    Candidate *scored = malloc(candidate_count * sizeof(Candidate));
    if (scored == NULL) {
        free(candidates);
        return failure;
    }
    for (size_t i = 0; i < candidate_count; i++) {
        scored[i].index = candidates[i];
        scored[i].distance = squared_distance(ds, query, candidates[i]);
    }
    free(candidates);

    qsort(scored, candidate_count, sizeof(Candidate), compare_candidates);

    size_t result_count = (k < candidate_count) ? k : candidate_count;

    RPSearchResult result;
    result.indices = malloc(result_count * sizeof(size_t));
    result.distances = malloc(result_count * sizeof(float));
    if (result.indices == NULL || result.distances == NULL) {
        free(result.indices);
        free(result.distances);
        free(scored);
        return failure;
    }
    for (size_t i = 0; i < result_count; i++) {
        result.indices[i] = scored[i].index;
        result.distances[i] = scored[i].distance;
    }
    result.count = result_count;

    free(scored);
    return result;
}

void rptree_search_free(RPSearchResult *result) {
    free(result->indices);
    free(result->distances);
    result->indices = NULL;
    result->distances = NULL;
    result->count = 0;    
}