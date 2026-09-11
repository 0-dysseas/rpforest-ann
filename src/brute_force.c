#include "brute_force.h"
#include <stdlib.h>

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

RPSearchResult brute_force_knn(const Dataset *ds, const float *query, size_t k) {
    RPSearchResult failure = {NULL, NULL, 0};

    Candidate *scored = malloc(ds->n * sizeof(Candidate));
    if (scored == NULL) {
        return failure;
    }
    for (size_t i = 0; i < ds->n; i++) {
        scored[i].index = i;
        scored[i].distance = dataset_squared_distance(ds, query, i);
    }
    qsort(scored, ds->n, sizeof(Candidate), compare_candidates);

    size_t result_count = (k < ds->n) ? k : ds->n;

    RPSearchResult result;
    result.indices = malloc(result_count * sizeof(size_t));
    result.distances = malloc(result_count * sizeof(float));
}