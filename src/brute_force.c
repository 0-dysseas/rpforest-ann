#include "brute_force.h"
#include <stdlib.h>

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

double recall_at_k(const RPSearchResult *approx, const RPSearchResult *exact, size_t k) {
    size_t hits = 0;
    for (size_t i = 0; i < approx->count; i++) {
        for (size_t j = 0; j < exact->count; j++) {
            if (approx->indices[i] == exact->indices[j]) {
                hits++;
                break;
            }
        }
    }
    return (double)hits / (double)k;
}
