#include "dataset.h"
#include <assert.h>
#include <stdlib.h>

Dataset dataset_create(size_t n, size_t dim) {
    Dataset ds;
    ds.n = n;
    ds.dim = dim;
    ds.data = malloc(n * dim * sizeof(float));
    return ds;
}

void dataset_free(Dataset *ds) {
    free(ds->data);
    ds->data = NULL;
    ds->n = 0;
    ds->dim = 0;
}

float *dataset_at(const Dataset *ds, size_t i) {
    assert(i < ds->n);
    return ds->data + i * ds->dim;
}