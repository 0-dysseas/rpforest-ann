#ifndef DATASET_H
#define DATASET_H

#include <stddef.h>

typedef struct {
    float *data;
    size_t n;
    size_t dim;
} Dataset;

Dataset dataset_create(size_t n, size_t dim);

void dataset_free(Dataset *ds);

// returns a pointer at the start of vector i's data in the dataset
float *dataset_at(const Dataset *ds, size_t i);

#endif