#include "generator.h"
#include "random_utils.h"
#include <math.h>
#include <stdlib.h>

// controls how quickly the per-dimention decay falls off.
// see formula A(t)=A_0*exp(-1/τ)
#define DECAY_TAU_DIVISOR 3.0

Dataset generate_dataset(size_t n, size_t dim, size_t k) {
    Dataset ds = dataset_create(n, dim);
    if (ds.data == NULL) {
        return ds;
    }

    double *loadings = malloc(dim * k * sizeof(double));

    double *decay = malloc(dim * sizeof(double));

    double *factors = malloc(k * sizeof(double));

    if (loadings == NULL || decay == NULL || factors == NULL) {
        free(loadings);
        free(decay);
        free(factors);
        dataset_free(&ds);
        return ds;
    }

    for (size_t j = 0; j < dim; j++) {
        for (size_t f = 0; f < k; f++) {
            loadings[j * k + f] = gaussian_random();
        }
        decay[j] = exp(-(double)j / ((double)dim / DECAY_TAU_DIVISOR));
    }

    for (size_t i = 0; i < n; i ++) {
        for (size_t f = 0; f < k; f++) {
            factors[f] = gaussian_random();
        }
    
        float *vec = dataset_at(&ds, i);
        for (size_t j = 0; j < dim; j++) {
            double signal = 0.0;
            for (size_t f = 0; f < k; f++) {
                signal += loadings[j * k + f] * factors[f];
            }
            double noise = gaussian_random();
            vec[j] = (float)(decay[j] * (signal + noise));
        }
    }

    free(loadings);
    free(decay);
    free(factors);
    dataset_free(&ds);
    return ds;    
}