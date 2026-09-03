#include "dataset.h"
#include "generator.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>

static double dimension_mean(const Dataset *ds, size_t j) {
    double sum = 0.0;
    for (size_t i = 0; i < ds->n; i++) {
        sum += dataset_at(ds, i)[j];
    }
    return sum / (double)ds->n;
}

static double dimension_variance(const Dataset *ds, size_t j, double mean) {
    double sum_sq = 0.0;
    for (size_t i = 0; i < ds->n; i++) {
        double diff = dataset_at(ds, i)[j] - mean;
        sum_sq += diff * diff;
    }
    return sum_sq / (double)ds->n;
}

static double dimension_correlation(const Dataset *ds, size_t j1, double mean1,
                                     size_t j2, double mean2) {
    double cov_sum = 0.0;
    double var1_sum = 0.0;
    double var2_sum = 0.0;
    for (size_t i = 0; i < ds->n; i++) {
        double d1 = dataset_at(ds, i)[j1] - mean1;
        double d2 = dataset_at(ds, i)[j2] - mean2;
        cov_sum += d1 * d2;
        var1_sum += d1 * d1;
        var2_sum += d2 * d2;
    }
    return cov_sum / sqrt(var1_sum * var2_sum);
}

int main(void) {
    srand(42);

    size_t n = 2000;
    size_t dim = 20;
    size_t k = 4;

    Dataset ds = generate_dataset(n, dim, k);
    if (ds.data == NULL) {
        fprintf(stderr, "generate_dataset failed\n");
        return 1;
    }

    double *means = malloc(dim * sizeof(double));
    double *variances = malloc(dim * sizeof(double));
    if (means == NULL || variances == NULL) {
        fprintf(stderr, "allocation failed\n");
        free(means);
        free(variances);
        dataset_free(&ds);
        return 1;
    }

    for (size_t j = 0; j < dim; j++) {
        means[j] = dimension_mean(&ds, j);
        variances[j] = dimension_variance(&ds, j, means[j]);
    }

    printf("per-dimension variance (expect a decreasing trend):\n");
    for (size_t j = 0; j < dim; j++) {
        printf("  dim %2zu: variance = %f\n", j, variances[j]);
    }

    size_t corr_dims = dim < 6 ? dim : 6;
    printf("\ncorrelation matrix, first %zu dimensions (expect some clearly nonzero off-diagonal values):\n", corr_dims);
    printf("      ");
    for (size_t c = 0; c < corr_dims; c++) {
        printf(" dim%2zu", c);
    }
    printf("\n");
    for (size_t r = 0; r < corr_dims; r++) {
        printf("dim%2zu:", r);
        for (size_t c = 0; c < corr_dims; c++) {
            double corr = dimension_correlation(&ds, r, means[r], c, means[c]);
            printf(" %6.3f", corr);
        }
        printf("\n");
    }

    free(means);
    free(variances);
    dataset_free(&ds);
    return 0;
}