#include "tree.h"
#include "random_utils.h"
#include <assert.h>
#include <stdlib.h>

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
        float dot = 0.0f;
        for (size_t d = 0; d < ds->dim; d++) {
            dot += vec[d] * normal[d];
        }

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