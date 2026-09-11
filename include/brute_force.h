#ifndef BRUTE_FORCE_H
#define BRUTE_FORCE_H

#include "dataset.h"
#include "tree.h"

RPSearchResult brute_force_knn(const Dataset *ds, const float *query, size_t k);

double recall_at_k(const RPSearchResult *approx, const RPSearchResult *exact, size_t k);

#endif