#ifndef GENERATOR_H
#define GENERATOR_H

#include "dataset.h"
#include <stdint.h>

Dataset generate_dataset(size_t n, size_t dim, size_t k, uint64_t initstate, uint64_t initseq);

#endif