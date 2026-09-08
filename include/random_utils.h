#ifndef RANDOM_UTILS_H
#define RANDOM_UTILS_H

#include <stdint.h>

typedef struct {
    uint64_t state;
    uint64_t inc;    
} Pcg32State;

// seed a PCG32 generator. initseq selects an independent output stream:
// two generators seedded with the same initstate but different initseq
// produce unrelated sequences
void pcg32_seed(Pcg32State *rng, uint64_t initstate, uint64_t initseq);

// returns a uniform random value across the full 32-bit range
uint32_t pcg32_random(Pcg32State *rng);

// return uniform random double in [0,1)
double uniform_random(Pcg32State *rng);

// retrun sample from standard normal distribution (mean 0, variance 1)
// using Box-Muller transform
double gaussian_random(Pcg32State *rng);

#endif 