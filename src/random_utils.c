#include "random_utils.h"
#include <math.h>

#define PI 3.14159265358979323846

void pcg32_seed(Pcg32State *rng, uint64_t initstate, uint64_t initseq) {
    rng->state = 0U;
    rng->inc = (initseq << 1u) | 1u;
    pcg32_random(rng);
    rng->state += initstate;
    pcg32_random(rng);
}

uint32_t pcg32_random(Pcg32State *rng) {
    uint64_t oldstate = rng->state;
    rng->state = oldstate * 6364136223846793005ULL + rng->inc;
    uint32_t xorshifted = (uint32_t)(((oldstate >> 18u) ^ oldstate) >> 27u);
    uint32_t rot = (uint32_t)(oldstate >> 59u);
    return (xorshifted >> rot) | (xorshifted << ((-rot) & 31u));
}

double uniform_random(Pcg32State *rng) {
    return pcg32_random(rng) / 4294967296.0;
}

double gaussian_random(Pcg32State *rng) {
    double u1 = uniform_random(rng);
    double u2 = uniform_random(rng);

    while (u1 <= 0.0) {
        u1 = uniform_random(rng);
    }

    return sqrt(-2.0 * log(u1)) * cos(2.0 * PI * u2);
}