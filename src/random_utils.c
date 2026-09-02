#include "random_utils.h"
#include <math.h>
#include <stdlib.h>

#define PI 3.14159265358979323846

double uniform_random(void) {
    return rand() / (RAND_MAX + 1.0);
}

double gaussian_random(void) {
    double u1 = uniform_random();
    double u2 = uniform_random();

    while (u1 <= 0.0) {
        u1 = uniform_random();
    }

    return sqrt(-2.0 * log(u1)) * cos(2.0 * PI * u2);
}