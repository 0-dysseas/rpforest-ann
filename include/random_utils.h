#ifndef RANDOM_UTILS_H
#define RANDOM_UTILS_H

// return uniform random double in [0,1)
double uniform_random(void);

// retrun sample from standard normal distribution (mean 0, variance 1)
// using Box-Muller transform
double gaussian_random(void);

#endif 