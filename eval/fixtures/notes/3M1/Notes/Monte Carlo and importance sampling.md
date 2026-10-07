---
tags: [monte-carlo, importance-sampling]
---
# Monte Carlo and importance sampling

## Basics

The plain sample mean has variance sigma squared over N, so the root-mean-square error shrinks like one over root N
regardless of dimension. The constant sigma squared can still be awful in high dimensions.

## Choosing the proposal

Importance weights are w = p / q, and q must be positive wherever |h| p is positive or contributions are never sampled.
The variance is better than plain Monte Carlo only if q is large where |h| p is large.

## My mistake with the sign

The crib says the best proposal is proportional to h times p. That is not a density when h changes sign. The proposal
that actually minimises the variance is proportional to the absolute value of h times p. If h has constant sign the
variance can be driven to zero. I should write the absolute value in the exam.
