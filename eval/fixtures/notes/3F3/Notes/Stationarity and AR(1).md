---
tags: [stationarity, confusion]
---
# Stationarity and AR(1)

## What I don't understand

Wide-sense stationarity only asks for a constant mean and a correlation that depends on the lag. Strict stationarity
asks for every finite-dimensional distribution to be shift invariant. An iid process is both. I still cannot decide
quickly for a process like a random constant: it is WSS if the second moment is finite, but time averages do not
converge to the ensemble mean.

## AR(1) variance

X_n = a X_{n-1} + W_n with |a| < 1 has variance sigma squared over (1 - a squared), but only for the stationary solution
started in the infinite past. If I start from an arbitrary finite initial state the variance changes with time until it
reaches that limit. Correlation decays geometrically as a to the power of the lag.

## Estimating the parameter

Take a as the ratio of R[1] to R[0], then recover the innovation variance from R[0] (1 - a squared).
