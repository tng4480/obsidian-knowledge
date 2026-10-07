---
tags: [bayes, confusion]
---
# Bayesian inference problems

## What I don't understand

I struggled with the difference between the likelihood and the posterior. The likelihood is a function of the
parameter with the data held fixed, and it does not have to integrate to one over the parameter. The posterior only
appears once I multiply by a prior and divide by the evidence. I kept treating the normalised Gaussian likelihood as if
it were already the posterior.

## Beta priors

The coin question gave a prior with two parameters, n0 and N0. It took me a while to see they act like pseudo-counts:
n0 imagined heads and N0 minus n0 imagined tails. After six heads and three tails the posterior just adds those counts, and
the MAP estimate is a weighted average of the prior mode and the sample fraction. Only a flat prior makes MAP equal
maximum likelihood.

## Noisy sensor

If the sensor noise variance goes to infinity the posterior for the depth falls back to the prior, which is
sensible. If it goes to zero the posterior sits at the measurement.
