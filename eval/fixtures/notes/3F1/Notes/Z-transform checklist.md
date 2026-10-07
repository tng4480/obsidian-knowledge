---
tags: [checklist, z-transform]
---
# Z-transform checklist

## Before using the final value theorem

Look at the poles of (1 - z^-1) X(z) first. The expression can have a harmless-looking limit as z goes to 1 while
the sequence has no limit at all, for example x_k = (-1)^k, which has a pole at -1. Only apply the formula once every
pole other than a simple one at z = 1 is strictly inside the unit circle.

## BIBO stability is not about one input

A system whose impulse response settles to a constant is still unstable if there is an uncancelled pole at z = 1,
because a bounded step input makes the output grow without limit. Checking the impulse response alone is not a proof.

## Feedback sign

Read the sign at the summing junction. Positive feedback gives T = L / (1 - L), negative feedback gives
T = L / (1 + L). I lost marks last year for assuming negative feedback.
