---
tags: [entropy, confusion]
---
# Entropy: things that tripped me up

## What I don't understand

I keep mixing up H(X|Y) with the entropy of an averaged conditional distribution. The worked example computes the
entropy for each value of Y first and only then weights by P(Y). Why is averaging the probabilities before taking the
log wrong? I got 0.99 instead of 0.965 the first time because I did it in the wrong order.

I also never trust my chain rule bookkeeping. H(X,Y) = H(X) + H(Y|X) is easy, but when there are three variables I
lose track of which one is conditioned on which.

## Pairwise versus mutual independence

Two fair bits A and B, and C = A xor B. Every pair is independent but the three together are not, so the joint
entropy is 2 bits and not 3. My mistake in the last supervision was assuming pairwise independence was enough to add
entropies.

## Fano or Shannon-Fano?

The crib calls the length-ceiling code "Fano" and quotes an expected length of 3.25. The textbook recursive
Shannon-Fano splitting gives a different number for the same probabilities. Check which construction a question means
before deciding an answer is wrong.
