---
tags: [bjt, confusion]
---
# Common-emitter stage and bias current

## What I don't understand

For the grounded-emitter question with a 20 V supply and a 300k/10k base divider I wrote down a gain of -400 straight
away, copying the crib. But nothing in the question tells me the quiescent collector current. The crib silently
assumes the collector sits at 10 V. If the gain is -gm Rc and gm = Ic / VT, then the gain moves with Ic. So the "answer"
is really a family of answers and I should state the assumption.

## Why VBE of about 0.6 V does not pin the current

The collector current is exponential in VBE, so a tenth of a volt of uncertainty changes Ic by a factor of fifty. A
fixed-VBE estimate and the crib's 10 V collector assumption give different input resistances and gains. Say which
model I am using.

## Checklist

- DC first with capacitors open, then AC with capacitors shorted.
- Include the source resistance when I calculate the gain from the source to the load.
- Coupling capacitors see the sum of the resistances on both sides.
