# Method

This repo scores interaction coupling between two traces.

In simulation mode those traces are modeled. In hardware-derived mode one trace
is replaced by a coherence proxy generated from calibration-style device
parameters.

The score can support statements such as:

- aligned traces score higher than misaligned traces in this model
- the score remains usable under realistic noise/drift assumptions

It cannot by itself establish that a real human and a real quantum device have
entered mutual recognition.
