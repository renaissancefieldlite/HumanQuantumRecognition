# Method

This repo scores interaction coupling between two traces.

In simulation mode those traces are modeled. In hardware-derived mode one trace
is replaced by a coherence proxy generated from calibration-style device
parameters.

The score supports statements such as:

- aligned traces score higher than misaligned traces in this model
- the score remains usable under realistic noise/drift assumptions

The next stronger step is real-session alignment work between a human trace and a device-side capture.
