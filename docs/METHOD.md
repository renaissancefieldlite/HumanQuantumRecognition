# Method

This repo scores interaction coupling between two traces.

In simulation mode those traces are modeled. In hardware-derived mode one trace
is replaced by a coherence proxy generated from calibration-style device
parameters.

In backend-capture mode the device-side trace is built from ordered FEZ session
artifacts using target-subspace probability as the backend coherence proxy.
That gives the repo a real execution rung without overstating it as direct
human-device recognition proof.

The score can support statements such as:

- aligned traces score higher than misaligned traces in this model
- the score remains usable under realistic noise/drift assumptions
- the score remains directional when the device-side trace is replaced by a
  real backend-fed session series

It cannot by itself establish that a real human and a real quantum device have
entered mutual recognition.
