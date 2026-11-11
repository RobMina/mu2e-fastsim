# calofastsim data

## digi_response_run1a.npz

Table of the opt-in parametric digi response (`calofastsim.param_response`, approximation W7
`digi_response=parametric`; schema 1).

- **What it holds.** Per bin of visible crystal energy (60 log bins, 0.01-300 MeV), a pool of at most 4,000 outcomes
  of the full device chain per (event, crystal) cell well inside the digitisation window: hit or not, hit energy /
  visible energy, hit time minus the energy-weighted folded step time (the cell's highest-energy hit). Also a
  survival curve in 2 ns bins of the cell time minus the event's window end (cells above 3 MeV). Hits stop between
  80 and 70 ns before the end, where a pulse no longer fits in the waveform. Without that curve the losses, which
  are whole events, came out spread over every event.
- **How it was built.** `make_steps_chain`'s stages (photoelectrons, digitisation, template fit, hits), float64, with
  the production response (Sim_crystal, CsI-waveform-2020-02-12, ReadoutNoise, CsI-propag-2020-07-29, PE statistics,
  travel time and noise on, reco bunch-time smear off: the chain adds it per event). The input was CaloShowerSteps of
  ten CeEndpoint MDC2025ba training files (001430_00000001, 101, ..., 901): 28,265 events, 312,265 cells (281,080
  inside the window), 112,898 with a hit. Of those cells, 60 had a second hit, and no hit came from noise alone.
- **Disjoint from the evaluation.** The held-out evaluation files (001430_00001000-1400) were not used.
