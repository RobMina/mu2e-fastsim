# Named-reference grid certification

The Run1A and Run2 named-reference G4 samples were extended on the grid by a bounded, one-off controller (not kept in
the repository): pilots first, then the remainder only after successful copy-out and persisted-count certification.
Dumps read art directly from its dCache location; outputs went to the configured data area. Missing or failed evidence
means stopped, not successful completion.

## Certification-only example

```bash
python3 -m crvfastsim.analyses.reference_certification DUMPDIR \
  --first 102 --last 104 --expected-events 50000 --output NEW_REPORT.json
```

This checks completeness relative to successful worker counts and consistency of the persisted accepted
identities with generated events. It does not independently reconstruct the Geant4 acceptance decision,
certify detector-response physics, or establish per-sector precision.
