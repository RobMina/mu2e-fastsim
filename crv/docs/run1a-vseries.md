# Run 1A V-series (library, transport, end to end)

`crvfastsim.validation.run1a_vseries` compares the fast Run 1A chain with Geant4 and stock on held-out shards. The
arms share each shard's CRY events, so every error is the paired per-shard error. Per shard:

1. `transport GEN --stage C --run-config run1a_v01` writes the fast primary-muon steps (V1 compares them with the
   Geant4 primary-muon steps).
2. `tier1b-apply fastC --tier1b run1a_v01 --run-config run1a_v01` adds the library descendants and showers (V4
   compares the steps per origin and sector, and the steps per event, with all Geant4 steps).
3. `runchain fast1b --steps-input --save-reco e2e.npz --run-config run1a_v01 --tier1b run1a_v01 ...` runs the fast
   response and reco of those steps: the end-to-end arm.
4. The same on the Geant4 step table (`g4.npz`) is the response control. It separates the response and reco
   difference from the transport and library difference.
5. The stock reco of the Geant4 steps (`CrvRecoTableDumper`) is the reference.

```bash
python3 -m crvfastsim.validation.run1a_vseries --pairs pairs.json --out DIR --plots
```

`pairs.json` lists one row per shard: `shard`, `g4_steps`, `fast_c`, `fast_1b`, `stock_reco`, `fast_reco`,
`g4_reco`. Outputs:

- `vseries.json`: the ratios and shape statistics of V1, V4, the end-to-end arm, the response control, and the end-to-end
  arm against the control;
- `crv_pe.png`: cluster PE with pulls;
- `crv_layers.png`: layers per cluster, and pulses of at least 5 PE and clusters per sector.

Statistics:

- ratio of the sums, with error sqrt(n/(n-1) sum_i (f_i - R r_i)^2) / sum r;
- per bin, the summed difference over sqrt(n var_i(f_i - r_i)) (the pull);
- chi2/ndf, the mean squared pull over the bins with a nonzero error.

## Conditions and statistics

The fast response must use the reference's conditions. The Run 1A stock job reads the Offline text conditions
`CRVConditions/data/{calib,status,photonyield}_run1a_v01.txt` and the channel map `run1a_v01.txt`;
`scripts/offline_conditions_csv.py` converts them to the `--calib` / `--deviation` CSVs. `--calib nominal` treats every
channel as live and ignores the 45 channels with status bit 1 ("ignore in reconstruction"), which shows up as about
1 % more pulses than stock in both sectors. `runchain` therefore refuses `--calib nominal` for `run1a_v01` and
`run1b_v01` (`RunConfig.conditions`).

The reference release matters too: Offline before v13_40_00 shares one photon generator, and one scintillation yield,
between sectors with the same lookup table (Run 1A T2 then runs at T1's yield; Offline#1956).

Shape statistics merge tail bins until each reference bin holds 25 entries. The pull errors come from n shards, so
chi2/ndf has the null expectation (n - 1)/(n - 3), reported as `chi2_ndf_null`.
