# Parked code

Code removed from `main` because nothing used it any more, or something else replaced it. Each item is kept on a
`park/<name>` branch cut from a commit that still has it. To bring an item back, restore its files from that branch
(`git checkout park/<name> -- <paths>`) and open a PR with its tests.

| item | branch | removed because | to revive |
|---|---|---|---|
| CE-only response table with pools: `tracker/trkfastsim/recoresp/data/ce_loophelix_run1a_pools.npz` (+ manifest), names `"pools"` / `"ceonly_pools"` in `recoresp.table.NAMED`, `table.POOLS` | `park/recoresp-ceonly-pools` | Superseded: the default full-event table (`"fulldts"`) carries pools. The CE-only steps miss the full events' secondaries (+2.84 pp CE efficiency against them). Only its own tests loaded it. The CE-only copula table `"ceonly"` stays. | Restore the npz, its manifest, the `POOLS` constant, the `NAMED` entries and the `SHIPPED` manifest entry in `recoresp/table.py`, and the tests that load `RT.POOLS` |
| Contact-disk-only calorimeter match: `calo_match(rule="contact")` in `beam/beamfastsim/centuple/assemble.py` | `park/centuple-contact-match` | Superseded by the stock-like rule (either disk, stock inclusion efficiency), which has been the default since the stock match was added. The command line never exposed the old rule, and only its own test called it. Cut-set-80 efficiency against stock 0.559: stock-like rule 0.548 (float32) / 0.545 (float64), contact rule 0.519 / 0.522. | Restore the `rule` argument and its branch in `calo_match`, and the test lines in `beam/tests/test_centuple.py` |
