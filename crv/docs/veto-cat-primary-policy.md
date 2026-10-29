# Staged Cat versus retained-primary donor association policies

## The contracts are different

`SimParticle::isPrimary()` in the supported Offline releases is a nonnull
generator pointer plus `creationCode == mu2ePrimary`. In a staged beam sample it
need not identify the original particle represented by that generator. A retained
stage primary and its generator may therefore have different PDGs. Generator
**pointer identity** and a **retained-original-primary model** are separate tests.

The `retained_primary` dumper policy remains the default and retains all its
checks, including signed-PDG equality for primaries. The cosmic shower library
still accepts only its schema-v2 retained-primary donor contract. No library
acceptance criterion is relaxed or bypassed for those inputs.

`dump_crvsteps_cat.fcl` explicitly requests `associationPolicy: "staged_cat"`.
This policy still checks:

- every step and nonnull parent pointer against selected SimParticle ProductID,
  available product and valid key;
- every nonnull generator pointer against selected GenParticle ProductID,
  available product, in-range key and referenced-versus-selected generator PDG;
- finite generator energy/time and integer narrowing limits.

It does not require a stage primary's own PDG to equal its referenced generator's
PDG. It does not rewrite ancestry, PDGs or any other payload. It is not a generic
validation-off option. Unknown policies fail; `staged_cat` requires step extraction
and cannot declare a full-generator stream.

## Persisted separation

Every event records `association_policy`. Default retained-primary output stays
`schema_version=2`; explicit staged-Cat output is **schema_version=3**. The strict
`genassoc.validate_donor` gate rejects schema 3. Thus staged-Cat tables cannot be
silently passed to the retained-primary cosmic donor builder, even when a particular
file happens to contain equal PDGs. Existing Cat-v2 census conversion and pool
contracts remain distinct from the association schema; source/config/dumper hashes
must bind the new extraction. Nothing relabels a partial or historical dump.

## Status and required checks

The policy has been user-built and exercised in bounded Cat and cosmic-donor
checks. A complete single Cat source preserves the retained alias's legacy payload
and corrected census, while both policies still reject a wrong generator ProductID.
The default cosmic donor passes its strict gate with all previous branches unchanged.
This is not an all-inventory or response-closure certificate. Each new build/area
still needs its own binary and configuration verification; there are no new dependencies.

After user build confirmation, use new output tags to:

1. Verify the loaded binary and effective `associationPolicy` on a small Cat sample
   spanning a known primary/gen-PDG mismatch. Check schema 3 and policy on every
   event; zero mismatches of pointer ProductID/key remain mandatory.
2. Process the full intended single source and compare its all-event census with
   catalog count and finished job count. Do not infer success from a partial file.
3. Compare compact payload/full-tuple census against an existing alias, without
   rereading or replacing the completed baseline source. Cat-v2 must retain the
   safe denominator. Separately validate both aliases and compare canonical content;
   a joint pool must reject them. Real extraction-contract differences may trigger
   rejection before the duplicate-content guard: report that actual gate, never
   patch metadata merely to obtain a preferred rejection.
4. Smoke the default retained-primary cosmic donor with its strict gate unchanged;
   a deliberately wrong generator collection must still fail before payload fill.

The read-only diagnostic `veto.cat_primary_audit` can inspect a retained flat dump:

```bash
python3 -m crvfastsim.veto.cat_primary_audit "$EXISTING_DUMP" \
  --event RUN SUBRUN EVENT --out "$NEW_REPORT"
```

It joins by full event tuple and generator key, counts mismatch species/creation
codes, retains input/tool hashes, and checks ProductIDs when both tables provide
them. It explicitly reports the absence of ProductID evidence in legacy dumps;
it cannot certify their associations. No art physics is interpreted through uproot:
this command consumes the dumper's flat trees only.

## Runtime validation tools

`veto.census_policy` validates persisted policy/schema, full event census and every
step/parent/generator ProductID/key. It explicitly exercises the strict cosmic
consumer's acceptance or rejection. It can compare every reference branch and
dtype exactly (equal NaNs included), using either a full file or an explicitly
labelled stored-prefix diagnostic:

```bash
python3 -m crvfastsim.veto.census_policy --probe "$NEW_PROBE_DIR" \
  --policy staged_cat --reference "$EXISTING_ALIAS_DUMP" \
  --events "$EXPECTED_SOURCE_EVENTS" --out "$NEW_CHECK_JSON"
```

The probe directory contains `dump.root` and `job.log`; success requires the art
exit-zero message and a finished event count equal to the persisted census.
`--compact FIRST.parquet SECOND.parquet` validates two single-file Cat-v2 pools,
requires identical canonical census/content, and reports the actual joint-pool
rejection without altering either input. `--negative-guard product|primary`, with
`--probe` and `--events`, requires a recorded exit-one status, the specified guard
and the exact number of completed prefix rows; a ProductID negative control with
zero completed events must have no event/gen/particle/step rows.

`scripts/dump_veto_policy_probe.sh` runs one user-built regression with exact area,
FCL, source, event count, new output directory and expected exit status arguments.
It captures the effective FCL, binary/source hashes, command, loader trace and
explicit process status. It never builds, stages inputs, overwrites an output or
changes the selected work area. Use the three `probe_*` FCLs only as negative
controls, not as production configurations. Positive completion and expected
rejection are distinct checks; a partial negative-control dump is never usable
as a census-complete Cat table.
