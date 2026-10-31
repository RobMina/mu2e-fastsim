#!/bin/bash
# Run the held-out shower-source diagnostic or the unchanged-primary validation on saved step tables.
set -eo pipefail
D=${CRVFS_DATA:?set CRVFS_DATA to the CRV data root (see site.env.example)}
O=${STEP_LAYERS_OUT:?set STEP_LAYERS_OUT to the step-layer study directory}
F=$D/thread_a_ledger_20261025_c3895aae/extracted20/tables
G=$D/valhist/g4ref_v1/g4all_tables
L=$O/tier1b_ctx/lib_prod_as_20files_ctx.root
mkdir -p "$O/takeover"
case ${1:?choose source or guard} in
  source)
    train=(); gen=(); evaluation=()
    for k in $(seq 1 10); do
      s=$(printf '001400_%08d' "$k")
      train+=("$D/g4ref_v1_tables/crvsteps.$s.root")
      gen+=("$D/g4ref_v1_tables/gen.$s.root")
    done
    for k in $(seq 11 20); do
      s=$(printf '001400_%08d' "$k")
      evaluation+=("$D/g4ref_v1_tables/gen.$s.root")
    done
    python3 -m valhist.stepclosure shower-source --base "$L" --train "${train[@]}" --gen "${gen[@]}" \
      --evaluation "${evaluation[@]}" --out "$O/takeover/lib_shower_g4train.root"
    bash scripts/run_step_layers.sh "$O/takeover/lib_shower_g4train.root" "$O/takeover/ccp_shower_g4train" 11-20 \
      --crossing-context --contain-steps --point-placement
    python3 -m valhist.steplayers posyz --arm "a0g=$G/g4all.{s}.root" \
      --arm "ccp=$O/fast1b_ccp/fast1b.{s}.root" \
      --arm "ccp_shower_g4train=$O/takeover/ccp_shower_g4train/fast1b.{s}.root" \
      --shards 11-20 --out "$O/takeover/posyz_heldout.json"
    python3 -m valhist.steplayers breakdown --ref "$G/g4all.{s}.root" --cand "$O/fast1b_ccp/fast1b.{s}.root" \
      --shards 11-20 --out "$O/takeover/ccp_heldout_breakdown.json"
    ;;
  guard)
    python3 -m valhist.stepclosure primary-guard --base "$F/fastC.{s}.root" \
      --candidate "$O/fast1b_ccp/fast1b.{s}.root" --out "$O/takeover/primary_guard.json"
    python3 -m crvfastsim v1compare "$O/fast1b_ccp/fast1b.*.root" --compare "$G/g4all.*.root" \
      --outdir "$O/takeover/v1_ccp"
    python3 -m crvfastsim tier1b-check "$O/fast1b_ccp/fast1b.001400_0000001[1-9].root" \
      "$O/fast1b_ccp/fast1b.001400_00000020.root" --compare "$G/g4all.001400_0000001[1-9].root" \
      "$G/g4all.001400_00000020.root" --outdir "$O/takeover/v3_ccp_s11_20"
    ;;
  *) echo 'choose source or guard' >&2; exit 2 ;;
esac
