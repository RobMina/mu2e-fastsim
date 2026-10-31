"""fastsimkit task `beam_point`: stopping-target metrics for one target geometry from the beam fast transport.

Registered in mu2efastsim.endpoints.kit by `register(TASKS, Task, Resources, FNAL_TASKS)`. Reads the Mu2e-internal
DS-entrance muons (g4ref table made from MuBeamCat) in place, so it runs only on an endpoint that mounts FNAL storage
(eaf), or with the muon table streamed into the job on a Slurm-over-SSH endpoint (UVA; kit_uva). Params:
  geom     geometry file: absolute path on FNAL storage, a name under beam/geom, or "nominal"
  muons    number of DS-entrance muons to transport (<= the table size, 105,350 for MDC2025ac)
  seed     random seed
  repeat   warm passes for the throughput metric (0 = none)
Metrics: stop_frac, stop_frac_err, stop_frac_mu_minus, stops, muons, foils, stops_foil_mean, stop_r_mean,
  stop_frac_upstream_half (fraction of stops in the upstream half of the foils), muons_per_s_warm, first_pass_s.
"""

PARAMS = ("geom", "muons", "seed", "repeat")
METRICS = ("stop_frac", "stop_frac_err", "stop_frac_mu_minus", "stops", "muons", "foils", "stops_foil_mean",
           "stop_r_mean", "stop_frac_upstream_half", "muons_per_s_warm", "first_pass_s")

BODY = '''@runtime_a@G4T=@fnal_data@/../beam_fastsim/g4ref_full/g4ref.npz   # kit fnal_data is .../crv_fastsim
[ -r "$G4T" ] || { echo "missing $G4T"; exit 3; }
case @p_geom@ in
  nominal) GEOM="" ;;
  /*) GEOM=@p_geom@ ;;
  *) GEOM="$M2FS_CODE/beam/geom/"@p_geom@ ;;
esac
@geometry_setup@[ -z "$GEOM" ] || [ -r "$GEOM" ] || { echo "missing geometry $GEOM"; exit 3; }
A python -c "import jax, sys; print(jax.devices()); sys.exit(jax.default_backend() != 'gpu')" || { echo "JAX not on the GPU"; exit 5; }
( cd "$M2FS_CODE" && PYTHONPATH="$M2FS_CODE/core:$M2FS_CODE/beam" XLA_PYTHON_CLIENT_PREALLOCATE=false \\
  "$M2FS_VENV/bin/python" -m beamfastsim point --g4 "$G4T" ${GEOM:+--geom "$GEOM"} --n @p_muons@ --seed @p_seed@ \\
  --repeat @p_repeat@ --batch 131072 --chunk 400 --out "$M2FS_OUT" ) > "$M2FS_OUT/point.txt" 2>&1 || { tail -20 "$M2FS_OUT/point.txt"; exit 6; }
tail -3 "$M2FS_OUT/point.txt"
'''


def register(tasks, task_cls, resources_cls, fnal_tasks):
    """Add beam_point to a fastsimkit TASKS dict (and to its FNAL-only set)."""
    tasks["beam_point"] = task_cls(params=PARAMS, metrics=METRICS,
                                   resources=resources_cls(cores=4, mem_gb=24, gpus=1, gpu_type="a100", minutes=30),
                                   body=BODY)
    fnal_tasks.add("beam_point")


class _UVADelegate:
    """beam_point on a Slurm-over-SSH endpoint (no FNAL storage): stream the muon table in (kit_uva)."""

    @staticmethod
    def applies(endpoint):
        return not getattr(endpoint, "fnal_data", None) and hasattr(endpoint, "stream_in")

    @staticmethod
    def make(kit):
        from .kit_uva import BeamPointStream
        return BeamPointStream(kit)


def register_delegate(delegates):
    """beam_point on endpoints without FNAL storage runs through kit_uva (inputs streamed into the job)."""
    delegates["beam_point"] = _UVADelegate
