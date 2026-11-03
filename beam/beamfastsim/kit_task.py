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

# Task beam_sob: the sob stand-in of the foils studies (beamfastsim.cesob point) for one target geometry, on CPU.
# Start table: the boards' own beam sample at the Coll5_Out plane (MuBeamCat Run1Baa, with its generated events), so
# stops per generated event need no scale. Params: geom, muons (0 = all), electrons (conversion electrons from the
# stops), seed. Metrics: sensitivity (sob) and its factors, the stop metrics of beam_point, timings.
SOB_TABLE = "g4ref_run1baa/g4ref_run1baa_vd8.npz"
SOB_Z_UP = 3900.0
SOB_CORES = 32
SOB_PARAMS = ("geom", "muons", "electrons", "seed")
SOB_METRICS = ("sensitivity", "sensitivity_err", "signal_box_low_mev", "signal_box_high_mev", "signal_rate",
               "dio_background", "cosmic_background", "total_background", "total_background_err", "signal_mpv_mev",
               "signal_fwhm_mev", "stops_per_gen_event", "stops_per_pot", "ce_acceptance", "ce_acceptance_err",
               "box_eff_per_stop", "box_eff_per_stop_err", "electrons", "plane_frac", "stop_frac", "stop_frac_err",
               "stops", "muons", "foils", "muons_per_event", "beam_s", "ce_s", "electrons_per_s")
SOB_BODY = '''@runtime_a@G4T=@fnal_data@/../beam_fastsim/''' + SOB_TABLE + '''
[ -r "$G4T" ] || { echo "missing $G4T"; exit 3; }
case @p_geom@ in
  nominal) GEOM="" ;;
  /*) GEOM=@p_geom@ ;;
  *) GEOM="$M2FS_CODE/beam/geom/"@p_geom@ ;;
esac
@geometry_setup@[ -z "$GEOM" ] || [ -r "$GEOM" ] || { echo "missing geometry $GEOM"; exit 3; }
( cd "$M2FS_CODE" && PYTHONPATH="$M2FS_CODE/core:$M2FS_CODE/beam" JAX_PLATFORMS=cpu \\
  "$M2FS_VENV/bin/python" -m beamfastsim.cesob point --g4 "$G4T" --z-up ''' + str(SOB_Z_UP) + ''' ${GEOM:+--geom "$GEOM"} \\
  --muons @p_muons@ --electrons @p_electrons@ --seed @p_seed@ --procs ''' + str(SOB_CORES) + ''' --out "$M2FS_OUT" ) \\
  > "$M2FS_OUT/point.txt" 2>&1 || { tail -20 "$M2FS_OUT/point.txt"; exit 6; }
tail -c 300 "$M2FS_OUT/point.txt"
'''


def register(tasks, task_cls, resources_cls, fnal_tasks):
    """Add beam_point to a fastsimkit TASKS dict (and to its FNAL-only set)."""
    tasks["beam_point"] = task_cls(params=PARAMS, metrics=METRICS,
                                   resources=resources_cls(cores=4, mem_gb=24, gpus=1, gpu_type="a100", minutes=30),
                                   body=BODY)
    fnal_tasks.add("beam_point")
    tasks["beam_sob"] = task_cls(params=SOB_PARAMS, metrics=SOB_METRICS,
                                 resources=resources_cls(cores=SOB_CORES, mem_gb=64, gpus=0, gpu_type="", minutes=40),
                                 body=SOB_BODY)
    fnal_tasks.add("beam_sob")


class _UVADelegate:
    """beam_point on a Slurm-over-SSH endpoint (no FNAL storage): stream the muon table in (kit_uva)."""

    @staticmethod
    def applies(endpoint):
        return not getattr(endpoint, "fnal_data", None) and hasattr(endpoint, "stream_in")

    @staticmethod
    def make(kit):
        from .kit_uva import BeamPointStream
        return BeamPointStream(kit)


class _UVASobDelegate(_UVADelegate):
    """beam_sob on a Slurm-over-SSH endpoint: the same streaming route, one CPU job."""

    @staticmethod
    def make(kit):
        from .kit_uva import BeamSobStream
        return BeamSobStream(kit)


def register_delegate(delegates):
    """beam_point and beam_sob on endpoints without FNAL storage run through kit_uva (inputs streamed into the job)."""
    delegates["beam_point"] = _UVADelegate
    delegates["beam_sob"] = _UVASobDelegate
