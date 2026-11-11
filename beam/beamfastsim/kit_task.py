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
# Start table: the design studies' own beam sample at the Coll5_Out plane (MuBeamCat Run1Baa, with its generated events), so
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

# Task beam_edep: the fast EdepAna file of one target geometry (beamfastsim.edepana point): beam stops on the design
# studies' table (as beam_sob), conversion electrons through the device chain (cetracker.DeviceChain), the calorimeter
# shower library at the disk fronts and the cesob wall proxy for side entries; one GPU. The file
# out/nts.mu2efastsim.edep.root (EDepAna/tree) is the input of the analysis server's approx_ce_sensitivity; metrics
# are the edep summary block, the CE acceptance at the default selection and the stops.
EDEP_DIR = "edepana"                                    # under the beam data root: shower library and tracker GDML
EDEP_LIBRARY = EDEP_DIR + "/shower_library_cal2.npz"    # calorimeter shower library (CAL2 production candidate)
EDEP_GDML = EDEP_DIR + "/mu2e_run1a_v01.gdml"
EDEP_EDGE = EDEP_DIR + "/edge_library_mdc2025ac.npz"  # Geant4 inner-edge calorimeter energies (calo library+edge)
EDEP_FILE = "nts.mu2efastsim.edep.root"
EDEP_CORES = 16
EDEP_PARAMS = ("geom", "muons", "electrons", "seed", "dtype")
EDEP_METRICS = ("n_events", "n_gen_events", "event_rate", "avg_calo_edep_per_event_mev",
                "avg_calo_edep_per_gen_event_mev", "n_events_calo_edep_above_50mev", "avg_trk_edep_per_event_mev",
                "avg_trk_edep_per_gen_event_mev", "n_events_vis_gt10", "ce_acceptance", "births", "tracker_entry",
                "tracker_exit", "disk_front", "side_candidates", "side_accepted", "wall_front", "edge_entry",
                "below_rmax", "library_fallback", "stored",
                "wall_s", "electrons_per_s", "stops_per_gen_event", "stops_per_pot", "stop_frac", "stop_frac_err",
                "beam_s")
EDEP_ARGS = ('--library "$EL" --edge-library "$EE" --gdml "$EG" --dtype @p_dtype@ --electrons @p_electrons@ --muons @p_muons@ '
             '--seed @p_seed@ --procs ' + str(EDEP_CORES - 1))
EDEP_BODY = '@runtime_a@G4T=@fnal_data@/../beam_fastsim/' + SOB_TABLE + '\n' + \
    'EL=@fnal_data@/../beam_fastsim/' + EDEP_LIBRARY + '; EG=@fnal_data@/../beam_fastsim/' + EDEP_GDML + '\n' + \
    'EE=@fnal_data@/../beam_fastsim/' + EDEP_EDGE + '\n' + '''\
for f in "$G4T" "$EL" "$EG" "$EE"; do [ -r "$f" ] || { echo "missing $f"; exit 3; }; done
case @p_geom@ in
  nominal) GEOM="" ;;
  /*) GEOM=@p_geom@ ;;
  *) GEOM="$M2FS_CODE/beam/geom/"@p_geom@ ;;
esac
@geometry_setup@[ -z "$GEOM" ] || [ -r "$GEOM" ] || { echo "missing geometry $GEOM"; exit 3; }
A python -c "import jax, sys; print(jax.devices()); sys.exit(jax.default_backend() != 'gpu')" || { echo "JAX not on the GPU"; exit 5; }
( cd "$M2FS_CODE" && PYTHONPATH="$M2FS_CODE/core:$M2FS_CODE/beam:$M2FS_CODE/calo:$M2FS_CODE/tracker" \\
  XLA_PYTHON_CLIENT_PREALLOCATE=false "$M2FS_VENV/bin/python" -m beamfastsim.edepana point --g4 "$G4T" \\
  --z-up ''' + str(SOB_Z_UP) + ''' ${GEOM:+--geom "$GEOM"} ''' + EDEP_ARGS + ''' --out "$M2FS_OUT" ) \\
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
                                 resources=resources_cls(cores=SOB_CORES, mem_gb=96, gpus=0, gpu_type="", minutes=40),
                                 body=SOB_BODY)
    fnal_tasks.add("beam_sob")
    tasks["beam_edep"] = task_cls(params=EDEP_PARAMS, metrics=EDEP_METRICS,
                                  resources=resources_cls(cores=EDEP_CORES, mem_gb=96, gpus=1, gpu_type="a100",
                                                          minutes=60),
                                  body=EDEP_BODY, defaults={"dtype": "float64"})
    fnal_tasks.add("beam_edep")


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


class _UVAEdepDelegate(_UVADelegate):
    """beam_edep on a Slurm-over-SSH endpoint: the beam table, shower library and GDML streamed in, one GPU job."""

    @staticmethod
    def make(kit):
        from .kit_uva import BeamEdepStream
        return BeamEdepStream(kit)


def register_delegate(delegates):
    """beam_point, beam_sob and beam_edep on endpoints without FNAL storage run through kit_uva (inputs streamed)."""
    delegates["beam_point"] = _UVADelegate
    delegates["beam_sob"] = _UVASobDelegate
    delegates["beam_edep"] = _UVAEdepDelegate
