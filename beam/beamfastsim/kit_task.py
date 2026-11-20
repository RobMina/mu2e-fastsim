"""fastsimkit task `beam_point`: stopping-target metrics for one target geometry from the beam fast transport.

Registered in mu2efastsim.endpoints.kit by `register(TASKS, Task, Resources, FNAL_TASKS)`. Reads the Mu2e-internal
DS-entrance muons (g4ref table made from MuBeamCat) in place, so it runs only on an endpoint that mounts FNAL storage
(eaf), or with the muon table streamed into the job on a Slurm-over-SSH endpoint (UVA; kit_uva). Params:
  geom     geometry file: absolute path on FNAL storage, a name under beam/geom, or "nominal"
  muons    number of DS-entrance muons to transport (<= the table size, 105,350 for MDC2025ac)
  seed     random seed
  repeat   warm passes for the throughput metric (0 = none)
  ds_bscale, tsd_bscale  optional field-map scales (default 1 = unscaled; mu2efastsim.bfield_scale), on the
           streaming route (UVA) only; they scale the maps the job reads, downstream of its start table
Metrics: stop_frac, stop_frac_err, stop_frac_mu_minus, stops, muons, foils, stops_foil_mean, stop_r_mean,
  stop_frac_upstream_half (fraction of stops in the upstream half of the foils), muons_per_s_warm, first_pass_s.
"""
import math

PARAMS = ("geom", "muons", "seed", "repeat", "ds_bscale", "tsd_bscale")
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
SOB_PARAMS = ("geom", "muons", "electrons", "seed", "ds_bscale", "tsd_bscale")
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
EDEP_PARAMS = ("geom", "muons", "electrons", "seed", "dtype", "ds_bscale", "tsd_bscale")
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

# Task beam_edep_cpu: beam_edep on ONE CPU core (a grid-slot-sized job, no GPU): the same point, float64 by default (on
# a CPU float32 is slower), JAX single-threaded, edepana's CPU default batch (5,000 electrons). One Run 1A design point
# with 100k electrons takes about 1.6 core-hours (beam/docs/edepana.md).
EDEP_CPU_MINUTES = 240      # the request outside the measured range below
EDEP_CPU_MEM_GB = 16
# Measured on UVA `standard` (sacct, 2026-11-18), all table muons (muons=0, the most muon work):
#   5,000 CE: elapsed 20-22 min, MaxRSS 8.9-9.4 GB (6 jobs); 10,000 CE: elapsed 29-42 min, MaxRSS <= 10.3 GB (74 jobs).
# Memory does not grow with the electrons (edepana's 5,000-electron batches); the longest elapsed is ~2 + 4 min per 1k CE.
# Inside the measured range (up to twice it) the request is 1.5x that time and 12 GB, so a job fits the scheduler's
# backfill gaps and bills 3 memory units instead of 4 (skill sizing-uva-slurm-jobs).
EDEP_CPU_MEASURED_ELECTRONS = 20_000
EDEP_CPU_MEASURED_MEM_GB = 12


def edep_cpu_resources(params, resources_cls):
    """The beam_edep_cpu request for these params: sized from the measured jobs above for 1..20,000 electrons, the
    flat EDEP_CPU_MINUTES x EDEP_CPU_MEM_GB otherwise (a count that has not been measured)."""
    n = int(params.get("electrons") or 0)
    if not 0 < n <= EDEP_CPU_MEASURED_ELECTRONS:
        return resources_cls(cores=1, mem_gb=EDEP_CPU_MEM_GB, gpus=0, gpu_type="", minutes=EDEP_CPU_MINUTES)
    longest = 2 + 4 * n / 1000
    minutes = min(EDEP_CPU_MINUTES, max(30, 5 * math.ceil(1.5 * longest / 5)))
    return resources_cls(cores=1, mem_gb=EDEP_CPU_MEASURED_MEM_GB, gpus=0, gpu_type="", minutes=minutes)
CPU_ENV = ("JAX_PLATFORMS=cpu OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 "
           "XLA_FLAGS=--xla_cpu_multi_thread_eigen=false")
EDEP_CPU_ARGS = EDEP_ARGS.replace("--procs " + str(EDEP_CORES - 1), "--procs 1")
GPU_CHECK = ('A python -c "import jax, sys; print(jax.devices()); sys.exit(jax.default_backend() != \'gpu\')" '
             '|| { echo "JAX not on the GPU"; exit 5; }\n')
EDEP_CPU_BODY = EDEP_BODY.replace(GPU_CHECK, "").replace("XLA_PYTHON_CLIENT_PREALLOCATE=false ", CPU_ENV + " ").replace(
    EDEP_ARGS, EDEP_CPU_ARGS)

# Task beam_centuple: the fast EventNtuple file of one target geometry (beamfastsim.centuple point): beam stops on the
# design studies' table, conversion electrons through the device chain with digis and the reco response, the
# calorimeter (device chain at the disk fronts, the Geant4 inner-edge table at the edge), the device trigger, donor
# events from stock development files, and the PyROOT writer in the Mu2e environment of the stock release (mounted
# /cvmfs). The file out/nts.mu2efastsim.centuple.root (EventNtuple/ntuple, stock split layout) is the input of the
# analysis server's EventNtuple analyses (e.g. Mu2eBO ce_chain's trigger_efficiency_ntuple); metrics are the ce_chain
# counts of the default selection, the chain counters and the stops. Inputs: beam data root / centuple (names in
# beamfastsim.centuple.__main__.DATA).
CENT_DIR = "centuple"
CENT_FILE = "nts.mu2efastsim.centuple.root"
CENT_CORES = 8
CENT_PARAMS = ("geom", "muons", "electrons", "seed", "dtype")
CENT_METRICS = ("n_events", "n_selected", "n_triggered", "efficiency", "n_generated", "births", "class_Q", "class_F",
                "class_N", "wall_s", "electrons_per_s", "beam_s", "stops_per_gen_event", "stops_per_pot", "stop_frac")
CENT_BODY = '@runtime_a@G4T=@fnal_data@/../beam_fastsim/' + SOB_TABLE + '\n' + \
    'CD=@fnal_data@/../beam_fastsim/' + CENT_DIR + '\n' + '''\
for f in "$G4T" "$CD/donor/donor.npz" "$CD/trigger/replay.fcl"; do [ -r "$f" ] || { echo "missing $f"; exit 3; }; done
[ -d /cvmfs/mu2e.opensciencegrid.org/Musings ] || { echo "no /cvmfs: the writer needs the Mu2e environment"; exit 3; }
case @p_geom@ in
  nominal) GEOM="" ;;
  /*) GEOM=@p_geom@ ;;
  *) GEOM="$M2FS_CODE/beam/geom/"@p_geom@ ;;
esac
@geometry_setup@[ -z "$GEOM" ] || [ -r "$GEOM" ] || { echo "missing geometry $GEOM"; exit 3; }
A python -c "import jax, sys; print(jax.devices()); sys.exit(jax.default_backend() != 'gpu')" || { echo "JAX not on the GPU"; exit 5; }
( cd "$M2FS_CODE" && PYTHONPATH="$M2FS_CODE/core:$M2FS_CODE/beam:$M2FS_CODE/calo:$M2FS_CODE/tracker:$M2FS_CODE" \\
  XLA_PYTHON_CLIENT_PREALLOCATE=false "$M2FS_VENV/bin/python" -m beamfastsim.centuple point --g4 "$G4T" \\
  --z-up ''' + str(SOB_Z_UP) + ''' ${GEOM:+--geom "$GEOM"} --data "$CD" --dtype @p_dtype@ --electrons @p_electrons@ \\
  --muons @p_muons@ --seed @p_seed@ --out "$M2FS_OUT" ) > "$M2FS_OUT/point.txt" 2>&1 || { tail -20 "$M2FS_OUT/point.txt"; exit 6; }
tail -c 300 "$M2FS_OUT/point.txt"
'''


# Task ptgt_point: one production-target design of the rod family (beamfastsim.ptgt_point): the device target + PS
# chain from the Geant4 thin-target bank to the TS-entrance states (cached per design on the endpoint), then tsdevice
# (the W1 TS + DS stage) to stops. Knobs Tlength, R_up, R_mid, R_dn (mm: the deck's polycone rod; R_up at the end the
# protons leave); pot per point; platform cpu or gpu (JAX backend); dtype float64 or float32. Inputs: the public
# Offline files and field maps (Mau13 for TS + DS, the G4beamline GA05 PSMap for the PS) staged from CVMFS, and three
# public-derived files in the endpoint data directory FASTSIMKIT_PTGT_DATA, checked against PTGT_DATA_SHA256: the
# thin-target bank (Geant4 11.3, QGSP_BERT_HP), the xs-only steel bank of the TS-window rings, the TS GDML (an export
# of the Offline Run 1A geometry). Metrics: stops_per_pot (+ Poisson error), the TS-entrance yields, timings.
PTGT_PARAMS = ("Tlength", "R_up", "R_mid", "R_dn", "pot", "seed", "dtype", "platform")
PTGT_DEFAULTS = {"pot": 100_000, "seed": 1, "dtype": "float64", "platform": "cpu"}
PTGT_METRICS = ("stops_per_pot", "stops_per_pot_err", "mu_minus_stops_per_pot", "stops", "pot", "ts_entrances",
                "ts_mu_minus_pi_minus_per_pot", "ts_mu_minus_pi_minus_per_pot_err", "ts_mu_pi_per_pot", "cached",
                "chain_s", "ts_s")
PTGT_DATA_SHA256 = {
    "bank.npz": "d44c7ca1a3357ed369b78202bd6eb8df3a567b9b18d8fb54810c2c30dfd53020",       # W thin-target bank v3
    "slab_bank.npz": "8019e6b77b6c40a752dfaaa87a2041fe4004b9e4001f6244cf3a8fa4b127d57e",  # G4_STAINLESS-STEEL, xs only
    "ts.gdml": "da1dbb04ad6973b633510241f6859af8ee06cfaa2312455b6aa151acddbe1fda",        # TS geometry, Run 1A
}
PTGT_MAPS = ("DSMap", "PSMap", "TSuMap_fix", "TSdMap", "PStoDumpAreaMap", "ProtonDumpAreaMap", "PSAreaMap", "WorldMap",
             "GA05/PSMap")
PTGT_FIELD_CONFIG = "Offline/Mu2eG4/geom/bfgeom_no_ds_v01.txt"
PTGT_FIELD_FILES = (PTGT_FIELD_CONFIG, "Offline/Mu2eG4/geom/bfgeom_v01.txt")   # the config and its include
PTGT_DS_GEOM = "Offline/Mu2eG4/geom/geom_run1_a.txt"
# Request sizing. CPU (measured, UVA standard, 8 threads, nominal rod, 20k POT): chain 3.5 ms/POT (float64) and
# 3.1 ms/POT (float32); tsdevice ~180 s fixed + ~0.02 s per TS entrant (~0.04 entrants/POT): 4.5 ms/POT with
# margin; MaxRSS 7.4 GB. GPU (A100, 1M-POT runs): chain 16-24 us/POT, tsdevice 13 us/POT, compile ~1-2 min.
# PTGT_FIXED_MINUTES covers staging, compile and the fixed tsdevice pass; the request is 1.5x the estimate.
PTGT_CPU_CORES = 8
PTGT_CPU_MEM_GB = 16
PTGT_CPU_S_PER_POT = 4.5e-3
PTGT_GPU_S_PER_POT = 1e-4
PTGT_FIXED_MINUTES = 10
PTGT_MAX_MINUTES = 24 * 60


def ptgt_resources(params, resources_cls):
    """The ptgt_point request for these params: 1.5x the estimated time, in 5-minute steps."""
    pot = int(params.get("pot") or PTGT_DEFAULTS["pot"])
    gpu = params.get("platform", PTGT_DEFAULTS["platform"]) == "gpu"
    est = PTGT_FIXED_MINUTES + pot * (PTGT_GPU_S_PER_POT if gpu else PTGT_CPU_S_PER_POT) / 60
    minutes = min(PTGT_MAX_MINUTES, 5 * math.ceil(1.5 * est / 5))
    if gpu:
        return resources_cls(cores=4, mem_gb=64, gpus=1, gpu_type="a100", minutes=minutes)
    return resources_cls(cores=PTGT_CPU_CORES, mem_gb=PTGT_CPU_MEM_GB, gpus=0, gpu_type="", minutes=minutes)


# ptgt_point needs the endpoint data directory and the staging of kit_uva (PtgtPointStream): it is delegate-only
# (endpoints.kit.DELEGATED_ONLY refuses a plain render); this body is never rendered.
PTGT_MOUNTED_BODY = 'echo "ptgt_point runs on the Slurm-over-SSH route (kit_uva.PtgtPointStream) only"; exit 3\n'


# Optional field-map scales of the beam tasks (default 1 = unscaled). They scale the maps the job reads (DSMap and
# DSExtension for ds_bscale, TSdMap for tsd_bscale), so they act only downstream of the task's start table: the
# Coll5_Out table of beam_sob / beam_edep(_cpu) carries the nominal TS3-TS5 history, so tsd_bscale is only partly
# represented there; the faithful path for tsd_bscale is W1's TS-entrance chain (tsdevice, GPU).
SCALE_DEFAULTS = {"ds_bscale": 1.0, "tsd_bscale": 1.0}


def register(tasks, task_cls, resources_cls, fnal_tasks):
    """Add beam_point to a fastsimkit TASKS dict (and to its FNAL-only set)."""
    tasks["beam_point"] = task_cls(params=PARAMS, metrics=METRICS,
                                   resources=resources_cls(cores=4, mem_gb=24, gpus=1, gpu_type="a100", minutes=30),
                                   body=BODY, defaults=dict(SCALE_DEFAULTS))
    fnal_tasks.add("beam_point")
    tasks["beam_sob"] = task_cls(params=SOB_PARAMS, metrics=SOB_METRICS,
                                 resources=resources_cls(cores=SOB_CORES, mem_gb=96, gpus=0, gpu_type="", minutes=40),
                                 body=SOB_BODY, defaults=dict(SCALE_DEFAULTS))
    fnal_tasks.add("beam_sob")
    tasks["beam_edep"] = task_cls(params=EDEP_PARAMS, metrics=EDEP_METRICS,
                                  resources=resources_cls(cores=EDEP_CORES, mem_gb=96, gpus=1, gpu_type="a100",
                                                          minutes=60),
                                  body=EDEP_BODY, defaults={"dtype": "float64", **SCALE_DEFAULTS})
    fnal_tasks.add("beam_edep")
    tasks["beam_edep_cpu"] = task_cls(params=EDEP_PARAMS, metrics=EDEP_METRICS,
                                      resources=resources_cls(cores=1, mem_gb=EDEP_CPU_MEM_GB, gpus=0, gpu_type="",
                                                              minutes=EDEP_CPU_MINUTES),
                                      body=EDEP_CPU_BODY, defaults={"dtype": "float64", **SCALE_DEFAULTS},
                                      sizer=lambda p: edep_cpu_resources(p, resources_cls))
    fnal_tasks.add("beam_edep_cpu")
    tasks["beam_centuple"] = task_cls(params=CENT_PARAMS, metrics=CENT_METRICS,
                                      resources=resources_cls(cores=CENT_CORES, mem_gb=96, gpus=1, gpu_type="a100",
                                                              minutes=90),
                                      body=CENT_BODY, defaults={"dtype": "float64"})
    fnal_tasks.add("beam_centuple")
    tasks["ptgt_point"] = task_cls(params=PTGT_PARAMS, metrics=PTGT_METRICS,
                                   resources=ptgt_resources(PTGT_DEFAULTS, resources_cls), body=PTGT_MOUNTED_BODY,
                                   defaults=dict(PTGT_DEFAULTS), sizer=lambda p: ptgt_resources(p, resources_cls))


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


class _UVAEdepCpuDelegate(_UVADelegate):
    """beam_edep_cpu on a Slurm-over-SSH endpoint: the same inputs streamed in, one CPU core."""

    @staticmethod
    def make(kit):
        from .kit_uva import BeamEdepCpuStream
        return BeamEdepCpuStream(kit)


class _UVACentupleDelegate(_UVADelegate):
    """beam_centuple on a Slurm-over-SSH endpoint: a GPU front job (inputs streamed in), then a CPU write job in the
    Mu2e environment through cvmfsexec (kit_uva.BeamCentupleStream)."""

    @staticmethod
    def make(kit):
        from .kit_uva import BeamCentupleStream
        return BeamCentupleStream(kit)


class _UVAPtgtDelegate(_UVADelegate):
    """ptgt_point on a Slurm-over-SSH endpoint: public inputs staged in the job, no stream (kit_uva.PtgtPointStream)."""

    @staticmethod
    def make(kit):
        from .kit_uva import PtgtPointStream
        return PtgtPointStream(kit)


def register_delegate(delegates):
    """beam_point, beam_sob and beam_edep on endpoints without FNAL storage run through kit_uva (inputs streamed)."""
    delegates["beam_point"] = _UVADelegate
    delegates["beam_sob"] = _UVASobDelegate
    delegates["beam_edep"] = _UVAEdepDelegate
    delegates["beam_edep_cpu"] = _UVAEdepCpuDelegate
    delegates["beam_centuple"] = _UVACentupleDelegate
    delegates["ptgt_point"] = _UVAPtgtDelegate
