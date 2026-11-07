"""Strict development adapter; no EWM rerun, nominal conditions, or time-based joins.

Timing provenance is caller supplied: truth = fast_inputs_dev1253_v2/response.root,
reco = four_stream_dev1253/response.root. Only highest-cycle ROOT tables are read.
"""
from dataclasses import fields, replace
from io import StringIO
from pathlib import Path

import awkward as ak
import numpy as np
import pandas as pd
import uproot

from .frames import IDS, CRV_FIELDS, read_event_log

CRV_DIRS = ("primaryCrv", "stopCrv", "eleCrv", "muCrv", "neutralCrv")
TRACKER_DIRS = ("primarySteps", "stopSteps", "eleSteps", "muSteps", "neutralSteps")
MICROBUNCH_PERIOD, DIGITIZATION_MARGIN = 1695.0, 100.0


def _table(root, name):
    return pd.DataFrame(root[name].arrays(library="np"))


def _integers(values, label, nonnegative=True):
    a = np.asarray(values)
    if (not np.issubdtype(a.dtype, np.number) or not np.isfinite(a).all()
            or np.any(a != np.floor(a)) or (nonnegative and np.any(a < 0))):
        raise ValueError(f"invalid integer {label}")
    return a


def _ids(table):
    return _integers(table[list(IDS)].to_numpy(), "event IDs")


def _in_census(table, census):
    _ids(table)
    found = table[list(IDS)].merge(census, on=list(IDS), how="left", validate="m:1", indicator=True)
    if (found._merge != "both").any():
        raise ValueError("table event absent from full-ID census")


def check_tracker_inputs(input_path, truth_path):
    """Certify all five ordered sgsteps tables, every branch and row (not a prefix)."""
    with uproot.open(input_path) as source, uproot.open(truth_path) as truth:
        for directory in TRACKER_DIRS:
            left, right = source[f"{directory}/sgsteps"], truth[f"{directory}/sgsteps"]
            if set(left.keys()) != set(right.keys()) or not ak.almost_equal(
                    left.arrays(right.keys(), library="ak"), right.arrays(library="ak"), rtol=0, atol=0):
                raise ValueError(f"changed tracker inputs: {directory}")
    return True


def read_inputs(input_path, truth_path, reco_path):
    """Return (keys[N,3], steps, pdg, mc_pbtime, reco_pbtime, event_length).

    The adjacent art.log is the independent all-event census, including zero-step
    events. Particle keys are scoped to full IDs AND their own stream. No event loops.
    """
    keys = read_event_log(Path(input_path).parent / "art.log")
    census = pd.DataFrame(keys, columns=IDS)
    if not len(keys) or census.duplicated().any():
        raise ValueError("missing or duplicate EventLister census")
    timing = []
    for path in (truth_path, reco_path):
        with uproot.open(path) as root:
            events = _table(root, "crvdump/events")
        if not np.array_equal(_ids(events), keys) or events[list(IDS)].duplicated().any():
            raise ValueError("saved timing full-ID census/order mismatch")
        if not (events.spill == 1).all() or not np.isfinite(events[["pbtime", "evlen"]]).all().all():
            raise ValueError("saved timing must be finite and on spill 1")
        if not (events.evlen > 0).all():
            raise ValueError("invalid event length")
        timing.append(events)
    if not np.array_equal(timing[0].evlen, timing[1].evlen):
        raise ValueError("truth/reco event length mismatch")
    check_tracker_inputs(input_path, truth_path)
    steps, pdgs = [], []
    with uproot.open(input_path) as root:
        for directory in CRV_DIRS:
            step, part, gen = [_table(root, f"{directory}/{name}") for name in ("steps", "particles", "gen")]
            for table in (step, part, gen):
                _in_census(table, census)
            _integers(step.sp, "step sp")
            _integers(part.key, "particle key")
            _integers(part.pdg, "particle pdg", nonnegative=False)
            payload = [k for k in CRV_FIELDS if k not in ("t0", "t1")]
            if not np.isfinite(step[payload]).all().all() or np.isinf(step[["t0", "t1"]]).any().any():
                raise ValueError("nonfinite CRV step payload or infinite endpoint time")
            products = "sp_product_id" in step and "sim_product_id" in part
            joined = step[[*IDS, "sp"] + (["sp_product_id"] if products else [])].assign(_row=np.arange(len(step))).merge(
                part[[*IDS, "key", "pdg"] + (["sim_product_id"] if products else [])], left_on=[*IDS, "sp"], right_on=[*IDS, "key"],
                how="left", sort=False, validate="m:1", indicator=True).sort_values("_row", kind="stable")
            if (joined._merge != "both").any():
                raise ValueError(f"dangling SimParticle in {directory}")
            steps.append(step)
            if products and not (joined.sp_product_id == joined.sim_product_id).all():
                raise ValueError(f"SimParticle product mismatch in {directory}")
            pdgs.append(joined.pdg.to_numpy(dtype=np.int64))
    return (keys, pd.concat(steps, ignore_index=True), np.concatenate(pdgs),
            timing[0].pbtime.to_numpy(), timing[1].pbtime.to_numpy(), timing[0].evlen.to_numpy())


def read_timing(keys, truth_path, reco_path):
    """Align explicit saved MC/reco timing to primary full IDs, without using hit times."""
    from .frames import _event_keys
    wanted = _event_keys(keys)
    if not len(keys) or len(np.unique(wanted)) != len(keys):
        raise ValueError("missing or duplicate primary event IDs")
    tables = []
    for path in (truth_path, reco_path):
        with uproot.open(path) as root:
            table = _table(root, "crvdump/events")
        census = _event_keys(_ids(table))
        if len(np.unique(census)) != len(census):
            raise ValueError("duplicate saved timing IDs")
        if not (table.spill == 1).all() or not np.isfinite(table[["pbtime", "evlen"]]).all().all():
            raise ValueError("saved timing must be finite and on spill 1")
        if not (table.evlen > 0).all():
            raise ValueError("invalid event length")
        order = np.argsort(census)
        rows = np.searchsorted(census[order], wanted)
        if np.any(rows >= len(census)) or not np.array_equal(census[order[rows]], wanted):
            raise ValueError("primary event absent from explicit timing census")
        tables.append(table.iloc[order[rows]])
    if not np.array_equal(tables[0].evlen, tables[1].evlen):
        raise ValueError("truth/reco event length mismatch")
    return tables[0].pbtime.to_numpy(), tables[1].pbtime.to_numpy(), tables[0].evlen.to_numpy()


def load_conditions(truth_root, photon_csv, geom, counters):
    """Load explicit CRVCalib/CRVPhoton, apply physical readout, trim nongeometry channels."""
    from crvfastsim import chain, readout
    with uproot.open(truth_root) as root:
        calib = _table(root, "crvdump/calib")
    names = "channel pedestal pulsearea pulseheight timeoffset status roc feb febch".split()
    calib = calib[names].sort_values("channel")
    lines = [s.split("#", 1)[0].strip() for s in Path(photon_csv).read_text().splitlines()]
    lines = [s for s in lines if s]
    if lines and lines[0].startswith("TABLE"):
        if lines.pop(0) != "TABLE CRVPhoton":
            raise ValueError("photon CSV names a different table")
    if not lines:
        raise ValueError("empty CRVPhoton export")
    photon = np.loadtxt(StringIO("\n".join(lines)), delimiter=",", ndmin=2)
    if photon.shape[1] != 2 or not np.isfinite(photon).all() or not np.isfinite(calib).all().all():
        raise ValueError("nonfinite or malformed CRV conditions")
    _integers(photon[:, 0], "photon channel")
    for name in ("channel", "status", "roc", "feb", "febch"):
        _integers(calib[name], name)
    photon = photon[np.argsort(photon[:, 0])]
    n = len(calib)
    if (n < 4 * geom.nbars or not np.array_equal(calib.channel, np.arange(n))
            or not np.array_equal(photon[:, 0], np.arange(n))):
        raise ValueError("missing/duplicate channels or incomplete calibrated coverage")
    if (calib[["pulsearea", "pulseheight"]] <= 0).any().any():
        raise ValueError("nonpositive pulse calibration")
    cond = chain.Conditions(*(calib[k].to_numpy(dtype=np.int64 if k in ("status", "roc", "feb") else float)
                              for k in names[1:-1]), deviation=photon[:, 1])
    cond = readout.apply(cond, geom, counters)
    return replace(cond, **{f.name: getattr(cond, f.name)[:4 * geom.nbars].copy() for f in fields(cond)})


def _require(module, expected):
    for key, value in expected.items():
        actual = module.get(key)
        if ((isinstance(value, bool) != isinstance(actual, bool)) or actual != value):
            raise ValueError(f"unsupported/missing {key}: {actual!r}; expected {value!r}")


def _matched(module, config, stock_names, field_names):
    mapping = dict(zip(stock_names.split(), field_names.split(), strict=True))
    _require(module, {k: getattr(config, v) for k, v in mapping.items()})
    return replace(config, **{v: module[k] for k, v in mapping.items()})


def build_config(expanded, *, jax=False, period=MICROBUNCH_PERIOD):
    """Build only the supported Run1A stock setting from parsed expanded.fcl.

    Caller uses digicond.parse_fhicl_dump. The microbunch period is an explicit
    kernel contract (not CRVDigitizationPeriod=12.5); margin is 100 ns. Missing
    physics settings fail, except documented C++ InputTag/local-table defaults.
    """
    from crvfastsim import runconfig
    from crvfastsim.response.sipm import SiPMConfig
    from crvfastsim.reco.waveform import WaveformConfig
    from crvfastsim.reco.pulsefit import RecoConfig
    if period != MICROBUNCH_PERIOD:
        raise ValueError("unsupported microbunch period")
    cfg = runconfig.get("run1a_v01").chain_config(jax=jax)
    producers = expanded["physics"]["producers"]
    labels = "CrvPhotons CrvSiPMCharges CrvWaveforms CrvDigi CrvRecoPulses CrvCoincidenceClusterFinder".split()
    modules = [producers[k] for k in labels]
    types = "CrvPhotonGenerator CrvSiPMChargeGenerator CrvWaveformsGenerator CrvDigitizer CrvRecoPulsesFinder CrvCoincidenceFinder".split()
    for module, kind in zip(modules, types):
        _require(module, {"module_type": kind})
    p, s, w, d, r, f = modules
    for module in modules[:4]:
        _require(module, {"simulateNZS": False})
    for module in modules[:3]:
        _require(module, {"digitizationStart": 200.0, "digitizationEnd": 1500.0})
        for tag in ("eventWindowMarkerTag", "protonBunchTimeMCTag"):
            _require({tag: module.get(tag, "EWMProducer")}, {tag: "EWMProducer"})
    for module in (p, s):
        _require(module, {"digitizationStartMargin": DIGITIZATION_MARGIN})
    _require(p, dict(CRVSectors=["T1", "T2"], reflectors=[0, 0], scintillationYields=[39794, 30482],
                     lookupTableFileNames=["CRVConditions/v6_0/LookupTable_6000_0"] * 2,
                     photonYieldScaleFactor=0.9, photonYieldVariationScale=1.0))
    _require({"localLookupTables": p.get("localLookupTables", False)}, {"localLookupTables": False})
    _require(s, dict(useSipmStatusDB=True, photonMapFileName="CRVConditions/v6_0/photonMap.root",
                     inactivePixels=[list(x) for x in SiPMConfig().inactive]))
    sipm = _matched(s, replace(SiPMConfig(), thermal_rate=1e-4),
        "nPixelsX nPixelsY overvoltage timeConstant capacitance AvalancheProbParam1 AvalancheProbParam2 CrossTalkProb ThermalRate TrapType0Prob TrapType1Prob TrapType0Lifetime TrapType1Lifetime",
        "n_pixels_x n_pixels_y overvoltage time_constant capacitance p1 p2 cross_talk thermal_rate trap0_prob trap1_prob trap0_lifetime trap1_lifetime")
    _require(w, {"useTimeOffsetDB": False, "timeOffsetScale": 1.0})
    wave = _matched(w, WaveformConfig(),
        "singlePEWaveformFileName singlePEWaveformPrecision singlePEWaveformStretchFactor singlePEWaveformMaxTime singlePEReferenceCharge minVoltage noise numberSamplesZS",
        "single_pe_file single_pe_precision single_pe_stretch single_pe_max_time single_pe_reference_charge min_voltage noise samples_zs")
    wave = _matched(d, wave, "ADCconversionFactor pedestal", "adc_conversion pedestal")
    _require(r, {"NZSdata": False})
    for tag in ("eventWindowMarkerTag", "protonBunchTimeTag"):
        _require({tag: r.get(tag, "EWMProducer")}, {tag: "EWMProducer"})
    reco = _matched(r, RecoConfig(),
        "minADCdifference defaultBeta minBeta maxBeta maxTimeDifference minPulseHeightRatio maxPulseHeightRatio LEtimeFactor samplesBefore samplesAfter pedestalUndershootThreshold useTimeOffsetDB timeOffsetScale ignoreChannels",
        "min_adc_difference default_beta min_beta max_beta max_time_difference min_pulse_height_ratio max_pulse_height_ratio le_time_factor samples_before samples_after pedestal_undershoot_threshold use_time_offset_db time_offset_scale ignore_channels")
    _matched(f, cfg.finder, "fiberSignalSpeed maxTimeInterval timeOffset ignoreSecondaryPulses bigClusterThreshold",
             "fiber_signal_speed max_time_interval time_offset ignore_secondary_pulses big_cluster_threshold")
    _require(f, {"compensateChannelStatus": list(cfg.finder.compensate_status_bits)})
    _require({"addOppositeSidePulses": f.get("addOppositeSidePulses", True)}, {"addOppositeSidePulses": True})
    cuts = f.get("sectorConfig", [])
    if [c.get("CRVSector") for c in cuts] != ["T1", "T2"]:
        raise ValueError("unsupported finder sectors")
    for cut in cuts:
        _matched(cut, cfg.finder.cuts[cut["CRVSector"]],
            "PEthreshold maxTimeDifferenceAdjacentPulses maxTimeDifference minSlope maxSlope maxSlopeDifference coincidenceLayers minClusterPEs initialClusterMaxDistance finalClusterMaxDistance",
            "pe_threshold max_dt_adjacent max_dt min_slope max_slope max_slope_diff coincidence_layers min_cluster_pes initial_max_distance final_max_distance")
    return replace(cfg, response=replace(cfg.response, photon_yield_scale=p["photonYieldScaleFactor"],
                   photon_yield_variation_scale=p["photonYieldVariationScale"]), sipm=sipm,
                   waveform=wave, pulse_reco=reco, window=(w["digitizationStart"], w["digitizationEnd"]), noise=True)
