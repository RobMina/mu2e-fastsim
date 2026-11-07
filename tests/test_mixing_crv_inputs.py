"""Synthetic-only CRV adapter contracts; no geometry downloads or real events."""
from copy import deepcopy
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import uproot

from mixing.crv_inputs import (CRV_DIRS, TRACKER_DIRS, build_config, check_tracker_inputs,
                               load_conditions, read_inputs)
from mixing.frames import CRV_FIELDS, IDS


KEYS = np.array([[1, 1, 1], [1, 2, 1], [2, 1, 1], [2, 1, 2]], dtype=np.int64)


def write_tree(path, name, data):
    with uproot.update(path) as root:
        root.mktree(name, {k: np.asarray(v) for k, v in data.items()})


def table(path, name):
    with uproot.open(path) as root:
        return root[name].arrays(library="np")


@pytest.fixture
def inputs(tmp_path):
    paths = [tmp_path / name for name in ("input.root", "truth.root", "reco.root")]
    for path in paths:
        with uproot.recreate(path):
            pass
    for i, path in enumerate(paths[1:]):
        events = dict(zip(IDS, KEYS.T))
        events.update(spill=np.ones(4, dtype=int), evlen=np.full(4, 1695.), pbtime=np.arange(4.) - 212.5 + i * .5)
        write_tree(path, "crvdump/events", dict(events, event=np.zeros(4, dtype=int)))  # stale cycle
        write_tree(path, "crvdump/events", events)
    (tmp_path / "art.log").write_text("\n".join(
        f"Event: {i} run: {r} subRun: {s} event: {e}" for i, (r, s, e) in enumerate(KEYS)))
    for directory in TRACKER_DIRS:
        for path in paths[:2]:
            write_tree(path, f"{directory}/sgsteps", dict(run=np.array([1]), subrun=np.array([1]),
                       event=np.array([1]), extra_branch=np.array([123.]), sp=np.array([7])))
    for stream, directory in enumerate(CRV_DIRS):
        take = np.array([2, 0, 2, 1])
        step = dict(zip(IDS, KEYS[take].T))
        step.update({key: np.arange(4, dtype=float) for key in CRV_FIELDS})
        step.update(sp=np.full(4, 7), bar=np.zeros(4, dtype=int), nstep=np.full(4, 4))
        part = dict(zip(IDS, KEYS[[1, 2, 0]].T))
        part.update(key=np.full(3, 7), pdg=np.array([11, 13, 22]) + 100 * stream)
        write_tree(paths[0], f"{directory}/steps", step)
        write_tree(paths[0], f"{directory}/particles", part)
        write_tree(paths[0], f"{directory}/gen", dict(zip(IDS, KEYS[:0].T), idx=np.array([], dtype=int)))
    return paths


def test_saved_timing_census_fullid_stream_join_and_order(inputs):
    keys, steps, pdg, mc, reco, length = read_inputs(*inputs)
    np.testing.assert_array_equal(keys, KEYS)
    assert isinstance(steps, pd.DataFrame) and len(steps) == 20
    np.testing.assert_array_equal(steps.t0, np.tile(np.arange(4), 5))
    np.testing.assert_array_equal(pdg, np.concatenate([np.array([13, 22, 13, 11]) + i * 100 for i in range(5)]))
    np.testing.assert_array_equal(mc, np.arange(4.) - 212.5)
    np.testing.assert_array_equal(reco, mc + .5)
    np.testing.assert_array_equal(length, np.full(4, 1695.))
    assert check_tracker_inputs(*inputs[:2]) is True


def test_all_five_empty_tables_preserve_events(inputs):
    for directory in CRV_DIRS:
        for name in ("steps", "particles", "gen"):
            path = f"{directory}/{name}"
            write_tree(inputs[0], path, {k: v[:0] for k, v in table(inputs[0], path).items()})
    keys, steps, pdg, *_ = read_inputs(*inputs)
    assert len(keys) == 4 and steps.empty and len(pdg) == 0


@pytest.mark.parametrize("which,mutation", [(1, "order"), (2, "run"), (1, "duplicate"),
                                          (2, "spill"), (1, "nan"), (2, "length")])
def test_bad_saved_timing(inputs, which, mutation):
    events = table(inputs[which], "crvdump/events")
    if mutation == "order":
        events = {k: v[::-1] for k, v in events.items()}
    elif mutation == "run":
        events["run"][0] = 9
    elif mutation == "duplicate":
        for k in IDS:
            events[k][1] = events[k][0]
    elif mutation == "spill":
        events["spill"][0] = 0
    elif mutation == "nan":
        events["pbtime"][0] = np.nan
    else:
        events["evlen"][0] += 1
    write_tree(inputs[which], "crvdump/events", events)
    with pytest.raises(ValueError):
        read_inputs(*inputs)


@pytest.mark.parametrize("mutation", ["missing", "duplicate", "order"])
def test_bad_independent_census(inputs, mutation):
    path = inputs[0].parent / "art.log"
    rows = path.read_text().splitlines()
    path.write_text("\n".join(rows[:-1] if mutation == "missing" else rows + rows[:1] if mutation == "duplicate" else rows[::-1]))
    with pytest.raises(ValueError):
        read_inputs(*inputs)


@pytest.mark.parametrize("mutation", ["duplicate", "dangling", "foreign_event", "negative"])
def test_bad_particle_association(inputs, mutation):
    name = "muCrv/particles" if mutation == "duplicate" else "muCrv/steps"
    rows = table(inputs[0], name)
    if mutation == "duplicate":
        rows = {k: np.r_[v, v[:1]] for k, v in rows.items()}
    elif mutation == "foreign_event":
        rows["subrun"][0] = 100
    else:
        rows["sp"][0] = 8 if mutation == "dangling" else -1
    write_tree(inputs[0], name, rows)
    with pytest.raises(ValueError):
        read_inputs(*inputs)


@pytest.mark.parametrize("mutation", ["extra_branch", "row", "length", "missing_branch"])
def test_tracker_comparison_not_prefix_or_selected_columns(inputs, mutation):
    name = "neutralSteps/sgsteps"
    rows = table(inputs[0], name)
    if mutation == "extra_branch":
        rows["not_in_reference"] = np.ones(1)
    elif mutation == "missing_branch":
        del rows["extra_branch"]
    elif mutation == "length":
        rows = {k: np.r_[v, v] for k, v in rows.items()}
    else:
        rows["extra_branch"][0] += 1
    write_tree(inputs[0], name, rows)
    with pytest.raises(ValueError, match="changed tracker"):
        read_inputs(*inputs)


@pytest.fixture
def conditions(tmp_path):
    path = tmp_path / "conditions.root"
    n = 12  # full calibration covers more bars than the response geometry
    with uproot.recreate(path) as root:
        root.mktree("crvdump/calib", dict(channel=np.arange(n)[::-1], pedestal=np.arange(n)[::-1] + 100.,
            pulsearea=np.full(n, 10.), pulseheight=np.full(n, 2.), timeoffset=np.arange(n)[::-1] * .1,
            status=np.zeros(n, dtype=int), roc=np.ones(n, dtype=int), feb=np.full(n, 2), febch=np.arange(n)[::-1]))
    photon = tmp_path / "photon.csv"
    photon.write_text("TABLE CRVPhoton\n# channel,deviation\n\n" + "\n".join(f"{i},{i * .01}" for i in range(n)[::-1]))
    geom = SimpleNamespace(nbars=2, sector=np.array([0, 1]), sector_names=["T1", "T2"])
    counters = {"crs.sectorNames": ["T1", "T2"], "crs.sipmsAtSide0T1": True,
                "crs.sipmsAtSide1T1": False, "crs.sipmsAtSide0T2": False, "crs.sipmsAtSide1T2": True}
    return path, photon, geom, counters


def test_explicit_conditions_trim_and_readout(conditions):
    cond = load_conditions(*conditions)
    np.testing.assert_array_equal(cond.pedestal, 100 + np.arange(8))
    np.testing.assert_array_equal(cond.deviation, np.arange(8) * .01)
    np.testing.assert_array_equal(cond.readout_mask, [True, False, True, False, False, True, False, True])
    assert len(cond.status) == 8 and cond.status.dtype.kind == "i"
    assert (cond.roc == 1).all() and (cond.feb == 2).all()


def test_dbtool_content_only_csv(conditions):
    path, photon, *_ = conditions
    expected = load_conditions(*conditions)
    photon.write_text(photon.read_text().replace("TABLE CRVPhoton\n", ""))
    actual = load_conditions(*conditions)
    np.testing.assert_array_equal(actual.deviation, expected.deviation)


@pytest.mark.parametrize("which,mutation", [("photon", "missing"), ("photon", "duplicate"),
    ("photon", "negative"), ("photon", "nan"), ("photon", "fractional"), ("photon", "header"),
    ("calib", "missing"), ("calib", "duplicate"), ("calib", "negative"), ("calib", "nan")])
def test_conditions_fail_closed(conditions, which, mutation):
    path, photon, *_ = conditions
    if which == "photon":
        text = photon.read_text()
        if mutation == "missing": text = "\n".join(text.splitlines()[:-1])
        if mutation == "duplicate": text += "\n0,0.5"
        if mutation == "negative": text = text.replace("0,0.0", "-1,0.0")
        if mutation == "nan": text = text.replace("0,0.0", "0,nan")
        if mutation == "fractional": text = text.replace("0,0.0", "0.5,0.0")
        if mutation == "header": text = text.replace("CRVPhoton", "OtherTable")
        photon.write_text(text)
    else:
        rows = table(path, "crvdump/calib")
        if mutation == "missing": rows = {k: v[:-1] for k, v in rows.items()}
        if mutation == "duplicate": rows["channel"][0] = rows["channel"][1]
        if mutation == "negative": rows["channel"][0] = -1
        if mutation == "nan": rows["pedestal"][0] = np.nan
        write_tree(path, "crvdump/calib", rows)
    with pytest.raises(ValueError):
        load_conditions(*conditions)


@pytest.fixture
def expanded():
    from trkfastsim.digi.digicond import parse_fhicl_dump
    return parse_fhicl_dump('''physics: { producers: {
      CrvPhotons: { module_type: CrvPhotonGenerator simulateNZS: false
        CRVSectors: ["T1", "T2"] reflectors: [0,0] scintillationYields: [39794,30482]
        lookupTableFileNames: ["CRVConditions/v6_0/LookupTable_6000_0", "CRVConditions/v6_0/LookupTable_6000_0"]
        photonYieldScaleFactor: 0.9 photonYieldVariationScale: 1.0
        digitizationStart: 200 digitizationEnd: 1500 digitizationStartMargin: 100 }
      CrvSiPMCharges: { module_type: CrvSiPMChargeGenerator simulateNZS: false useSipmStatusDB: true
        digitizationStart: 200 digitizationEnd: 1500 digitizationStartMargin: 100
        photonMapFileName: "CRVConditions/v6_0/photonMap.root"
        nPixelsX: 40 nPixelsY: 40 overvoltage: 3 timeConstant: 13.3 capacitance: 8.84e-14
        AvalancheProbParam1: 0.607 AvalancheProbParam2: 2.7 CrossTalkProb: 0.04 ThermalRate: 1e-4
        TrapType0Prob: 0 TrapType1Prob: 0 TrapType0Lifetime: 5 TrapType1Lifetime: 50
        inactivePixels: [[18,18],[18,19],[18,20],[18,21],[19,18],[19,19],[19,20],[19,21],
                         [20,18],[20,19],[20,20],[20,21],[21,18],[21,19],[21,20],[21,21]] }
      CrvWaveforms: { module_type: CrvWaveformsGenerator simulateNZS: false
        digitizationStart: 200 digitizationEnd: 1500 useTimeOffsetDB: false timeOffsetScale: 1
        singlePEWaveformFileName: "Offline/CRVResponse/data/singlePEWaveform_v3.txt"
        singlePEWaveformPrecision: 0.5 singlePEWaveformStretchFactor: 1.047 singlePEWaveformMaxTime: 100
        singlePEReferenceCharge: 2.652e-13 minVoltage: 0.0275 noise: 4e-4 numberSamplesZS: 12 }
      CrvDigi: { module_type: CrvDigitizer simulateNZS: false ADCconversionFactor: 2300 pedestal: 100 }
      CrvRecoPulses: { module_type: CrvRecoPulsesFinder NZSdata: false minADCdifference: 40 defaultBeta: 19
        minBeta: 5 maxBeta: 40 maxTimeDifference: 15 minPulseHeightRatio: 0.7 maxPulseHeightRatio: 1.5
        LEtimeFactor: 0.985 samplesBefore: 4 samplesAfter: 3 pedestalUndershootThreshold: 6
        useTimeOffsetDB: true timeOffsetScale: 1 ignoreChannels: true }
      CrvCoincidenceClusterFinder: { module_type: CrvCoincidenceFinder fiberSignalSpeed: 140
        maxTimeInterval: 20 timeOffset: 34.5 ignoreSecondaryPulses: false bigClusterThreshold: 50000
        compensateChannelStatus: [0,1,2] sectorConfig: [
          { CRVSector: T1 PEthreshold: 8 maxTimeDifferenceAdjacentPulses: 20 maxTimeDifference: 20
            minSlope: -6 maxSlope: 6 maxSlopeDifference: 3 coincidenceLayers: 3 minClusterPEs: 0
            initialClusterMaxDistance: 250 finalClusterMaxDistance: 0 },
          { CRVSector: T2 PEthreshold: 8 maxTimeDifferenceAdjacentPulses: 20 maxTimeDifference: 20
            minSlope: -6 maxSlope: 6 maxSlopeDifference: 3 coincidenceLayers: 3 minClusterPEs: 0
            initialClusterMaxDistance: 250 finalClusterMaxDistance: 0 } ] }
    } }''')


def test_stock_config_from_expanded(expanded):
    before = deepcopy(expanded)
    cfg = build_config(expanded, jax=True)
    assert cfg.jax and cfg.window == (200., 1500.)
    assert cfg.response.photon_yield_scale == .9 and cfg.sipm.thermal_rate == 1e-4
    assert cfg.waveform.noise == 4e-4 and cfg.pulse_reco.use_time_offset_db
    assert cfg.finder.cuts["T1"].coincidence_layers == 3
    assert expanded == before


@pytest.mark.parametrize("module,key,value", [
    ("CrvPhotons", "module_type", "Unknown"), ("CrvPhotons", "CRVSectors", ["T2", "T1"]),
    ("CrvPhotons", "photonYieldScaleFactor", .94), ("CrvPhotons", "localLookupTables", True),
    ("CrvPhotons", "digitizationStartMargin", 50), ("CrvPhotons", "digitizationEnd", 100000),
    ("CrvSiPMCharges", "TrapType0Prob", .1), ("CrvSiPMCharges", "ThermalRate", 0),
    ("CrvSiPMCharges", "simulateNZS", True), ("CrvWaveforms", "useTimeOffsetDB", True),
    ("CrvWaveforms", "noise", float("nan")), ("CrvDigi", "pedestal", 101),
    ("CrvRecoPulses", "NZSdata", True), ("CrvRecoPulses", "ignoreChannels", 1),
    ("CrvCoincidenceClusterFinder", "ignoreSecondaryPulses", True)])
def test_unsupported_stock_settings(expanded, module, key, value):
    expanded["physics"]["producers"][module][key] = value
    with pytest.raises(ValueError, match="unsupported"):
        build_config(expanded)


def test_missing_settings_and_changed_finder(expanded):
    f = expanded["physics"]["producers"]["CrvCoincidenceClusterFinder"]
    f["sectorConfig"][0]["PEthreshold"] = 9
    with pytest.raises(ValueError, match="PEthreshold"):
        build_config(expanded)
    f["sectorConfig"][0]["PEthreshold"] = 8
    del expanded["physics"]["producers"]["CrvWaveforms"]["noise"]
    with pytest.raises(ValueError, match="noise"):
        build_config(expanded)
    with pytest.raises(ValueError, match="period"):
        build_config(expanded, period=1700)


@pytest.mark.parametrize("wrong", [False, True])
def test_optional_simparticle_product_identity(inputs, wrong):
    for directory in CRV_DIRS:
        step = table(inputs[0], f"{directory}/steps")
        part = table(inputs[0], f"{directory}/particles")
        step["sp_product_id"] = np.full(len(step["sp"]), 42)
        part["sim_product_id"] = np.full(len(part["key"]), 43 if wrong else 42)
        write_tree(inputs[0], f"{directory}/steps", step)
        write_tree(inputs[0], f"{directory}/particles", part)
    if wrong:
        with pytest.raises(ValueError, match="product mismatch"):
            read_inputs(*inputs)
    else:
        assert len(read_inputs(*inputs)[2]) == 20


def test_reco_tag_and_finder_mode_fail_closed(expanded):
    producers = expanded["physics"]["producers"]
    producers["CrvRecoPulses"]["protonBunchTimeTag"] = "OtherTiming"
    with pytest.raises(ValueError, match="protonBunchTimeTag"):
        build_config(expanded)
    producers["CrvRecoPulses"]["protonBunchTimeTag"] = "EWMProducer"
    producers["CrvCoincidenceClusterFinder"]["addOppositeSidePulses"] = False
    with pytest.raises(ValueError, match="addOppositeSidePulses"):
        build_config(expanded)


def test_direct_timing_aligns_full_ids_and_keeps_mc_reco_separate(inputs):
    from mixing.crv_inputs import read_timing
    keys = KEYS[[2, 0, 3]]
    mc, reco, length = read_timing(keys, inputs[1], inputs[2])
    np.testing.assert_array_equal(mc, np.array([2., 0., 3.]) - 212.5)
    np.testing.assert_array_equal(reco, mc + .5)
    np.testing.assert_array_equal(length, [1695.] * 3)
    with pytest.raises(ValueError, match="absent"):
        read_timing(np.array([[9, 1, 1]]), inputs[1], inputs[2])
    with pytest.raises(ValueError, match="duplicate"):
        read_timing(KEYS[[0, 0]], inputs[1], inputs[2])


@pytest.mark.parametrize("mutation", ["duplicate", "spill", "nan", "length"])
def test_direct_timing_rejects_invalid_or_mismatched_census(inputs, mutation):
    from mixing.crv_inputs import read_timing
    events = table(inputs[2], "crvdump/events")
    if mutation == "duplicate":
        for k in IDS:
            events[k][1] = events[k][0]
    elif mutation == "spill":
        events["spill"][0] = 0
    elif mutation == "nan":
        events["pbtime"][0] = np.nan
    else:
        events["evlen"][0] += 1
    write_tree(inputs[2], "crvdump/events", events)
    with pytest.raises(ValueError):
        read_timing(KEYS, inputs[1], inputs[2])
