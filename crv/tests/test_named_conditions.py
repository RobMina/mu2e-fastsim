"""Synthetic named-conditions closure, source binding, and fail-closed loading."""
import dataclasses
import json
from pathlib import Path

import numpy as np
import pytest
import uproot

from crvfastsim import chain, named_conditions as NC, runconfig


@pytest.fixture
def bundle(tmp_path):
    counters = tmp_path / "geometry.txt"
    counters.write_text('''vector<string> crs.sectorNames = { "T1" };
int crs.nLayers = 1;
double crs.scintillatorBarThickness = 20;
double crs.scintillatorBarWidth = 50;
double crs.gapSmall = 1;
double crs.gapLarge = 2;
double crs.gapBetweenModules = 3;
vector<double> crs.gapBetweenLayers = {};
double crs.layerOffset = 0;
vector<double> crs.layerDirectionT1 = {0,1,0};
vector<double> crs.gapDirectionT1 = {1,0,0};
vector<double> crs.offsetDirectionT1 = {1,0,0};
double crs.scintillatorBarLengthT1 = 6000;
int crs.nModulesT1 = 1;
int crs.nCountersPerModuleT1 = 2;
vector<double> crs.firstCounterT1 = {0,0,0};
''')
    rc = dataclasses.replace(runconfig.get("run1a_v01"), counters=str(counters))
    geom = rc.geometry()
    n = 4 * geom.nbars
    source = tmp_path / "source"
    source.mkdir()
    (source / "calib_source.txt").write_text("TABLE CRVTime\n#channel,timeOffset\n" +
        "".join(f"{i},0.123\n" for i in range(n)) + "TABLE CRVSiPM\n" +
        "".join(f"{i},100,11.4,394.6\n" for i in range(n)))
    (source / "status_source.txt").write_text("TABLE CRVBadChan\n2,5\n")
    (source / "photonyield_source.txt").write_text("TABLE CRVPhoton\n" +
        "".join(f"{i},0.007\n" for i in range(n)))
    (source / "ordinal_source.txt").write_text("Channel ROC FEB FEBchannel\n" +
        "".join(f"{i} 1 1 {i}\n" for i in range(n - 1)))
    (source / "resolved.fcl").write_text('''services: {
  DbService: { purpose: "Sim_best" version: "v1_5"
    textFile: ["Offline/CRVConditions/data/calib_run1a_v01.txt",
               "Offline/CRVConditions/data/status_run1a_v01.txt",
               "Offline/CRVConditions/data/photonyield_run1a_v01.txt"] }
  ProditionsService: {
    crvOrdinal: {useDb: false filePath: "Offline/CRVConditions/data"}
    crvCalib: {useDb: true}
    crvStatus: {useDb: true}
    crvPhotonYield: {useDb: true}
  }
}
''')
    (source / "stock.log").write_text("synthetic stock log\n")
    rebind(source)
    a, _ = NC._source_payload(source, n)
    cal = {key: a[:, j].astype(np.int32 if key in ("channel", "status", "roc", "feb", "febch") else np.float32)
           for j, key in enumerate(NC.CALIB)}
    cal["run"] = np.full(n, 1400, dtype=np.int32)
    ev = {"run": np.array([1400, 1400]), "subrun": np.array([101, 101]), "event": np.array([12, 15]),
          "spill": np.array([1, 1]), "evlen": np.array([1700., 1700.]), "pbtime": np.array([0., 0.])}
    write_root(source, cal, ev)
    return source, rc, geom, tmp_path / "output"


def rebind(source):
    filenames = {"calib_source.txt": "calib_run1a_v01.txt", "status_source.txt": "status_run1a_v01.txt",
                 "photonyield_source.txt": "photonyield_run1a_v01.txt", "ordinal_source.txt": "run1a_v01.txt"}
    (source / "manifest.txt").write_text("area=/synthetic\n" + "".join(
        f"{NC._sha(source / key)}  /cvmfs/test/Offline/CRVConditions/data/{val}\n" for key, val in filenames.items()))


def write_root(source, cal, ev):
    with uproot.recreate(source / "crvreco.root") as root:
        root.mktree("crvdump/calib", cal)
        root.mktree("crvdump/events", ev)


def mutate_root(source, table, fn):
    with uproot.open(source / "crvreco.root") as root:
        cal = root["crvdump/calib"].arrays(library="np")
        ev = root["crvdump/events"].arrays(library="np")
    fn(cal if table == "calib" else ev)
    write_root(source, cal, ev)


def test_roundtrip(bundle):
    source, rc, geom, out = bundle
    manifest = NC.export_conditions(source, rc, out, geom)
    cond, meta = NC.load_conditions(manifest, rc, geom)
    legacy = chain.Conditions.from_csv(out / "calib.csv", out / "photon.csv")
    for field in dataclasses.fields(cond):
        np.testing.assert_array_equal(getattr(cond, field.name), getattr(legacy, field.name))
    assert cond.pulsearea[0] == float(np.float32(394.6))
    assert cond.status.tolist() == [0, 0, 5, 0, 0, 0, 0, 0]
    assert cond.roc[-1] == cond.feb[-1] == -1
    assert meta["stock_event_identity"]["first_event"] == 12
    assert "not a database IOV" in meta["scope"]
    with pytest.raises(ValueError, match="output already exists"):
        NC.export_conditions(source, rc, out, geom)


@pytest.mark.parametrize("table,key,idx,value,match", [
    ("calib", "channel", 1, 0, "duplicate"),
    ("calib", "channel", 1, 8, "out-of-range"),
    ("calib", "pulsearea", 0, 0, "positive"),
    ("calib", "pulseheight", 0, np.nan, "nonfinite"),
    ("calib", "status", 0, -1, "integer"),
    ("calib", "status", 0, 4, "mismatch"),
    ("calib", "run", 0, 1430, "calib run"),
    ("calib", "timeoffset", 0, 0.124, "float32"),
    ("calib", "roc", 0, -1, "partial unmapped"),
    ("calib", "febch", 1, 0, "duplicate mapped"),
    ("calib", "febch", 1, 64, "range"),
    ("events", "subrun", 1, 102, "one stock run/subrun"),
    ("events", "run", 0, 1430, "one stock run/subrun"),
    ("events", "event", 1, 12, "duplicate stock event"),
])
def test_stock_rejections(bundle, table, key, idx, value, match):
    source, rc, geom, out = bundle
    mutate_root(source, table, lambda x: x[key].__setitem__(idx, value))
    with pytest.raises(ValueError, match=match):
        NC.export_conditions(source, rc, out, geom)
    assert not out.exists()


@pytest.mark.parametrize("file,before,after,match", [
    ("photonyield_source.txt", "7,0.007\n", "", "coverage"),
    ("photonyield_source.txt", "7,0.007", "0,0.007", "duplicate"),
    ("photonyield_source.txt", "7,0.007", "7,nan", "nonfinite"),
    ("photonyield_source.txt", "7,0.007", "7,1e100", "float32"),
    ("calib_source.txt", "TABLE CRVTime", "TABLE Wrong", "unexpected"),
    ("calib_source.txt", "0,0.123", "0,0.123,4", "column count"),
    ("status_source.txt", "2,5", "2.5,5", "integer"),
    ("status_source.txt", "2,5", "2,5\n2,5", "duplicate"),
    ("status_source.txt", "2,5", "2,-1", "integer"),
    ("ordinal_source.txt", "1 1 1 1", "1 1 1 0", "duplicate mapped"),
    ("ordinal_source.txt", "1 1 1 1", "1 -1 -1 -1", "integer"),
    ("ordinal_source.txt", "Channel", "NotChannel", "header"),
    ("ordinal_source.txt", "6 1 1 6", "6 1 2 6", "mismatch"),
])
def test_source_rejections(bundle, file, before, after, match):
    source, rc, geom, out = bundle
    path = source / file
    path.write_text(path.read_text().replace(before, after))
    rebind(source)
    with pytest.raises(ValueError, match=match):
        NC.export_conditions(source, rc, out, geom)
    assert not out.exists()


@pytest.mark.parametrize("before,after,match", [
    ("photonyield_run1a_v01.txt", "photonyield_run1a_v01_perfect.txt", "missing/ambiguous"),
    ("crvPhotonYield: {useDb: true}", "crvPhotonYield: {useDb: false}", "must be true"),
    ('version: "v1_5"', 'version: "v1_3"', "Sim_best"),
    ('version: "v1_5"', 'version: "v1_5" version: "v1_5"', "duplicate"),
    ('filePath: "Offline/CRVConditions/data"', 'filePath: "different"', "backing-path"),
])
def test_resolved_binding_rejections(bundle, before, after, match):
    source, rc, geom, out = bundle
    p = source / "resolved.fcl"
    p.write_text(p.read_text().replace(before, after))
    with pytest.raises(ValueError, match=match):
        NC.export_conditions(source, rc, out, geom)


def test_source_bytes_must_match_backing_receipt(bundle):
    source, rc, geom, out = bundle
    p = source / "photonyield_source.txt"
    p.write_text(p.read_text() + "# changed bytes\n")
    with pytest.raises(ValueError, match="backing-path"):
        NC.export_conditions(source, rc, out, geom)


@pytest.mark.parametrize("name", [*NC.INPUTS, "calib.csv", "photon.csv", "counters.txt"])
def test_hash_drift(bundle, name):
    source, rc, geom, out = bundle
    manifest = NC.export_conditions(source, rc, out, geom)
    p = out / ("inputs" if name in NC.INPUTS else "") / name
    with p.open("ab") as stream:
        stream.write(b"\n")
    with pytest.raises(ValueError, match="hash/size"):
        NC.load_conditions(manifest, rc, geom)


def test_loader_identity_and_payload(bundle):
    source, rc, geom, out = bundle
    manifest = NC.export_conditions(source, rc, out, geom)
    with pytest.raises(ValueError, match="configuration"):
        NC.load_conditions(manifest, dataclasses.replace(rc, photon_yield_scale=0.1), geom)
    changed = dataclasses.replace(geom, center=geom.center + 1)
    with pytest.raises(ValueError, match="geometry/counters"):
        NC.load_conditions(manifest, rc, changed)
    m = json.loads(manifest.read_text())
    a = np.loadtxt(out / "calib.csv", delimiter=",")
    a[0, 1] += 1
    np.savetxt(out / "calib.csv", a, delimiter=",")
    m["assets"]["calib.csv"] = NC._record(out / "calib.csv", "calib.csv")
    manifest.write_text(json.dumps(m))
    with pytest.raises(ValueError, match="payload mismatch"):
        NC.load_conditions(manifest, rc, geom)


def test_parser_rejects_extra_tables_and_accepts_empty_status(tmp_path):
    p = tmp_path / "status"
    p.write_text("TABLE CRVBadChan\n")
    assert NC._text_tables(p, {"CRVBadChan": 2})["CRVBadChan"].shape == (0, 2)
    p.write_text("TABLE CRVBadChan\nTABLE CRVPhoton\n")
    with pytest.raises(ValueError, match="unexpected table"):
        NC._text_tables(p, {"CRVBadChan": 2})


def test_full_width_event_identity(bundle):
    source, rc, geom, out = bundle
    mutate_root(source, "events", lambda x: x["event"].__setitem__(1, 2**40 + 1))
    m = NC.export_conditions(source, rc, out, geom)
    assert NC.load_conditions(m, rc, geom)[1]["stock_event_identity"]["event_max"] == 2**40 + 1


def test_missing_stock_channel(bundle):
    source, rc, geom, out = bundle
    mutate_root(source, "calib", lambda x: x.update({k: v[:-1] for k, v in x.items()}))
    with pytest.raises(ValueError, match="coverage"):
        NC.export_conditions(source, rc, out, geom)


def test_shuffled_stock_and_float32_comparison(bundle):
    source, rc, geom, out = bundle
    def modify(cal):
        cal.update({k: v[::-1] for k, v in cal.items()})
        cal["pulsearea"] = np.full(len(cal["run"]), 394.6, dtype=float)
    mutate_root(source, "calib", modify)
    manifest = NC.export_conditions(source, rc, out, geom)
    assert NC.load_conditions(manifest, rc, geom)[0].pulsearea[0] == float(np.float32(394.6))


@pytest.mark.parametrize("field", ["run_config", "stock_event_identity", "config_sha256", "scope"])
def test_manifest_identity_drift(bundle, field):
    source, rc, geom, out = bundle
    manifest = NC.export_conditions(source, rc, out, geom)
    m = json.loads(manifest.read_text())
    m[field] = "tampered"
    manifest.write_text(json.dumps(m))
    with pytest.raises(ValueError):
        NC.load_conditions(manifest, rc, geom)


def test_loader_path_and_duplicate_keys(bundle):
    source, rc, geom, out = bundle
    manifest = NC.export_conditions(source, rc, out, geom)
    original = manifest.read_text()
    m = json.loads(original)
    m["assets"]["calib.csv"]["path"] = "../calib.csv"
    manifest.write_text(json.dumps(m))
    with pytest.raises(ValueError, match="asset path"):
        NC.load_conditions(manifest, rc, geom)
    manifest.write_text(original[:-2] + ', "schema": "duplicate"}\n')
    with pytest.raises(ValueError, match="duplicate JSON"):
        NC.load_conditions(manifest, rc, geom)


def test_dangling_output_link_refused(bundle):
    source, rc, geom, out = bundle
    out.symlink_to(out.parent / "missing")
    with pytest.raises(ValueError, match="output already exists"):
        NC.export_conditions(source, rc, out, geom)


def test_cli(bundle, monkeypatch, capsys):
    source, rc, geom, out = bundle
    monkeypatch.setattr(runconfig, "get", lambda name: rc)
    NC.main(["export", "--source", str(source), "--run-config", rc.name, "--outdir", str(out)])
    assert str(out / "manifest.json") in capsys.readouterr().out
    NC.load_conditions(out / "manifest.json", rc, geom)


def test_literal_source_integer_required(bundle):
    source, rc, geom, out = bundle
    p = source / "photonyield_source.txt"
    p.write_text(p.read_text().replace("1,0.007", "1.00000000000000001,0.007"))
    rebind(source)
    with pytest.raises(ValueError, match="literal integer"):
        NC.export_conditions(source, rc, out, geom)


def test_fcl_flattening_and_unresolved_refusal(tmp_path):
    p = tmp_path / "resolved.fcl"
    p.write_text('services.DbService.version: "v1_5"\nservices: { DbService: {purpose: "Sim_best"} }')
    assert NC._fcl_values(p) == {"services.DbService.version": "v1_5", "services.DbService.purpose": "Sim_best"}
    p.write_text('services.DbService: @local::DbService')
    with pytest.raises(ValueError, match="unresolved"):
        NC._fcl_values(p)


def test_bundle_relocation(bundle, tmp_path):
    import shutil
    source, rc, geom, out = bundle
    NC.export_conditions(source, rc, out, geom)
    moved = tmp_path / "relocated"
    out.rename(moved)
    shutil.rmtree(source)
    cond, _ = NC.load_conditions(moved / "manifest.json", rc, geom)
    assert len(cond.deviation) == 4 * geom.nbars
