"""valhist.compare: merge keeps under/overflow, statistics, titles, class and Sumw2; compare reports flow fractions."""
import json

import numpy as np
import uproot
from uproot.writing import identify

from valhist import compare as VC

STEP = "Validation/CrvSteps_steps_noName"


def th1(values, lo, hi, title="t", sumw2=None, dtype=np.float64):
    v = np.asarray(values, dtype=dtype)
    return identify.to_TH1x(fName="h", fTitle=title, data=v, fEntries=float(v.sum()), fTsumw=float(v[1:-1].sum()),
                            fTsumw2=float(v[1:-1].sum()), fTsumwx=1.5, fTsumwx2=2.5,
                            fSumw2=None if sumw2 is None else np.asarray(sumw2, float),
                            fXaxis=identify.to_TAxis("xaxis", "y [mm]", len(v) - 2, lo, hi))


def th2(values):
    v = np.asarray(values, dtype=np.float64)
    nx, ny = v.shape[0] - 2, v.shape[1] - 2
    return identify.to_TH2x(fName="h", fTitle="map", data=v.ravel(order="F"), fEntries=float(v.sum()),
                            fTsumw=float(v.sum()), fTsumw2=float(v.sum()), fTsumwx=0, fTsumwx2=0, fTsumwy=0,
                            fTsumwy2=0, fTsumwxy=0, fSumw2=None,
                            fXaxis=identify.to_TAxis("xaxis", "", nx, 0, 1), fYaxis=identify.to_TAxis("yaxis", "", ny, 0, 1))


def write(path, objs):
    with uproot.recreate(path) as f:
        for k, o in objs.items():
            f[k] = o


def shards(tmp_path):
    # posy: all entries overflow (the extracted CRV is above the booked range); x: mixed; float TH1F with Sumw2
    a = {f"{STEP}/posy": th1([0, 0, 0, 0, 7], -2000, 3000, "Start position Y"),
         f"{STEP}/posx": th1([1, 2, 3, 0, 2], 0, 3, sumw2=[1, 2, 3, 0, 2]),
         f"{STEP}/E": th1([0, 4, 5, 0], 0, 1, dtype=np.float32),
         "crvRecoDQM/map": th2(np.arange(20, dtype=float).reshape(4, 5))}
    b = {f"{STEP}/posy": th1([0, 0, 0, 0, 5], -2000, 3000, "Start position Y"),
         f"{STEP}/posx": th1([0, 1, 1, 1, 0], 0, 3, sumw2=[0, 1, 1, 1, 0]),
         f"{STEP}/E": th1([1, 1, 1, 1], 0, 1, dtype=np.float32),
         "crvRecoDQM/map": th2(np.ones((4, 5)))}
    write(tmp_path / "s1.root", a)
    write(tmp_path / "s2.root", b)
    return a, b


def test_merge_keeps_flow_stats_class_and_sumw2(tmp_path):
    shards(tmp_path)
    out = tmp_path / "m.root"
    VC.merge(str(out), [str(tmp_path / "s*.root")])
    with uproot.open(out) as f:
        y = f[f"{STEP}/posy"]
        assert y.values(flow=True).tolist() == [0, 0, 0, 0, 12]
        assert y.values().sum() == 0
        assert y.title == "Start position Y"
        assert y.member("fEntries") == 12
        assert y.axis().member("fTitle") == "y [mm]"
        assert np.allclose(y.axis().edges(), np.linspace(-2000, 3000, 4))
        x = f[f"{STEP}/posx"]
        assert x.values(flow=True).tolist() == [1, 3, 4, 1, 2]
        assert x.variances(flow=True).tolist() == [1, 3, 4, 1, 2]
        assert x.member("fTsumwx") == 3.0
        e = f[f"{STEP}/E"]
        assert f.classnames()[f"{STEP}/E;1"] == "TH1F"
        assert e.values(flow=True).tolist() == [1, 5, 6, 1]
        m = f["crvRecoDQM/map"]
        assert np.array_equal(m.values(flow=True), np.arange(20, dtype=float).reshape(4, 5) + 1)


def test_merge_drops_sumw2_when_a_source_lacks_it(tmp_path):
    write(tmp_path / "s1.root", {"crvRecoDQM/h": th1([0, 1, 0], 0, 1, sumw2=[0, 1, 0])})
    write(tmp_path / "s2.root", {"crvRecoDQM/h": th1([0, 1, 0], 0, 1)})
    VC.merge(str(tmp_path / "m.root"), [str(tmp_path / "s*.root")])
    with uproot.open(tmp_path / "m.root") as f:
        assert len(f["crvRecoDQM/h"].member("fSumw2")) == 0


def test_merge_alias_and_compare_report_flow(tmp_path):
    shards(tmp_path)
    write(tmp_path / "c1.root", {k.replace("steps_noName", "fastCrvSteps_noName"): v
                                 for k, v in {f"{STEP}/posy": th1([0, 0, 0, 0, 9], -2000, 3000),
                                              f"{STEP}/posx": th1([0, 2, 2, 2, 1], 0, 3)}.items()})
    VC.merge(str(tmp_path / "ref.root"), [str(tmp_path / "s*.root")])
    VC.merge(str(tmp_path / "cand.root"), [str(tmp_path / "c1.root")],
             ["CrvSteps_fastCrvSteps_noName=CrvSteps_steps_noName"])
    s = VC.compare(str(tmp_path / "ref.root"), str(tmp_path / "cand.root"), tmp_path / "cmp")
    assert s["all_flow"] == [f"{STEP}/posy"]
    assert s["flow_fractions"][f"{STEP}/posy"] == [1.0, 1.0]
    assert s["flow_fractions"][f"{STEP}/posx"] == [3 / 11, 1 / 7]
    assert s["sets"]["Validation"]["all_flow"] == 1 and s["sets"]["Validation"]["with_flow"] == 2
    assert f"{STEP}/posy" not in s["flagged"]
    rows = {l.split("\t")[0]: dict(zip(h, l.split("\t")[1:]))
            for h in [(tmp_path / "cmp" / "report.tsv").read_text().split("\n")[0].split("\t")[1:]]
            for l in (tmp_path / "cmp" / "report.tsv").read_text().strip().split("\n")[1:]}
    assert rows[f"{STEP}/posy"]["status"] == "all_flow"
    assert float(rows[f"{STEP}/posy"]["over_ref"]) == 12 and float(rows[f"{STEP}/posy"]["over_cand"]) == 9
    assert json.loads((tmp_path / "cmp" / "summary.json").read_text())["all_flow"] == [f"{STEP}/posy"]
