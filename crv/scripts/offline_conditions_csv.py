"""Offline CRV text conditions (CRVConditions/data) -> the runchain conditions CSVs (--calib, --deviation).

    python3 scripts/offline_conditions_csv.py --calib calib_run1a_v01.txt --status status_run1a_v01.txt \
        --photonyield photonyield_run1a_v01.txt --ordinal run1a_v01.txt --out-calib C.csv --out-deviation D.csv

calib: tables CRVTime (channel,timeOffset) and CRVSiPM (channel,pedestal,calibPulseHeight,calibPulseArea); status:
CRVBadChan (channel,status); photonyield: CRVPhoton (channel,photonYieldDeviation); ordinal: the channel map
(Channel ROC FEB FEBchannel, whitespace separated). Output columns as chain.Conditions.from_csv reads them:
channel,pedestal,pulsearea,pulseheight,timeoffset,status,roc,feb,febch, and channel,deviation.
"""
import argparse

import numpy as np


def tables(path):
    """{TABLE name: rows of floats} of an Offline DbService text file (comment lines start with #)."""
    out, cur = {}, None
    for line in open(path):
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("TABLE"):
            cur = out.setdefault(line.split()[1], [])
            continue
        cur.append([float(v) for v in line.replace(",", " ").split()])
    return {k: np.asarray(v, float).reshape(len(v), -1) for k, v in out.items()}


def ordinal(path):
    rows = [l.split() for l in open(path) if l.strip() and l.split()[0].isdigit()]
    return np.asarray(rows, np.int64)


def convert(calib, status, photonyield, ordinal_map):
    """(calib rows [n, 9], deviation rows [n, 2]) over the channels of the SiPM table."""
    c = tables(calib)
    sipm, time = c["CRVSiPM"], c["CRVTime"]
    n = int(sipm[:, 0].max()) + 1
    ch = np.arange(n)

    def dense(rows, j):
        a = np.zeros(n)
        a[rows[:, 0].astype(int)] = rows[:, j]
        return a

    st = tables(status).get("CRVBadChan", np.zeros((0, 2)))
    om = ordinal(ordinal_map)
    roc, feb, febch = (dense(om.astype(float), j) for j in (1, 2, 3))
    calib_rows = np.c_[ch, dense(sipm, 1), dense(sipm, 3), dense(sipm, 2), dense(time, 1), dense(st, 1), roc, feb, febch]
    dev = tables(photonyield)["CRVPhoton"]
    return calib_rows, np.c_[ch, dense(dev, 1)]


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    for a in ("calib", "status", "photonyield", "ordinal", "out-calib", "out-deviation"):
        p.add_argument(f"--{a}", required=True)
    a = p.parse_args()
    c, d = convert(a.calib, a.status, a.photonyield, a.ordinal)
    np.savetxt(a.out_calib, c, delimiter=",", fmt=["%d", "%.4f", "%.4f", "%.4f", "%.4f", "%d", "%d", "%d", "%d"],
               header="channel,pedestal,pulsearea,pulseheight,timeoffset,status,roc,feb,febch")
    np.savetxt(a.out_deviation, d, delimiter=",", fmt=["%d", "%.4f"], header="channel,deviation")
    print(f"{len(c)} channels, {int((c[:, 5] != 0).sum())} with a status bit, deviation sd {d[:, 1].std():.4f}")


if __name__ == "__main__":
    main()
