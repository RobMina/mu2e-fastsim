"""Event-clustered EX adjacent-layer descendant correlation and tail moments."""
from __future__ import annotations

import numpy as np
import pandas as pd

from crvfastsim import tables, tier1b as T
from crvfastsim.analyses.census import classify
from . import steplayers as S


def fill(steps, parts, events, geom):
    s = classify(steps, parts, geom.sector)
    s["layer"] = T.layer_rank(geom)[s.bar.to_numpy()]
    s["group"] = T._group_index(geom, s.bar.to_numpy())
    mu = s[(s.cls == "mu_primary") & (s.group == 0)]
    desc = s[(s.cls == "mu_descendant") & (s.group == 0)]
    keys = tables.KEY + ["prim_key"]
    x = mu[keys + ["layer"]].drop_duplicates()
    d = desc.groupby(keys + ["layer"], sort=False).edep.sum().rename("value").reset_index()
    x = x.merge(d, on=keys + ["layer"], how="left").fillna({"value": 0.})
    next_ = x.rename(columns={"value": "y"}).assign(layer=lambda f: f.layer - 1)
    pair = x.rename(columns={"value": "x"}).merge(next_, on=keys + ["layer"], how="inner")
    if len(pair):
        a, b = pair.x.to_numpy(), pair.y.to_numpy()
        values = np.stack([np.ones(len(pair)), a, b, a*a, b*b, a*b], axis=1)
        ev = S.evkey(pair)
        _, inverse = np.unique(ev, return_inverse=True)
        per_event = np.zeros((int(inverse.max()) + 1, 6))
        np.add.at(per_event, inverse, values)
        first, second = per_event.sum(0), per_event.T @ per_event
    else:
        first, second = np.zeros(6), np.zeros((6, 6))
    count = steps.groupby(tables.KEY, sort=False).size().rename("count").reset_index()
    count = events.merge(count, on=tables.KEY, how="left").fillna({"count": 0.})["count"].to_numpy()
    thresholds = np.asarray([0, 1, 10, 30, 100, 300, 1000])
    return dict(correlation_first=first, correlation_second=second,
                tail_thresholds=thresholds, tail_counts=(count[:, None] >= thresholds).sum(0), events=len(events))


def add(acc, new):
    for k, v in new.items():
        if k == "tail_thresholds":
            if k in acc:
                np.testing.assert_array_equal(acc[k], v)
            acc[k] = v.copy()
        else:
            acc[k] = acc.get(k, 0) + v


def summarize(acc):
    total = np.asarray(acc["correlation_first"], float)
    n, x, y, x2, y2, xy = total
    if n == 0:
        return dict(correlation=None, correlation_se=None)
    mx, my = x/n, y/n
    vx, vy, cv = x2/n - mx*mx, y2/n - my*my, xy/n - mx*my
    if vx <= 0 or vy <= 0:
        return dict(correlation=None, correlation_se=None)
    den = np.sqrt(vx*vy)
    corr = cv/den
    dcov = np.asarray([0., -my, -mx, 0., 0., 1.])
    dvx, dvy = np.asarray([0., -2*mx, 0., 1., 0., 0.]), np.asarray([0., 0., -2*my, 0., 1., 0.])
    grad_mean = dcov/den - .5*corr*(dvx/vx + dvy/vy)
    gradient = grad_mean/n
    gradient[0] = -np.dot(grad_mean, total/n)/n
    variance = float(gradient @ np.asarray(acc["correlation_second"]) @ gradient)
    events = int(acc["events"])
    return dict(correlation=float(corr), correlation_se=float(np.sqrt(max(variance, 0.))),
                adjacent_pairs=int(n), tail_thresholds=np.asarray(acc["tail_thresholds"]).tolist(),
                tail_counts=np.asarray(acc["tail_counts"]).tolist(), events=events,
                tail_fractions=(np.asarray(acc["tail_counts"])/max(events, 1)).tolist())
