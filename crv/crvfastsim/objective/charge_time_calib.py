"""Profile channel gains and timing offsets in a smooth CRV response fit.

Inputs are matched, positive-charge observations and the expected PE and time from a
nominal response model. For distance d along the fibre, fit

    log(observed PE / expected PE) = log_scale + log_gain[channel] - delta_attenuation * d
    observed time - expected time = time_zero + time_offset[channel] + delta_slowness * d

The two sets of channel terms have zero mean over the observed channels. The
corrections are relative to the nominal model, not absolute transport constants.
The model does not describe threshold efficiency, missed hits or saturation.
"""
import argparse
import json

import numpy as np


def _profile(index, nchan, distance, residual, weights):
    weight_sum = np.bincount(index, weights=weights, minlength=nchan)
    if not np.isfinite(weight_sum).all() or np.any(weight_sum <= 0):
        raise ValueError("each observed channel needs finite positive total weight")
    xd = np.bincount(index, weights=weights * distance, minlength=nchan) / weight_sum
    yd = np.bincount(index, weights=weights * residual, minlength=nchan) / weight_sum
    x, y = distance - xd[index], residual - yd[index]
    denominator = np.dot(weights, x * x)
    if not np.isfinite(denominator) or denominator <= np.finfo(float).eps * np.dot(weights, distance * distance):
        raise ValueError("the shared slope requires within-channel distance variation")
    slope = np.dot(weights * x, y) / denominator
    intercepts = yd - slope * xd
    global_intercept = intercepts.mean()
    errors = y - slope * x
    return global_intercept, slope, intercepts - global_intercept, np.sqrt(np.average(errors**2, weights=weights))


def fit(channel, distance_mm, pe, expected_pe, time_ns, expected_time_ns, *, charge_weight=None, time_weight=None):
    """Return shared corrections and aligned per-channel gains and offsets.

    Weights, when provided, are inverse variances in log-PE and ns respectively.
    Select finite, unsaturated observations with reliable track distances before calling.
    Channel IDs and distance are held fixed; the solve is linear in log-gain, inverse
    attenuation length and inverse signal speed.
    """
    channel = np.asarray(channel)
    if channel.ndim != 1 or channel.size == 0 or channel.dtype.kind not in "iu" or np.any(channel < 0):
        raise ValueError("channel must contain nonnegative integer IDs")

    def values(array, name, *, positive=False):
        out = np.asarray(array, dtype=float)
        if out.shape != channel.shape or not np.isfinite(out).all() or (positive and np.any(out <= 0)):
            raise ValueError(f"{name} must be finite and aligned with channel" + (" and positive" if positive else ""))
        return out

    distance = values(distance_mm, "distance_mm")
    if np.any(distance < 0):
        raise ValueError("distance_mm must be nonnegative")
    channels, index = np.unique(channel, return_inverse=True)
    log_pe = np.log(values(pe, "pe", positive=True)) - np.log(values(expected_pe, "expected_pe", positive=True))
    dt = values(time_ns, "time_ns") - values(expected_time_ns, "expected_time_ns")
    wc = np.ones(len(channel)) if charge_weight is None else values(charge_weight, "charge_weight", positive=True)
    wt = np.ones(len(channel)) if time_weight is None else values(time_weight, "time_weight", positive=True)
    log_scale, atten_slope, log_gain, rms_log_pe = _profile(index, len(channels), distance, log_pe, wc)
    t0, slowness, offset, rms_time = _profile(index, len(channels), distance, dt, wt)
    return dict(channel=channels.tolist(), log_scale=float(log_scale), log_gain=log_gain.tolist(),
                delta_attenuation_per_mm=float(-atten_slope), time_zero_ns=float(t0), time_offset_ns=offset.tolist(),
                delta_slowness_ns_per_mm=float(slowness), rms_log_pe=float(rms_log_pe), rms_time_ns=float(rms_time))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", help="NPZ of aligned arrays named as the fit() arguments")
    parser.add_argument("--out", help="write the fit as JSON instead of printing it")
    args = parser.parse_args(argv)
    with np.load(args.input, allow_pickle=False) as data:
        result = fit(**data)
    text = json.dumps(result, indent=2, allow_nan=False)
    if args.out:
        with open(args.out, "w") as stream:
            stream.write(text + "\n")
    else:
        print(text)


if __name__ == "__main__":
    main()
