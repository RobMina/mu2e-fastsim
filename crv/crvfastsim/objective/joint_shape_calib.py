"""Model-implied gain ratios from existing measured pulse fields; never a calibration fit/update."""
import numpy as np

from . import joint_compare as J, bars as G


def grouped_quantiles(channel, values):
    """Finite one-dimensional values, vectorized exact linear quantiles per observed channel."""
    channel, values = np.asarray(channel), np.asarray(values)
    J.require(channel.ndim == values.ndim == 1 and channel.shape == values.shape and
              channel.dtype.kind in 'iu' and np.all(np.isfinite(values)), 'invalid grouped ratio inputs')
    if not len(channel):
        return dict(channel=np.empty(0, int), count=np.empty(0, int), quantiles=np.empty((0, 5)))
    order = np.lexsort((values, channel))
    c, v = channel[order], values[order]
    keys, start, count = np.unique(c, return_index=True, return_counts=True)
    frac = (count[:, None]-1)*np.array([0., .16, .5, .84, 1.])[None, :]
    lo, hi = np.floor(frac).astype(int), np.ceil(frac).astype(int)
    q = v[start[:, None]+lo]*(1-(frac-lo)) + v[start[:, None]+hi]*(frac-lo)
    return dict(channel=keys, count=count, quantiles=q)


def audit(h, calibration):
    """H/PEheight and e*H*beta/PEarea assume the recorded Gumbel fit conventions."""
    table = np.asarray(calibration)
    J.require(table.ndim == 2 and table.shape[1] == 9, 'expected frozen nine-column calibration')
    channels = table[:, 0]
    J.require(np.all(channels == np.floor(channels)) and np.all(channels >= 0), 'invalid calibration channels')
    J._unique(channels, 'calibration channel')
    order = np.argsort(channels)
    channels, table = channels[order].astype(np.int64), table[order]
    valid = np.ones(len(h['channel']), bool)
    for name in ('pe', 'height_pe', 'pulse_height', 'beta'):
        valid &= np.isfinite(h[name]) & (h[name] > 0)
    c = h['channel'][valid]
    area = np.e*h['pulse_height'][valid]*h['beta'][valid]/h['pe'][valid]
    height = h['pulse_height'][valid]/h['height_pe'][valid]
    valid_ratio = np.isfinite(area) & np.isfinite(height)
    c, area, height = c[valid_ratio], area[valid_ratio], height[valid_ratio]
    aa, hh = grouped_quantiles(c, area), grouped_quantiles(c, height)
    J.require(np.array_equal(aa['channel'], hh['channel']), 'gain channel sets differ')
    index = np.searchsorted(channels, aa['channel'])
    J.require(np.all(index < len(channels)) and np.array_equal(channels[index], aa['channel']), 'missing frozen gain')
    J.require(np.all(table[index, 2:4] > 0) and np.all(np.isfinite(table[index, 2:4])), 'invalid frozen gains')
    rows = []
    for i, ch in enumerate(aa['channel']):
        a, b = aa['quantiles'][i], hh['quantiles'][i]
        rows.append(dict(channel=int(ch), pulses=int(aa['count'][i]),
                         implied_area_q0_q16_q50_q84_q100=a.tolist(),
                         implied_height_q0_q16_q50_q84_q100=b.tolist(),
                         frozen_mc_area=float(table[index[i], 2]), frozen_mc_height=float(table[index[i], 3]),
                         area_median_ratio_to_mc=float(a[2]/table[index[i], 2]),
                         height_median_ratio_to_mc=float(b[2]/table[index[i], 3]),
                         area_full_range_relative=float((a[-1]-a[0])/a[2]),
                         height_full_range_relative=float((b[-1]-b[0])/b[2])))
    _, body, _ = G.bar_slot_body_layer(aa['channel']//4)
    pooled = {}
    for b, name in enumerate(G.BODY_NAMES):
        selected = np.flatnonzero(body == b)
        pooled[name] = dict(channels=len(selected), pulses=int(aa['count'][selected].sum()),
            area_ratio_channel_medians=J.stats([rows[i]['area_median_ratio_to_mc'] for i in selected]),
            height_ratio_channel_medians=J.stats([rows[i]['height_median_ratio_to_mc'] for i in selected]),
            max_area_channel_full_range_relative=max((rows[i]['area_full_range_relative'] for i in selected), default=None),
            max_height_channel_full_range_relative=max((rows[i]['height_full_range_relative'] for i in selected), default=None))
    return dict(population='Original selected timing members, finite positive PE/heightPE/height/beta; not pair-support weighted.',
                excluded_pulses=int(len(valid)-len(c)), channels=rows, bodies=pooled,
                formulas={'area': 'e * pulseHeight * pulseBeta / PEs', 'height': 'pulseHeight / PEsPulseHeight'},
                calibration_changed=False, historical_constants_certified=False,
                caution='Algebraically model-implied gains, not independently retrieved historical DB payload, noise measurement, correction or timing-effect estimate.')
