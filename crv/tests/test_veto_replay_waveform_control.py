"""Keyed waveform replay contracts, using only synthetic waveform assets."""
from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from crvfastsim.reco import pulsefit as PF, waveform as WF
from crvfastsim.veto import replay_waveform_control as R


@pytest.fixture
def channels():
    return pd.DataFrame({
        "event": [1, 1, 1, 1, 2, 1], "chan": [0, 1, 4, 8, 0, 12],
        "roc": [1, 1, 1, 2, 1, 1], "feb": [2, 2, 3, 2, 2, 2],
        "pedestal": [100.] * 6, "pulsearea": [1000.] * 6,
        "pulseheight": [20.] * 6, "timeoffset": [2.] * 6,
        "status": [0, 2, 0, 0, 0, 0], "w": [-205.] * 6,
    })


@pytest.fixture
def synthetic(monkeypatch):
    cfg = WF.WaveformConfig()
    t = np.arange(0., cfg.single_pe_max_time, cfg.single_pe_precision)
    template = .06 * np.exp(-.5 * ((t - 35.) / 14.) ** 2)
    monkeypatch.setattr(WF, "single_pe_template", lambda *args: template.copy())
    return cfg, template


@pytest.fixture
def charges(synthetic):
    cfg, _ = synthetic
    rows = [(1, 0, 310.25), (1, 0, 330.75), (1, 0, 900.25),
            (1, 1, 320.25), (2, 0, 420.5)]
    rows += [(1, 4, t) for t in np.arange(300.25, 661., 60.)]
    q = pd.DataFrame(rows, columns=["event", "chan", "time"])
    q["charge"] = cfg.single_pe_reference_charge
    return q


def one_charge(cfg, time=310.25, charge=None):
    return pd.DataFrame({"event": [1], "chan": [1], "time": [time],
                         "charge": [cfg.single_pe_reference_charge if charge is None else charge]})


def canonical(frame, keys):
    return frame.sort_values(keys).reset_index(drop=True)


def test_phases_are_event_roc_feb_keyed_including_quiet_channels(channels):
    phases = R.phase_table(channels, 71)
    assert phases.phase.between(0., 12.5, inclusive="left").all()
    assert phases.groupby(["event", "roc", "feb"]).phase.nunique().eq(1).all()
    assert phases.phase.nunique() == 4
    assert phases.loc[phases.chan == 12, "phase"].iloc[0] == phases.phase.iloc[0]
    np.testing.assert_array_equal(phases.label, channels.event * R.MULT + channels.chan)
    for reduced in (channels.sample(frac=1, random_state=8), channels.iloc[[1, 2, 4]]):
        actual = R.phase_table(reduced, 71).set_index("label")
        expected = phases.set_index("label").loc[actual.index]
        pd.testing.assert_frame_equal(actual, expected, check_exact=True)
    assert not np.array_equal(phases.phase, R.phase_table(channels, 72).phase)


def test_noise_is_keyed_by_channel_and_absolute_tdc_not_array_position():
    tdc = np.array([-3, 0, 25, 26, 1000], dtype=np.int64)
    baseline = R.sample_noise(71, 1, 4, tdc, .0004)
    permutation = np.array([4, 1, 0, 3, 2])
    np.testing.assert_array_equal(R.sample_noise(71, 1, 4, tdc[permutation], .0004), baseline[permutation])
    np.testing.assert_array_equal(R.sample_noise(71, 1, 4, tdc[[1, 3]], .0004), baseline[[1, 3]])
    matrix = R.sample_noise(71, np.array([2, 1])[:, None], np.array([4, 4])[:, None], tdc, .0004)
    np.testing.assert_array_equal(matrix[1], baseline)
    assert not np.array_equal(matrix[0], baseline)
    assert not np.array_equal(R.sample_noise(71, 1, 5, tdc, .0004), baseline)
    np.testing.assert_array_equal(R.sample_noise(71, 1, 4, tdc, 0.), np.zeros(len(tdc)))


def test_synthesis_matches_production_and_independent_sample_sum(channels, charges, synthetic):
    cfg, template = synthetic
    D, samples, clusters, conditions = R.synthesize(charges, channels, 71, cfg)
    assert len(D["chan"]) > 0 and len(clusters) == 5
    q = charges.assign(label=R.labelled(charges)).sort_values(["label", "time"], kind="stable")
    phases = conditions.set_index("label").phase
    noise = samples.noise.to_numpy().reshape(len(clusters), -1)

    class NoiseOracle:
        def normal(self, loc, scale, size):
            assert loc == 0 and scale == cfg.noise and tuple(size) == noise.shape
            return noise.copy()

    expected = WF.synthesize(q.label.to_numpy(), q.time.to_numpy(), q.charge.to_numpy(),
                             lambda labels: phases.loc[labels].to_numpy(), cfg,
                             window=(200., 1500.), rng=NoiseOracle())
    assert D.keys() == expected.keys()
    for key in D:
        np.testing.assert_array_equal(D[key], expected[key], err_msg=key)
    for cluster in clusters.itertuples():
        s = samples[samples.cluster == cluster.cluster]
        local = q[(q.event == cluster.event) & (q.chan == cluster.chan)
                  & q.time.between(cluster.first_charge, cluster.last_charge)]
        voltage = np.zeros(len(s))
        for charge in local.itertuples():
            elapsed = s.time_relative.to_numpy() - charge.time
            index = np.rint(elapsed / cfg.single_pe_precision).astype(int)
            keep = (elapsed >= 0) & (index >= 0) & (index < len(template)) & cluster.prekeep
            voltage[keep] += template[index[keep]] * charge.charge / cfg.single_pe_reference_charge
        np.testing.assert_allclose(s.voltage_noiseless, voltage, rtol=1e-15, atol=0)
        np.testing.assert_array_equal(s.voltage, s.voltage_noiseless + s.noise * s.valid)
        np.testing.assert_array_equal(s.above_zs, s.valid & s.cluster_prekeep & (s.voltage > cfg.min_voltage))
        np.testing.assert_array_equal(s.precast_adc, np.floor(s.voltage * cfg.adc_conversion + cfg.pedestal + .5))
        assert s.valid.sum() == cluster.length
    assert (~samples.valid).any()
    np.testing.assert_array_equal(samples.loc[~samples.valid, "voltage"], 0.)
    index = samples[samples.valid].set_index(["event", "chan", "tdc"])
    assert index.index.is_unique
    for i, label in enumerate(D["chan"]):
        keys = [(label // R.MULT, label % R.MULT, D["starttdc"][i] + j) for j in range(cfg.samples_zs)]
        np.testing.assert_array_equal(D["volt"][i], index.voltage.reindex(keys, fill_value=0.))


def test_unchanged_channel_exact_under_shuffle_removal_and_padding(channels, charges, synthetic):
    cfg, _ = synthetic
    baseline = R.synthesize(charges, channels, 71, cfg)
    variants = [
        (charges.sample(frac=1, random_state=6), channels.sample(frac=1, random_state=4)),
        (charges[charges.chan != 0], channels[channels.chan != 0]),
        (charges[charges.chan == 1], channels[channels.chan == 1]),
        (charges, channels, replace(cfg, samples_zs=16)),
    ]
    target = (baseline[1].event == 1) & (baseline[1].chan == 1) & baseline[1].valid
    expected = canonical(baseline[1][target].drop(columns="cluster"), ["tdc"])
    widths = []
    for variant in variants:
        q, c, *config = variant
        actual = R.synthesize(q, c, 71, config[0] if config else cfg)
        widths.append(len(actual[1]) // len(actual[2]))
        s = actual[1]
        s = s[(s.event == 1) & (s.chan == 1) & s.valid].drop(columns="cluster")
        pd.testing.assert_frame_equal(canonical(s, ["tdc"]), expected, check_exact=True)
        if not config:
            for key in ("chan", "start", "tdc0", "starttdc", "volt", "adc"):
                a = actual[0][key][actual[0]["chan"] == R.MULT + 1]
                b = baseline[0][key][baseline[0]["chan"] == R.MULT + 1]
                np.testing.assert_array_equal(a, b)
    assert widths[2] < widths[0]
    assert widths[3] == widths[0] + 4


def test_digi_start_window_inclusive_and_tail_not_clipped(channels, synthetic):
    cfg, _ = synthetic
    cfg = replace(cfg, noise=0.)
    q = one_charge(cfg)
    baseline = R.synthesize(q, channels, 71, cfg)[0]
    assert len(baseline["chan"]) == 1
    start = baseline["start"][0]
    for window in ((start, start), (start, start + 1), (start - 1, start)):
        D = R.synthesize(q, channels, 71, cfg, window=window)[0]
        for key in ("start", "starttdc", "volt", "adc"):
            np.testing.assert_array_equal(D[key], baseline[key])
        assert D["volt"].shape == (1, cfg.samples_zs)
        assert D["start"][0] + (cfg.samples_zs - 1) * 12.5 > window[1]
        assert (D["volt"][0, 1:] > 0).any()
    epsilon = np.nextafter(start, np.inf)
    assert len(R.synthesize(q, channels, 71, cfg, window=(epsilon, epsilon))[0]["chan"]) == 0


def test_reconstruct_full_recovers_beyond_default_cap_and_preserves_ledger(channels):
    start = 20
    t = (start + np.arange(216)) * PF.PERIOD
    voltage = sum(PF.gumbel(t, 600., (start + 6 + 24 * i) * PF.PERIOD, 19.) for i in range(9))
    adc = np.rint(100. + voltage).astype(np.int16)
    D = {"chan": np.full(18, R.MULT + 1, dtype=np.int64),
         "starttdc": start + np.arange(18) * 12, "adc": adc.reshape(18, 12)}
    D = {key: values[::-1].copy() for key, values in D.items()}
    pulses, waveforms, samples = R.reconstruct(D, channels)
    capped = R.reconstruct(D, channels, full=False)[0]
    assert len(pulses) == 9 and len(capped) == PF.RecoConfig().max_pulses == 8
    expected = PF.reconstruct_full(adc[None, :], np.array([216]), np.array([start]),
                                   np.array([100.]), np.array([1000.]), np.array([20.]))
    near = expected.pop("wf_near_threshold")
    for key, values in expected.items():
        np.testing.assert_array_equal(pulses[key], values)
    np.testing.assert_array_equal(pulses.event, 1)
    np.testing.assert_array_equal(pulses.chan, 1)
    np.testing.assert_array_equal(pulses.time_relative, pulses.time + 2.)
    np.testing.assert_array_equal(pulses.time_reco, pulses.time + 2. - 205.)
    assert pulses.channel_ignored.all()
    assert waveforms.n.tolist() == [216] and waveforms.starttdc.tolist() == [start]
    np.testing.assert_array_equal(waveforms.near_threshold, near)
    assert waveforms.peak_margin_adc.iloc[0] == adc.max() - 100 - PF.RecoConfig().min_adc_difference
    np.testing.assert_array_equal(samples.adc, adc)
    np.testing.assert_array_equal(samples.tdc, start + np.arange(216))


@pytest.mark.parametrize("empty_charges", [True, False])
def test_empty_charges_and_empty_digis(channels, synthetic, empty_charges):
    cfg, _ = synthetic
    q = one_charge(cfg, charge=0.)
    if empty_charges:
        q = q.iloc[:0]
    D, samples, clusters, conditions = R.synthesize(q, channels, 71, replace(cfg, noise=0.))
    assert D["adc"].shape == D["volt"].shape == (0, cfg.samples_zs)
    assert len(conditions) == len(channels)
    if empty_charges:
        assert samples.empty and clusters.empty
    else:
        assert len(samples) > 0 and len(clusters) == 1
    p, w, s = R.reconstruct(D, conditions)
    assert p.empty and w.empty and s.empty
    assert {"event", "chan", "time_relative", "time_reco", "channel_ignored"} <= set(p)
    assert {"tdc", "adc", "event", "chan", "wf"} <= set(s)


def test_extreme_amplitude_fails_safe_adc_comparison(channels, synthetic):
    cfg, _ = synthetic
    q = one_charge(cfg, charge=1000 * cfg.single_pe_reference_charge)
    with pytest.raises(ValueError, match="ADC narrowing"):
        R.synthesize(q, channels, 71, replace(cfg, noise=0.))


@pytest.mark.parametrize("fault", ["drop", "duplicate", "shift"])
def test_independent_zs_scan_rejects_missing_duplicate_shifted_digis(channels, synthetic, monkeypatch, fault):
    cfg, _ = synthetic
    original = WF.synthesize
    def corrupted(*args, **kwargs):
        d = original(*args, **kwargs)
        n = len(d["chan"])
        assert n > 0
        for key in list(d):
            if key.startswith("cluster_"):
                continue
            d[key] = d[key][:0] if fault == "drop" else np.concatenate([d[key], d[key][:1]]) if fault == "duplicate" else d[key]
        if fault == "shift":
            d["starttdc"] = d["starttdc"] + 1
        return d
    monkeypatch.setattr(WF, "synthesize", corrupted)
    with pytest.raises(ValueError, match="zero-suppression start"):
        R.synthesize(one_charge(cfg), channels, 71, cfg)


def test_waveform_length_bound_fails_closed(channels, synthetic):
    cfg, _ = synthetic
    q = pd.concat([one_charge(cfg, time=t) for t in np.arange(300., 14001., 100.)], ignore_index=True)
    with pytest.raises(ValueError, match="length bound"):
        R.synthesize(q, channels, 71, cfg)


@pytest.mark.parametrize("column,value", [("event", -1), ("event", 2**20), ("event", .5),
                                          ("chan", -1), ("chan", R.MULT), ("chan", 1.5),
                                          ("event", np.nan), ("chan", np.inf)])
def test_invalid_labels_rejected(column, value):
    frame = pd.DataFrame({"event": [1], "chan": [1]})
    frame[column] = value
    with pytest.raises(ValueError, match="event/channel"):
        R.labelled(frame)


def test_duplicate_or_missing_conditions_rejected(channels, synthetic):
    cfg, _ = synthetic
    with pytest.raises(ValueError, match="duplicate channel"):
        R.phase_table(pd.concat([channels, channels.iloc[:1]]), 71)
    with pytest.raises(ValueError, match="charge/conditions"):
        R.synthesize(one_charge(cfg).assign(chan=99), channels, 71, cfg)
    D = {"chan": np.array([R.MULT + 99]), "starttdc": np.array([20]),
         "adc": np.full((1, 12), 100, dtype=np.int16)}
    with pytest.raises(KeyError):
        R.reconstruct(D, channels)


@pytest.mark.parametrize("column,value", [("time", np.nan), ("charge", np.inf), ("charge", -1.)])
def test_invalid_charge_values_rejected(channels, synthetic, column, value):
    cfg, _ = synthetic
    with pytest.raises(ValueError, match="charge/conditions"):
        R.synthesize(one_charge(cfg).assign(**{column: value}), channels, 71, cfg)
