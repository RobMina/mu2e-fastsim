"""Per-path agreement figure and numbers from synthetic compare.py reports."""
import importlib.util
import json
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('trigger_plot_agreement', ROOT / 'trigger' / 'plot_agreement.py')
plot_agreement = importlib.util.module_from_spec(spec)
spec.loader.exec_module(plot_agreement)


def report(lumi_replay=3):
    row = {'bit': 425, 'stock_pass': 40, 'replay_pass': 40, 'mismatches': 0, 'stock_only': 0, 'replay_only': 0}
    lumi = {'bit': 800, 'stock_pass': 1, 'replay_pass': 1 + lumi_replay, 'mismatches': lumi_replay,
            'stock_only': 0, 'replay_only': lumi_replay}
    return {'events': 100, 'paths': {'calo_cluster_50': row, 'lumiStream': lumi, 'apr_proton': {**row, 'stock_pass': 0,
            'replay_pass': 0}}, 'mismatches': lumi_replay, 'nonstateful_mismatches': 0, 'stateful_paths': ['lumiStream']}


def test_summary_counts_exact_paths():
    s = plot_agreement.summary(report())
    assert (s['paths'], s['paths_exact'], s['event_path_decisions']) == (3, 2, 300)
    assert s['differing_paths'] == {'lumiStream': {'stock_pass': 1, 'replay_pass': 4, 'stock_only': 0, 'replay_only': 3}}


def test_cli_writes_png_and_numbers(tmp_path, monkeypatch):
    pytest.importorskip('matplotlib')
    ce, cosmic = tmp_path / 'ce.json', tmp_path / 'cosmic.json'
    ce.write_text(json.dumps(report()))
    cosmic.write_text(json.dumps(report(0)))
    out, numbers = tmp_path / 'fig.png', tmp_path / 'numbers.json'
    monkeypatch.setattr(sys, 'argv', ['plot_agreement.py', '--sample', f'CE={ce}', '--sample', f'Cosmic={cosmic}',
                                      '--cost', 'CE=8.2', '--output', str(out), '--numbers', str(numbers)])
    plot_agreement.main()
    assert out.stat().st_size > 0
    result = json.loads(numbers.read_text())
    assert result['CE']['stock_ms_per_event'] == '8.2' and result['Cosmic']['paths_exact'] == 3


def test_rejects_different_path_lists(tmp_path):
    pytest.importorskip('matplotlib')
    other = report()
    other['paths'].pop('apr_proton')
    with pytest.raises(ValueError, match='different path list'):
        plot_agreement.plot([('CE', report()), ('Cosmic', other)], {}, tmp_path / 'x.png')
