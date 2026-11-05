"""Existing-plugin runtime adaptation and stock-only bridge configurations fail closed."""
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'trigger'))
import adapter_probe
import plugin_runtime
import replay


CONFIG = '''Digitize:{services:{GeometryService:{inputFile:"production-geometry"} DbService:{purpose:Sim_best version:v1_5}}
physics:{producers:{makeSD:{module_type:StockDigi}}} }'''


def test_adapter_preserves_stock_context_and_creates_only_distinct_injected_digis():
    dump, inject, receipt = adapter_probe.prepare(CONFIG, '/private/input.art', '/private/stock.root')
    assert dump['physics']['analyzers']['trkdigi']['StrawDigiTag'] == 'makeSD::Digitize'
    assert inject['physics']['analyzers']['trkdigi']['compareTo'] == 'makeSD::Digitize'
    assert inject['physics']['producers'] == {'tableSD': {'module_type': 'StrawDigisFromTable', 'tableFile': '/private/stock.root'}}
    assert inject['outputs']['Combined']['outputCommands'] == ['keep *']
    assert 'SelectEvents' not in inject['outputs']['Combined']
    assert replay.plain(inject['services']['GeometryService']) == {'inputFile': 'production-geometry'}
    assert dump['services']['TFileService']['fileName'] == 'stock-digis.root'
    assert 'no fast physics' in receipt['scope']


@pytest.mark.parametrize('process', ['Digitize', 'TriggerReplay', 'bad:process'])
def test_adapter_never_shadows_existing_process(process):
    with pytest.raises(ValueError):
        adapter_probe.prepare(CONFIG, '/private/in.art', '/private/table.root', process)


def test_private_runtime_binds_existing_library_hashes_without_replacing_release(tmp_path):
    source = Path(__file__).resolve().parents[1] / 'container/offline/offline_uva.sh'
    lib = tmp_path / 'plugins'
    lib.mkdir()
    (lib / 'libtrkfastsim_Test_module.so').write_bytes(b'existing-binary')
    output = tmp_path / 'runtime'
    receipt = plugin_runtime.prepare(source, lib, output)
    adapted = (output / 'offline_plugins.sh').read_text()
    assert 'CET_PLUGIN_PATH="$OFFLINE_PLUGIN_DIR:' in adapted
    assert 'LD_LIBRARY_PATH="$OFFLINE_PLUGIN_DIR:' in adapted
    assert 'OFFLINE_WARM_REPEAT OFFLINE_PLUGIN_DIR;' in adapted
    assert 'muse setup ||' in adapted and 'mu2e --version' in adapted
    assert receipt['plugins'][0]['bytes'] == len(b'existing-binary')
    frozen = output / 'plugins/libtrkfastsim_Test_module.so'
    assert frozen.read_bytes() == b'existing-binary'
    (lib / 'libtrkfastsim_Test_module.so').write_bytes(b'owner-later-change')
    assert frozen.read_bytes() == b'existing-binary'
    assert 'sha256sum -c SHA256SUMS' in adapted
    assert 'compatibility' in receipt['scope']
    plugin_runtime.prepare(source, lib, tmp_path / 'geometry-runtime', capture_geometry=True)
    captured = (tmp_path / 'geometry-runtime/offline_plugins.sh').read_text()
    assert 'runtime_geometry/Offline/Mu2eG4/' in captured
    assert 'cp -a "$geometry_root/Offline/Mu2eG4/geom"' in captured
    with pytest.raises(ValueError):
        plugin_runtime.prepare(source, lib, output)


def test_unsupported_runner_or_absent_libraries_refused(tmp_path):
    runner = tmp_path / 'runner.sh'
    runner.write_text('unsupported')
    with pytest.raises(ValueError, match='anchors'):
        plugin_runtime.prepare(runner, tmp_path, tmp_path / 'output')
