"""Source/patch assertions only: deliberately not a C++ or native geometry test."""
import importlib.util
import json
import re
import shutil
import subprocess
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('inventory_make_patch', HERE / 'make_patch.py')
gen = importlib.util.module_from_spec(spec)
spec.loader.exec_module(gen)
HEADER = (HERE / 'payload' / 'Mu2eG4PlacementInventory.hh').read_text()
PATCH = (HERE / 'placement-inventory.patch').read_text()
MANIFEST = json.loads((HERE / 'source-manifest.json').read_text())


def require(text):
    guards = [
        'if (debug.boundaryRecorderPrefix().empty()) return;',
        'if (!maxBytes_ || maxRows_ < 3)',
        'if (name.empty())',
        'fail("duplicate configured root")',
        'if (roots_.empty())',
        'if (G4Threading::IsMultithreadedApplication())',
        'if (depth > depthLimit)',
        'pv->IsReplicated() || pv->IsParameterised() || !dynamic_cast<const G4PVPlacement*>(pv)',
        'if (!active_.insert(pv).second)',
        'if (placements_.size() >= placementLimit || placements_.size() >= maxRows_ - 2)',
        'if (!siblings.emplace(std::pmr::string(name.data(), name.size(), &memory_), child->GetCopyNo()).second)',
        'if (++root.count != 1)',
        'if (root.count != 1)',
        'if (matches != 1)',
        'if (totals.rows >= maxRows_ || row.size() > maxBytes_ - totals.bytes)',
        'if (n > limit_ - used_)',
        'if (rows >= maxRows_)',
        'if (draft.size() > maxBytes_ - totals.bytes)',
        'if (end.size() != draft.size())',
        'if (expected && (totals.bytes + end.size() != expected->bytes || totals.rows + 1 != expected->rows))',
        'Count(n) > limit_ - data.size()',
        'O_WRONLY | O_CREAT | O_EXCL',
        'if (::fsync(fd) != 0)',
        'if (::close(fd) != 0)',
        'if (n <= 0)',
        'if (!std::isfinite(x))',
        'if (!solids_[logicals_[logical].solid].supported) ++unsupported_;',
    ]
    for guard in guards:
        assert guard in text, guard
    assert text.index('inventory.capture();') < text.index('inventory.serialize(g4Run, -1)')
    assert text.index('inventory.serialize(g4Run, -1)') < text.index('inventory.writeExclusive(g4Run, expected)')
    assert 'walk(world, none, 0, false)' in text
    assert not re.search(r'\b(?:rows_|paths_)\b', text)
    assert 'std::pmr::vector<Placement>' in text
    assert 'std::pmr::map<const G4VSolid*, Count>' in text
    assert 'std::pmr::new_delete_resource()->allocate(n, alignment)' in text
    assert 'static constexpr Count memoryLimit = 268435456;' in text
    assert 'std::min(maxBytes_, memoryLimit)' in text
    assert 'mu2e-placement-inventory-v2' in text
    assert 'navigator->GetWorldVolume()' in text
    assert 'lv->GetDaughter(i)' in text
    assert '#include <cstddef>' in text
    assert 'for (std::size_t i = 0; i < daughters; ++i)' in text
    assert 'for (int i = 0; i < daughters; ++i)' not in text
    assert 'pv->GetObjectTranslation()' in text and 'pv->GetObjectRotationValue()' in text
    assert 'root.placement = id;' in text
    assert 'selected = true;' in text
    assert 'id, depth + 1, selected)' in text[text.index('lv->GetDaughter(i)'):]
    for forbidden in ['LocateGlobalPoint', 'LocateGlobalPointAndSetup', 'SetTranslation(',
                      'SetRotation(', 'SetWorldVolume(', 'ComputeTransformation(',
                      'G4UniformRand', 'G4Random', 'SetTrackStatus(', 'CalculateExtent(',
                      '#include "TFile', '#include "TTree', 'ComputeDimensions(']:
        assert forbidden not in text
    assert 'unsupported_metadata_only' in text
    assert 'native_parameters' in text
    for key in ['parent_id', 'candidate_root_id', 'unique_paths', 'root_placements',
                'logical_volume_id', 'material_id', 'solid_id', 'helper_allocator_max_bytes',
                'rows_including_footer', 'bytes_including_footer_decimal',
                'candidate_placements', 'unsupported_solid_placements']:
        assert key in text
    assert 'GetOriginalParameters()' in text and 'planeLimit' in text
    assert 'Text native(std::min(maxBytes_, textLimit), "native_stream_info solid="' in text
    for marker in ('buffer_used=', 'buffer_limit=', 'attempted_write=', 'file_cap='):
        assert marker in text
    for stage in ('material', 'solid', 'logical_volume', 'placement', 'begin', 'footer'):
        assert 'context("' + stage + '"' in text
    assert 'substr(0, 80)' in text
    assert text.count('StreamInfo(') == 1
    return guards


def test_source_invariants():
    require(HEADER)


def test_guard_deletion_mutations_are_detected():
    for guard in require(HEADER):
        with pytest.raises(AssertionError):
            require(HEADER.replace(guard, '/* deliberately removed guard */', 1))


def test_manifest_patch_footprint_and_hook_order():
    assert gen.sha(PATCH.encode()) == MANIFEST['patch_sha256']
    assert gen.sha(HEADER.encode()) == MANIFEST['patched'][gen.HEADER]
    assert MANIFEST['no_config_patch'] is True
    assert set(MANIFEST['patched']) == {gen.RUN, gen.HEADER}
    assert set(re.findall(r'^\+\+\+ b/(.+)$', PATCH, re.M)) == {gen.RUN, gen.HEADER}
    assert 'Mu2eG4Config.hh\n' not in PATCH
    assert gen.ANCHOR + gen.HOOK in PATCH.replace('\n+', '\n')
    assert 'if (!debug_.boundaryRecorderPrefix().empty())' in gen.HOOK
    assert 'Mu2eG4PlacementInventory::write(debug_, aRun->GetRunID());' in gen.HOOK
    assert '.inventory.run' in HEADER
    assert 'boundaryRecorderPrefix' in MANIFEST['requires_external_boundary_debug_contract']


def verify_base(root):
    for name, expected in MANIFEST['sources'].items():
        assert gen.sha((root / name).read_bytes()) == expected, 'source drift: ' + name
    assert not (root / gen.HEADER).exists(), 'header collision'


def run_patch(root, reverse=False, dry=False):
    args = ['patch', '--batch', '--fuzz=0', '-p1', '-R' if reverse else '--forward']
    if dry:
        args.append('--dry-run')
    proc = subprocess.run(args, cwd=root, input=PATCH, text=True, capture_output=True)
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert not re.search(r'offset|fuzz', proc.stdout + proc.stderr, re.I)


@pytest.fixture
def source(request):
    value = request.config.getoption('--offline')
    if value is None:
        pytest.skip('explicit --offline required for read-only source applicability test')
    return Path(value).resolve()


def test_exact_source_patch_apply_reverse_and_drift(source, tmp_path):
    verify_base(source)
    assert gen.build(source) == (PATCH, MANIFEST)
    assert shutil.which('patch')
    for name in MANIFEST['sources']:
        dest = tmp_path / name
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source / name, dest)
    verify_base(tmp_path)
    run_patch(tmp_path, dry=True)
    run_patch(tmp_path)
    for name, expected in MANIFEST['patched'].items():
        assert gen.sha((tmp_path / name).read_bytes()) == expected
    patched = (tmp_path / gen.RUN).read_text()
    assert gen.ANCHOR + gen.HOOK in patched
    assert patched.index(gen.HOOK) < patched.index('navigator->CheckMode(')
    assert gen.INCLUDE + '\n#include "Offline/' + gen.HEADER + '"' in patched
    run_patch(tmp_path, reverse=True)
    # patch may preserve a now-empty added file; unlink only in disposable directory.
    if (tmp_path / gen.HEADER).exists():
        assert (tmp_path / gen.HEADER).stat().st_size == 0
        (tmp_path / gen.HEADER).unlink()
    verify_base(tmp_path)
    (tmp_path / gen.RUN).write_bytes((tmp_path / gen.RUN).read_bytes() + b'\n// drift\n')
    with pytest.raises(AssertionError, match='source drift'):
        verify_base(tmp_path)
    verify_base(source)


def test_existing_boundary_adapter_contract_and_disjoint_patch(source, tmp_path):
    # Read sibling review generator; all generated/applied files remain disposable.
    boundary = HERE.parent / 'boundary'
    spec = importlib.util.spec_from_file_location('boundary_make_patch', boundary / 'make_patch.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    patch, _ = module.build(source)
    changed = set(re.findall(r'^\+\+\+ b/(.+)$', patch, re.M))
    assert not changed.intersection(MANIFEST['patched'])
    cfg = (source / 'Mu2eG4/inc/Mu2eG4Config.hh').read_text()
    cfg += '\n' + patch
    for declaration in gen.CONTRACT.values():
        assert declaration in cfg
    # Both review patches apply together without offsets/config overlap.
    names = changed | set(MANIFEST['sources'])
    for name in names:
        src = source / name
        if src.exists():
            dest = tmp_path / name
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(src, dest)
    result = subprocess.run(['patch', '--batch', '--fuzz=0', '-p1', '--forward'],
                            cwd=tmp_path, input=patch, text=True, capture_output=True)
    assert result.returncode == 0, result.stdout + result.stderr
    run_patch(tmp_path, dry=True)
    run_patch(tmp_path)
    for declaration in gen.CONTRACT.values():
        assert (tmp_path / 'Mu2eG4/inc/Mu2eG4Config.hh').read_text().count(declaration) == 1
