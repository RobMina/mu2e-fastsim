"""Review-source and pure-data checks, NOT art execution or ROOT integration."""
import copy
import hashlib
import importlib.util
import json
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('manifest_review', HERE / 'validate_review.py')
v = importlib.util.module_from_spec(spec)
spec.loader.exec_module(v)
INCIDENT = (HERE / 'src/CaloIncidentManifest_module.cc').read_text()
SOURCE = (HERE / 'src/CaloInputEventManifest_module.cc').read_text()
FCL = (HERE / 'fcl/manifest_ce_review.fcl').read_text()


def fixture(counts=(1, 0, 2)):
    inputs, events, products, roots = [], [], [], []
    ids = [[7, 8, i + 1] for i in range(len(counts))]
    for i, count in enumerate(counts):
        base = dict(ordinal=i, run=7, subrun=8, event=i + 1)
        inputs.append(dict(base))
        events.append(dict(base, present=True, n_roots=count, product_id=31, single_root_supported=count == 1))
        products.append(dict(base, present=True, size=count, product_id=31, kind=2,
                             type='mu2e::StageParticleCollection', branch_name='StageParticles_generate__TEST.'))
        for j in range(count):
            r = dict(base, product_id=31, index=j, kind=2, pdg=11, creation_code=42,
                     parent_valid=True, parent_product_id=20, parent_key=1)
            r.update({key: 0.125 for key in ('time', 'position_x', 'position_y', 'position_z',
                                           'momentum_px', 'momentum_py', 'momentum_pz', 'momentum_e')})
            roots.append(r)
    return inputs, events, products, roots, ids


def test_snapshot_and_proposal_pins():
    v.check_snapshot()
    pins = json.loads((HERE / 'source_manifest.json').read_text())
    for rel, digest in pins['proposal_files'].items():
        assert v.sha256(HERE / rel) == digest


def test_snapshot_tampering_rejected(tmp_path):
    pins = {'files': {'fake.hh': hashlib.sha256(b'original').hexdigest()}}
    (tmp_path / 'fake.hh').write_text('tampered')
    manifest = tmp_path / 'manifest.json'; manifest.write_text(json.dumps(pins))
    with pytest.raises(ValueError, match='snapshot mismatch'):
        v.check_snapshot(tmp_path, manifest)


def test_only_explicit_designated_stage_input():
    assert 'consumes<StageParticleCollection>(tag_)' in INCIDENT
    assert 'getByLabel(tag_, h)' in INCIDENT
    for forbidden in ('getMany', 'GenParticleCollection>', 'CaloShower', 'StatusG4', 'getValidHandle<SimParticle', 'getValidHandle<GenParticle'):
        assert forbidden not in INCIDENT
    assert 'consumes<' not in SOURCE and 'getByLabel' not in SOURCE
    assert 'tag_.process().empty()' in INCIDENT


def test_two_phase_schedule_before_filters_and_g4():
    path = FCL.split('physics.PrimaryPath :', 1)[1].split(']', 1)[0]
    names = ('inputEventManifest', 'TargetStopResampler', 'generate', 'incidentManifest', 'genCounter', 'GenFilter', 'Common.g4Sequence')
    assert [path.index(x) for x in names] == sorted(path.index(x) for x in names)
    assert 'physics.trigger_paths : [ PrimaryPath ]' in FCL
    assert 'num_threads : 1' in FCL and 'num_schedules : 1' in FCL
    assert 'generate::CaloLibraryCE' in FCL


def test_event_and_root_rows_precede_rejection():
    analyze = INCIDENT.split('bool CaloIncidentManifest::filter', 1)[1]
    assert analyze.index('events_->Fill()') < analyze.index('roots_->Fill()') < analyze.index('throw cet::exception')
    assert 'count_ == 1' in analyze
    assert 'if (present_) for (auto const& g : *h)' in analyze
    assert 'mode_ == "require_one"' in analyze
    assert 'MANIFEST_MISSING' in analyze
    assert 'continue;' not in analyze


def test_numeric_identity_and_native_double_no_parent_dereference():
    for field in ('"kind"', '"product_id"', '"index"', '"parent_product_id"', '"parent_key"', '"creation_code"'):
        assert field in INCIDENT
    assert 'std::array<double, 3> position_' in INCIDENT
    assert 'std::array<double, 4> momentum_' in INCIDENT
    assert 'double time_' in INCIDENT
    assert 'g.position()' in INCIDENT and 'g.momentum()' in INCIDENT and 'g.time()' in INCIDENT
    assert 'parent->' not in INCIDENT and 'startMomentum' not in INCIDENT
    assert 'productDescription().branchName()' in INCIDENT


def test_both_build_descriptions_exist_but_are_unwired():
    cmake = (HERE / 'CMakeLists.txt').read_text()
    scons = (HERE / 'src/SConscript').read_text()
    assert 'CaloInputEventManifest CaloIncidentManifest' in cmake
    assert 'ROOT::Tree' in cmake and 'art_root_io_TFileService' in scons


def test_manifest_writers_are_nonselecting_trigger_path_filters():
    for module in (SOURCE, INCIDENT):
        assert '#include "art/Framework/Core/EDFilter.h"' in module
        assert 'public art::EDFilter' in module
        assert 'art::EDFilter(p)' in module
        assert 'return true;' in module
        assert 'return false;' not in module
        assert 'EDAnalyzer' not in module
    assert 'bool filter(art::Event& e) override' in SOURCE
    assert 'bool CaloIncidentManifest::filter(art::Event& e)' in INCIDENT
    assert 'physics.filters.inputEventManifest' in FCL
    assert 'physics.filters.incidentManifest' in FCL
    assert 'physics.analyzers.' not in FCL


def test_empty_multiple_and_exact_order_are_preserved():
    v.validate_rows(*fixture())
    assert [e['n_roots'] for e in fixture()[1]] == [1, 0, 2]


def test_strict_mode_rejects_empty_and_multiple():
    for n in (0, 2):
        with pytest.raises(ValueError, match='strict single-root'):
            v.validate_rows(*fixture((n,)), mode='require_one')
    v.validate_rows(*fixture((1,)), mode='require_one')


@pytest.mark.parametrize('corruption', ['drop_empty', 'root_order', 'event_order', 'missing', 'extra_root', 'wrong_kind', 'wrong_product', 'bad_flag', 'nonfinite'])
def test_negative_audits(corruption):
    inp, events, products, roots, ids = copy.deepcopy(fixture())
    if corruption == 'drop_empty': events.pop(1)
    elif corruption == 'root_order': roots[1], roots[2] = roots[2], roots[1]
    elif corruption == 'event_order': inp.reverse()
    elif corruption == 'missing': events[1]['present'] = False
    elif corruption == 'extra_root': roots.append(dict(roots[-1]))
    elif corruption == 'wrong_kind': roots[0]['kind'] = 1
    elif corruption == 'wrong_product': roots[0]['product_id'] = 99
    elif corruption == 'bad_flag': events[1]['single_root_supported'] = True
    elif corruption == 'nonfinite': roots[0]['time'] = float('nan')
    with pytest.raises(ValueError):
        v.validate_rows(inp, events, products, roots, ids)


def test_external_hashes_not_self_attested(tmp_path):
    art = tmp_path / 'input.art'; art.write_bytes(b'input bytes')
    snapshot = tmp_path / 'source_manifest.json'; snapshot.write_text('{}')
    fcl = tmp_path / 'expanded.fcl'
    fcl.write_text('a: {expandedFclSha256: "placeholder"}\nb: {expandedFclSha256: "placeholder"}\n')
    digest = v.normalized_fcl_sha256(fcl)
    fcl.write_text(fcl.read_text().replace('placeholder', digest))
    assert v.normalized_fcl_sha256(fcl) == digest
    claim = {'source_sha256': v.sha256(art), 'source_snapshot_sha256': v.sha256(snapshot), 'expanded_fcl_sha256': digest}
    v.verify_claims(claim, art, snapshot, fcl)
    art.write_bytes(b'changed input bytes')
    with pytest.raises(ValueError, match='external hash mismatch'):
        v.verify_claims(claim, art, snapshot, fcl)


def test_explicit_frame_handle_and_direct_dependencies():
    assert '#include "art/Framework/Principal/Handle.h"' in INCIDENT
    assert 'coordinateFrame_{"Mu2e"}' in INCIDENT
    assert 'm->Branch("coordinate_frame"' in INCIDENT
    cmake = (HERE / 'CMakeLists.txt').read_text()
    scons = (HERE / 'src/SConscript').read_text()
    for library in ('art::Utilities', 'art_root_io::tfile_support', 'cetlib::cetlib'):
        assert library in cmake
    for library in ("'art_Utilities'", "'cetlib'", "'cetlib_except'"):
        assert library in scons


@pytest.mark.parametrize('text', ['[[1.0,2,3]]', '[[true,2,3]]', '[["1",2,3]]',
                                  '[[1,2]]', '[[1,2,3,4]]', '[[1,-2,3]]',
                                  '[[4294967296,2,3]]', '[[1e0,2,3]]'])
def test_expected_ids_reject_noninteger_or_out_of_bounds(text):
    with pytest.raises(ValueError):
        v.expected_event_ids(text)


def test_expected_ids_bounds_and_empty():
    assert v.expected_event_ids('[[0,4294967295,1]]').shape == (1, 3)
    assert v.expected_event_ids('[]').shape == (0, 3)
    v.validate_rows(*fixture(()))


def test_fcl_hash_skips_comments_and_strings(tmp_path):
    path = tmp_path / 'expanded.fcl'
    decoys = ('# expandedFclSha256: "comment"\n'
              '/* expandedFclSha256: "block" */\n'
              '// expandedFclSha256: "line"\n'
              'note: "expandedFclSha256: \\"inside\\""\n')
    actual = 'a: { expandedFclSha256: "one" }\nb: { expandedFclSha256: "two" }\n'
    path.write_text(decoys + actual)
    expected = decoys + actual.replace('"one"', '"<NORMALIZED_FCL_SHA256>"').replace('"two"', '"<NORMALIZED_FCL_SHA256>"')
    assert v.normalized_fcl_sha256(path) == hashlib.sha256(expected.encode()).hexdigest()
    path.write_text(decoys)
    with pytest.raises(ValueError, match='exactly two'):
        v.normalized_fcl_sha256(path)


@pytest.mark.parametrize('text', ['expandedFclSha256: nope',
                                  'expandedFclSha256: "unterminated',
                                  '/* expandedFclSha256: "unterminated"',
                                  'notexpandedFclSha256: "a"\nnotexpandedFclSha256: "b"'])
def test_fcl_hash_rejects_invalid_or_lookalike_tokens(tmp_path, text):
    path = tmp_path / 'expanded.fcl'; path.write_text(text)
    with pytest.raises(ValueError):
        v.normalized_fcl_sha256(path)


def column_fixture(counts=(1, 0, 2)):
    import numpy as np
    inp, events, products, roots, ids = fixture(counts)
    def convert(rows, template):
        return {key: np.asarray([row[key] for row in rows], dtype=type(value))
                for key, value in template.items()}
    # Templates make zero-row columns retain their native scalar dtype.
    samples = fixture((1,))[:4]
    return tuple(convert(rows, sample[0]) for rows, sample in
                 zip((inp, events, products, roots), samples)) + (v.expected_event_ids(json.dumps(ids)),)


def test_vectorized_columns_empty_and_many():
    for counts in ((), (0,), (2, 0, 3), (1,) * 10000):
        v.validate_columns(*column_fixture(counts))


def test_float32_native_state_rejected_without_widening():
    import numpy as np
    inp, events, products, roots, ids = column_fixture()
    roots['time'] = roots['time'].astype(np.float32)
    with pytest.raises(ValueError, match='must be double'):
        v.validate_columns(inp, events, products, roots, ids)


def test_root_columns_read_once_no_row_materialization():
    import numpy as np
    class Tree:
        num_entries = 2
        calls = 0
        def arrays(self, library):
            assert library == 'np'
            self.calls += 1
            return {'ordinal': np.arange(2)}
    tree = Tree()
    assert np.array_equal(v.read_columns(tree)['ordinal'], [0, 1])
    assert tree.calls == 1
    source = (HERE / 'validate_review.py').read_text()
    hot_path = source.split('def audit(', 1)[1]
    assert 'validate_rows(' not in hot_path and 'read_rows' not in source



def test_fcl_normalization_preserves_crlf_and_escaped_strings(tmp_path):
    path = tmp_path / 'expanded.fcl'
    text = 'a: { expandedFclSha256: "old" }\r\nb: { expandedFclSha256: "old" }\r\n'
    path.write_bytes(text.encode())
    expected = text.replace('"old"', '"<NORMALIZED_FCL_SHA256>"')
    assert v.normalized_fcl_sha256(path) == hashlib.sha256(expected.encode()).hexdigest()


def test_column_audit_has_only_fixed_schema_loops():
    import ast
    source = ast.parse((HERE / 'validate_review.py').read_text())
    audit = next(n for n in source.body if isinstance(n, ast.FunctionDef) and n.name == 'validate_columns')
    loops = [n for n in ast.walk(audit) if isinstance(n, (ast.For, ast.While))]
    assert len(loops) == 4  # three tables, two fixed EventID schemas, eight state fields
    assert not any(isinstance(n, (ast.ListComp, ast.DictComp, ast.GeneratorExp)) for n in ast.walk(audit))
    assert 'to_list' not in ast.unparse(audit)
