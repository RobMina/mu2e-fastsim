"""Synthetic, bounded documentary fixtures; no retained audit or physics inputs."""
import gzip
import hashlib
import io
import json
from pathlib import Path
import tarfile

import pytest

from crvfastsim.veto import cat_jobdef_audit as A


def names(count):
    return [f'dts.mu2e.NeutralsFlash.MDC2025ac.001430_{i:08d}.art' for i in range(count)]


def spec(count=8, block_size=2):
    return dict(template={'code': '',
        'setup': '/cvmfs/mu2e.opensciencegrid.org/Musings/SimJob/MDC2025ad/setup.sh',
        'tbs': {'subrunkey': '', 'event_id': {'source.maxEvents': 2147483647},
                'outfiles': {'outputs.out.fileName': 'dts.mu2e.NeutralsFlashCat.MDC2025ad.sequence.art'},
                'inputs': {'source.fileNames': [block_size, None]}, 'sequencer_from_index': True},
        'jobname': 'cnf.mu2e.NeutralsFlashCat.MDC2025ad.0.tar'},
        filenames_path=['tbs', 'inputs', 'source.fileNames', 1],
        group_size_path=['tbs', 'inputs', 'source.fileNames', 0],
        sequencer_path=['tbs', 'sequencer_from_index'],
        max_events_path=['tbs', 'event_id', 'source.maxEvents'], expected_items=count)


def document(contract=None, files=None):
    contract = contract or spec()
    result = json.loads(json.dumps(contract['template']))
    result['tbs']['inputs']['source.fileNames'][1] = names(contract['expected_items']) if files is None else files
    return json.dumps(result).encode()


def archive(tmp_path, payload, extra=(), kind=None, name='jobpars.json'):
    path = tmp_path / 'synthetic.tar'
    with tarfile.open(path, 'w:gz', format=tarfile.USTAR_FORMAT) as tar:
        for member_name, data, member_kind in [(name, payload, kind), *extra]:
            info = tarfile.TarInfo(member_name)
            info.size = len(data)
            if member_kind:
                info.type = member_kind
                info.linkname = 'outside'
            tar.addfile(info, io.BytesIO(data))
    return path


def scan(path, payload, **kw):
    return A.scan_archive(path, spec(), archive_size=path.stat().st_size, jobpars_size=len(payload),
                          indices=(1, 3), block_size=2, **kw)


def test_stream_and_hash_convention(tmp_path):
    payload = document(files=list(reversed(names(8))))
    path = archive(tmp_path, payload, [('mu2e.fcl', b'hello', None)])
    result = scan(path, payload)
    assert result['archive_read_sha256'] == hashlib.sha256(path.read_bytes()).hexdigest()
    assert result['jobpars']['sha256'] == hashlib.sha256(payload).hexdigest()
    assert result['jobpars']['filename_count'] == 8
    block = result['jobpars']['blocks'][1]
    expected = hashlib.sha256(('\n'.join(sorted(names(8)[4:6])) + '\n').encode()).hexdigest()
    assert block == dict(start=2, stop=4, count=2, unique_count=2, parents_sha256=expected)
    assert len(result['members']) == 2


def test_matches_mismatches_and_duplicates_reported():
    first = A.parent_hash(names(2))
    blocks = {i: dict(start=i * 2, stop=(i + 1) * 2, count=2, unique_count=2, parents_sha256=first)
              for i in (1, 3)}
    saved = {i: dict(parent_count=2, parents_sha256=first) for i in blocks}
    assert all(row['matches'] for row in A.compare(blocks, saved))
    blocks[3]['parents_sha256'] = '0' * 64
    rows = A.compare(blocks, saved)
    assert [r['index'] for r in rows if not r['matches']] == [3]
    blocks[3]['parents_sha256'] = first
    blocks[3]['unique_count'] = 1
    assert A.compare(blocks, saved)[1]['matches'] is False


def test_only_target_block_retained_and_duplicate_not_tuned():
    payload = document(files=[names(1)[0]] * 8)
    result = A.scan_jobpars(io.BytesIO(payload), len(payload), spec(), indices=(1, 3), block_size=2)
    assert set(result['blocks']) == {1, 3}
    assert all(row['unique_count'] == 1 for row in result['blocks'].values())
    assert not any('filenames' in row for row in result['blocks'].values())


@pytest.mark.parametrize('mutate', [
    lambda p: p + b'{}',
    lambda p: p[:-1],
    lambda p: p.replace(b'"setup":', b'"setup": "x", "setup":'),
    lambda p: p.replace(b'"source.fileNames": [2,', b'"source.fileNames": [true,'),
    lambda p: p.replace(b'"source.fileNames": [2,', b'"source.fileNames": [2.0,'),
    lambda p: p.replace(b'"source.fileNames": [2,', b'"source.fileNames": [3,'),
    lambda p: p.replace(b'"source.fileNames": [2,', b'"source.fileNames": [02,'),
    lambda p: p.replace(b'"source.fileNames": [2,', b'"source.fileNames": [NaN,'),
    lambda p: p.replace(b'"source.fileNames": [2,', b'"extra": 1, "source.fileNames": [2,'),
    lambda p: p.replace(b'"code": "", ', b''),
    lambda p: p.replace(b'"sequencer_from_index": true', b'"sequencer_from_index": 1'),
    lambda p: p.replace(b'2147483647', b'2147483646'),
    lambda p: p.replace(b'"source.fileNames": [2, [', b'"source.fileNames": [2, [null,'),
    lambda p: p.replace(b'"source.fileNames": [2, [', b'"source.fileNames": [2, [['),
    lambda p: p.replace(b'/cvmfs/mu2e.opensciencegrid.org/Musings/SimJob/MDC2025ad/setup.sh', b'/cvmfs/\x01setup.sh'),
    lambda p: p.replace(b'/cvmfs/mu2e.opensciencegrid.org/Musings/SimJob/MDC2025ad/setup.sh', b'\\ud800'),
    lambda p: p.replace(b'/cvmfs/mu2e.opensciencegrid.org/Musings/SimJob/MDC2025ad/setup.sh', b'\\q'),
    lambda p: p.replace(b'/cvmfs/mu2e.opensciencegrid.org/Musings/SimJob/MDC2025ad/setup.sh', b'\xff'),
    lambda p: p.replace(b'/cvmfs/mu2e.opensciencegrid.org/Musings/SimJob/MDC2025ad/setup.sh', b'a' * 4097),
])
def test_malformed_and_schema_fail_closed(mutate):
    payload = mutate(document())
    with pytest.raises((ValueError, UnicodeError)):
        A.scan_jobpars(io.BytesIO(payload), len(payload), spec(), indices=(1, 3), block_size=2)


@pytest.mark.parametrize('files', [names(7), names(9), [], ['/pnfs/' + names(1)[0]] * 8, [False] * 8])
def test_filename_count_shape_and_path_fail(files):
    payload = document(files=files)
    with pytest.raises(ValueError):
        A.scan_jobpars(io.BytesIO(payload), len(payload), spec(), indices=(1, 3), block_size=2)


@pytest.mark.parametrize('payload', [b'{"a":1,"a":2}', b'{"a":1,}', b'[1,]', b'{} []', b'NaN',
                                     b'1e400', b'-01', b'[' * 14 + b'0' + b']' * 14,
                                     b'[' + b','.join([b'0'] * 129) + b']'])
def test_small_json_caps_and_structure(payload):
    with pytest.raises(ValueError):
        A.small_json(payload)


def test_byte_boundaries_and_escaped_strings():
    class Tiny:
        def __init__(self, data):
            self.io = io.BytesIO(data)

        def read(self, count):
            assert count <= A.CHUNK
            return self.io.read(min(count, 1))

    data = b'{"abc":"\\u0061\\\\b\\\"c", "x":[true,false,null,-1]}'
    parser = A.JSONStream(Tiny(data))
    assert parser.value() == json.loads(data)
    parser.finish()
    payload = document()
    A.scan_jobpars(Tiny(payload), len(payload), spec(), indices=(1, 3), block_size=2)


@pytest.mark.parametrize('kind,name', [(tarfile.SYMTYPE, 'jobpars.json'), (tarfile.LNKTYPE, 'jobpars.json'),
    (tarfile.DIRTYPE, 'directory'), (tarfile.FIFOTYPE, 'fifo'), (tarfile.XHDTYPE, 'pax'),
    (None, '../jobpars.json'), (None, '/jobpars.json'), (None, 'file.root'), (None, 'file.art')])
def test_unsafe_tar_members(tmp_path, kind, name):
    payload = document()
    path = archive(tmp_path, payload, kind=kind, name=name)
    with pytest.raises(ValueError):
        scan(path, payload)


def test_duplicate_tar_and_member_count_cap(tmp_path):
    payload = document()
    path = archive(tmp_path, payload, [('jobpars.json', payload, None)])
    with pytest.raises(ValueError, match='duplicate'):
        scan(path, payload)
    path = archive(tmp_path, payload, [(f'file_{i}.txt', b'', None) for i in range(A.MAX_MEMBERS)])
    with pytest.raises(ValueError, match='member cap'):
        scan(path, payload)


def test_tar_small_member_size_missing_and_total_cap(tmp_path, monkeypatch):
    payload = document()
    path = archive(tmp_path, payload, [('huge.txt', b'x' * (A.SMALL_MEMBER_CAP + 1), None)])
    with pytest.raises(ValueError, match='member byte cap'):
        scan(path, payload)
    path = archive(tmp_path, payload, name='other.json')
    with pytest.raises(ValueError, match='missing'):
        scan(path, payload)
    path = archive(tmp_path, payload)
    monkeypatch.setattr(A, 'TOTAL_TAR_CAP', 600)
    with pytest.raises(ValueError, match='stream byte cap'):
        scan(path, payload)


def test_trailing_truncated_and_bad_tar(tmp_path):
    payload = document()
    path = archive(tmp_path, payload)
    raw = gzip.decompress(path.read_bytes())
    for altered in (raw + b'not zero', raw[:512], b'not a header' + raw[12:]):
        path.write_bytes(gzip.compress(altered))
        with pytest.raises((ValueError, tarfile.HeaderError)):
            scan(path, payload)
    path.write_bytes(gzip.compress(raw)[:-5])
    with pytest.raises((ValueError, EOFError)):
        scan(path, payload)


def test_byte_caps_exact_sizes_and_read_requests(tmp_path):
    reader = A.BoundedReader(io.BytesIO(b'abc'), 2)
    with pytest.raises(ValueError, match='byte cap'):
        reader.read(3)
    with pytest.raises(ValueError, match='unbounded'):
        reader.read(-1)
    payload = document()
    with pytest.raises(ValueError, match='size'):
        A.scan_jobpars(io.BytesIO(payload), len(payload) + 1, spec(), indices=(1, 3), block_size=2)
    path = archive(tmp_path, payload)
    with pytest.raises(ValueError):
        A.scan_archive(path, spec(), archive_size=1, jobpars_size=len(payload), indices=(1, 3), block_size=2)


def test_contract_control_types_and_caps():
    for key, value in [('expected_items', True), ('expected_items', A.MAX_ITEMS + 1),
                       ('expected_items', 4), ('filenames_path', ['missing']),
                       ('sequencer_path', ['setup']), ('unexpected', 1)]:
        contract = spec()
        contract[key] = value
        with pytest.raises(ValueError):
            A.validate_contract(contract, indices=(1, 3), block_size=2)


def test_exact_pin_checks_output_exclusion_and_symlinks(tmp_path):
    source = tmp_path / 'source.json'
    source.write_bytes(b'{}')
    assert A.small_bytes(source, 2, A.sha(b'{}')) == b'{}'
    with pytest.raises(ValueError, match='SHA256'):
        A.small_bytes(source, 2, '0' * 64)
    with pytest.raises(ValueError, match='cap'):
        A.small_bytes(source, 1, A.sha(b'{}'))
    alias = tmp_path / 'alias.json'
    alias.symlink_to(source)
    with pytest.raises(ValueError, match='symlink'):
        A.small_bytes(alias, 10, A.sha(b'{}'))
    with pytest.raises(ValueError, match='outside'):
        A.outside(tmp_path / 'report.json', [tmp_path])
    with pytest.raises(ValueError, match='overwrite'):
        A.outside(source, [])
    with pytest.raises(ValueError, match='physics'):
        A.checked_path(tmp_path / 'selected.parquet')


def test_saved_metadata_exact_twelve_pins(tmp_path):
    parents = names(A.BLOCK_SIZE)
    sources = {}
    for index in A.INDICES:
        name = A.CAT.format(index)
        text = (f'File Name: {name}\nFile Id: {index}\nEvent Count: 12\n'
                f'dh.sequencer: 001430_{index:08d}\ndh.sha256: {"a" * 64}\nParents:\n' +
                ''.join(f'  {parent}\n' for parent in reversed(parents)))
        path = tmp_path / f'sam_{index}.txt'
        path.write_text(text)
        sources[name] = {'sam': dict(file_name=name, file_id=str(index), event_count=12,
            source_sha256='a' * 64, parent_count=len(parents), parents_sha256=A.parent_hash(parents),
            metadata_sha256=A.sha(path.read_bytes()))}
    provenance = dict(sources=sources, parent_overlaps='inherited and never evaluated')
    rows, bindings = A.saved_metadata(tmp_path, provenance)
    assert set(rows) == set(A.INDICES) and len(bindings) == 12
    first = sources[A.CAT.format(A.INDICES[0])]['sam']
    for field, value in [('file_name', 'wrong'), ('parent_count', True), ('parents_sha256', '0' * 64),
                         ('metadata_sha256', '0' * 64), ('event_count', True), ('file_id', 'wrong')]:
        original = first[field]
        first[field] = value
        with pytest.raises(ValueError):
            A.saved_metadata(tmp_path, provenance)
        first[field] = original


def test_production_run_fails_before_any_archive_access_on_wrong_pins(tmp_path, monkeypatch):
    monkeypatch.setattr(A, 'open_regular', lambda *a, **kw: pytest.fail('must reject pins before IO'))
    with pytest.raises(ValueError, match='campaign archive pin'):
        A.run(tmp_path, tmp_path, tmp_path / 'out', expected_contract_sha256='0' * 64,
              expected_archive_sha256='0' * 64, expected_provenance_sha256=A.PROVENANCE_SHA256)


def test_standard_library_only_source_no_old_audit_imports():
    import ast
    module = ast.parse(Path(A.__file__).read_text())
    imported = {n.module.split('.')[0] for n in ast.walk(module) if isinstance(n, ast.ImportFrom)}
    imported |= {alias.name.split('.')[0] for n in ast.walk(module) if isinstance(n, ast.Import) for alias in n.names}
    assert imported <= {'__future__', 'argparse', 'gzip', 'hashlib', 'io', 'json', 'os', 'pathlib', 're', 'stat', 'tarfile'}


@pytest.fixture
def production_shaped(tmp_path, monkeypatch):
    root = tmp_path / 'evidence' / 'continuation_v1'
    root.mkdir(parents=True)
    config = tmp_path / 'config'
    config.mkdir()
    outputs = tmp_path / 'outputs'
    outputs.mkdir()
    payload = document()
    tar = archive(config, payload)
    contract = config / 'contract.json'
    contract.write_text(json.dumps(spec()))
    sources = {}
    for index in (1, 3):
        parents = names(8)[index * 2:(index + 1) * 2]
        name = A.CAT.format(index)
        text = (f'File Name: {name}\nFile Id: {index}\nEvent Count: 12\n'
                f'dh.sequencer: 001430_{index:08d}\ndh.sha256: {"a" * 64}\nParents:\n' +
                ''.join(f'  {parent}\n' for parent in parents))
        path = root / f'sam_{index}.txt'
        path.write_text(text)
        sources[name] = {'sam': dict(file_name=name, file_id=str(index), event_count=12,
            source_sha256='a' * 64, parent_count=2, parents_sha256=A.parent_hash(parents),
            metadata_sha256=A.sha(path.read_bytes()))}
    pp = root / 'provenance_final.json'
    pp.write_text(json.dumps(dict(sources=sources)))
    monkeypatch.setattr(A, 'ARCHIVE', tar)
    monkeypatch.setattr(A, 'ARCHIVE_BYTES', tar.stat().st_size)
    monkeypatch.setattr(A, 'ARCHIVE_SHA256', A.sha(tar.read_bytes()))
    monkeypatch.setattr(A, 'PROVENANCE_SHA256', A.sha(pp.read_bytes()))
    monkeypatch.setattr(A, 'INDICES', (1, 3))
    monkeypatch.setattr(A, 'BLOCK_SIZE', 2)
    original_scan = A.scan_archive
    monkeypatch.setattr(A, 'scan_archive', lambda path, contract: original_scan(path, contract,
        archive_size=A.ARCHIVE_BYTES, jobpars_size=len(payload), indices=(1, 3), block_size=2))
    return dict(root=root, contract=contract, out=outputs / 'report.json',
                expected_contract_sha256=A.sha(contract.read_bytes()),
                expected_archive_sha256=A.ARCHIVE_SHA256, expected_provenance_sha256=A.PROVENANCE_SHA256)


def test_exclusive_output_and_source_snapshot(production_shaped, monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail('namespace discovery/extraction forbidden')

    monkeypatch.setattr(Path, 'glob', forbidden)
    monkeypatch.setattr(Path, 'rglob', forbidden)
    monkeypatch.setattr(tarfile.TarFile, 'extractall', forbidden)
    result = A.run(**production_shaped)
    assert result['all_blocks_match'] is True
    assert result['source_sha256'] == A.sha(result['source_snapshot'].encode())
    assert result['archive_before_sha256'] == result['archive_after_sha256'] == result['archive_read_sha256']
    assert result['physics_executed'] is False
    assert result['parent_overlap_recomputed'] is False
    assert production_shaped['out'].stat().st_size < A.OUTPUT_CAP
    saved_bytes = production_shaped['out'].read_bytes()
    with pytest.raises(ValueError, match='overwrite'):
        A.run(**production_shaped)
    assert production_shaped['out'].read_bytes() == saved_bytes


def test_valid_mismatch_publishes_report(production_shaped, monkeypatch):
    original_scan = A.scan_archive

    def mismatching(*args, **kwargs):
        result = original_scan(*args, **kwargs)
        result['jobpars']['blocks'][3]['parents_sha256'] = 'f' * 64
        return result

    monkeypatch.setattr(A, 'scan_archive', mismatching)
    result = A.run(**production_shaped)
    assert result['all_blocks_match'] is False and result['mismatch_indices'] == [3]
    assert production_shaped['out'].exists()


def test_archive_rechecked_even_when_scan_rejected(production_shaped, monkeypatch):
    original_hash = A.file_hash
    calls = []

    def recording(path, *args, **kwargs):
        if path == A.ARCHIVE:
            calls.append(path)
        return original_hash(path, *args, **kwargs)

    def rejected(*args, **kwargs):
        raise ValueError('scan rejected')

    monkeypatch.setattr(A, 'file_hash', recording)
    monkeypatch.setattr(A, 'scan_archive', rejected)
    with pytest.raises(ValueError, match='scan rejected'):
        A.run(**production_shaped)
    assert len(calls) == 2
    assert not production_shaped['out'].exists()


def test_late_metadata_change_and_output_cap(production_shaped, monkeypatch):
    original_scan = A.scan_archive

    def mutate(*args, **kwargs):
        result = original_scan(*args, **kwargs)
        (production_shaped['root'] / 'sam_1.txt').write_text('changed')
        return result

    monkeypatch.setattr(A, 'scan_archive', mutate)
    with pytest.raises(ValueError, match='changed after read'):
        A.run(**production_shaped)
    assert not production_shaped['out'].exists()


def test_output_cap_before_publication(production_shaped, monkeypatch):
    monkeypatch.setattr(A, 'OUTPUT_CAP', 1)
    with pytest.raises(ValueError, match='output byte cap'):
        A.run(**production_shaped)
    assert not production_shaped['out'].exists()


def test_real_chunk_boundaries_and_single_block_memory():
    contract = spec(count=4000, block_size=2)
    payload = document(contract)
    assert len(payload) > A.CHUNK * 2
    result = A.scan_jobpars(io.BytesIO(payload), len(payload), contract, indices=(1, 1999), block_size=2)
    assert result['bytes'] == len(payload)
    assert set(result['blocks']) == {1, 1999}
    assert result['blocks'][1999]['parents_sha256'] == A.parent_hash(names(4000)[3998:])


def test_excess_keys_and_regular_member_cap(tmp_path, monkeypatch):
    payload = json.dumps({str(i): 0 for i in range(129)}).encode()
    with pytest.raises(ValueError, match='object keys'):
        A.small_json(payload)
    payload = document()
    path = archive(tmp_path, payload, [('mu2e.fcl', b'hello', None)])
    monkeypatch.setattr(A, 'MAX_REGULAR', 1)
    with pytest.raises(ValueError, match='regular member cap'):
        scan(path, payload)


def test_read_hash_and_after_hash_mismatch_block_publication(production_shaped, monkeypatch):
    original_scan = A.scan_archive

    def wrong_read_hash(*args, **kwargs):
        result = original_scan(*args, **kwargs)
        result['archive_read_sha256'] = 'f' * 64
        return result

    monkeypatch.setattr(A, 'scan_archive', wrong_read_hash)
    with pytest.raises(ValueError, match='compressed bytes differ'):
        A.run(**production_shaped)
    assert not production_shaped['out'].exists()

    def change_archive(*args, **kwargs):
        result = original_scan(*args, **kwargs)
        payload = A.ARCHIVE.read_bytes()
        A.ARCHIVE.write_bytes(payload[:-1] + bytes([payload[-1] ^ 1]))
        return result

    monkeypatch.setattr(A, 'scan_archive', change_archive)
    with pytest.raises(ValueError, match='archive changed after'):
        A.run(**production_shaped)
    assert not production_shaped['out'].exists()


def test_unknown_count_requires_explicit_exact_bytes_and_reports_count():
    contract = spec(count=None)
    with pytest.raises(ValueError, match='requires exact_jobpars_bytes'):
        A.validate_contract(contract, indices=(1, 3), block_size=2)
    for count in (8, 10, 4000):
        payload = document(contract, names(count))
        contract['exact_jobpars_bytes'] = len(payload)
        result = A.scan_jobpars(io.BytesIO(payload), len(payload), contract, indices=(1, 3), block_size=2)
        assert result['filename_count'] == count
        assert set(result['blocks']) == {1, 3}
        assert result['sha256'] == A.sha(payload)
    with pytest.raises(ValueError, match='contract byte pin'):
        A.scan_jobpars(io.BytesIO(payload), len(payload) + 1, contract, indices=(1, 3), block_size=2)


@pytest.mark.parametrize('count', [0, 2, 6, 7, 9, 11])
def test_unknown_count_rejects_incomplete_or_missing_blocks(count):
    contract = spec(count=None)
    payload = document(contract, names(count))
    contract['exact_jobpars_bytes'] = len(payload)
    with pytest.raises(ValueError):
        A.scan_jobpars(io.BytesIO(payload), len(payload), contract, indices=(1, 3), block_size=2)


def test_unknown_count_cannot_exceed_cap(monkeypatch):
    contract = spec(count=None)
    payload = document(contract, names(10))
    contract['exact_jobpars_bytes'] = len(payload)
    monkeypatch.setattr(A, 'MAX_ITEMS', 8)
    with pytest.raises(ValueError, match='filename count cap'):
        A.scan_jobpars(io.BytesIO(payload), len(payload), contract, indices=(1, 3), block_size=2)


@pytest.mark.parametrize('value', [None, True, -1, 0, 1.0, '123', A.JOBPARS_BYTES + 1])
def test_exact_byte_pin_types_and_cap(value):
    contract = spec(count=None)
    contract['exact_jobpars_bytes'] = value
    with pytest.raises(ValueError, match='byte pin'):
        A.validate_contract(contract, indices=(1, 3), block_size=2)


@pytest.mark.parametrize('path', [
    ['tbs', 'inputs', 'source.fileNames', True],
    ['tbs', 'inputs', 'source.fileNames', '1'],
    ['tbs', 'inputs', 'source.fileNames', -1],
    ['tbs', 'inputs', 'source.fileNames', 2],
    ['tbs', 'inputs', 'source.fileNames', 1.0],
])
def test_filename_path_is_exact_integer_index(path):
    contract = spec()
    contract['filenames_path'] = path
    with pytest.raises(ValueError):
        A.validate_contract(contract, indices=(1, 3), block_size=2)


@pytest.mark.parametrize('mutate', [
    lambda d: d.update(code='changed'),
    lambda d: d.update(jobname='other.tar'),
    lambda d: d.update(unexpected=1),
    lambda d: d.pop('setup'),
    lambda d: d['tbs']['inputs']['source.fileNames'].append(1),
    lambda d: d['tbs']['outfiles'].update({'outputs.out.fileName': 'other.art'}),
    lambda d: d['tbs'].update(subrunkey='changed'),
    lambda d: d['tbs']['event_id'].update(unexpected=1),
])
def test_full_root_is_compared_strictly(mutate):
    data = json.loads(document())
    mutate(data)
    payload = json.dumps(data).encode()
    with pytest.raises(ValueError):
        A.scan_jobpars(io.BytesIO(payload), len(payload), spec(), indices=(1, 3), block_size=2)


def test_archive_read_never_stats_or_checks_namespace(tmp_path, monkeypatch):
    payload = document()
    path = archive(tmp_path, payload)
    size, digest = path.stat().st_size, A.sha(path.read_bytes())
    flags, original_open = [], A.os.open

    def opening(path, mode):
        flags.append(mode)
        return original_open(path, mode)

    def forbidden(*args, **kwargs):
        raise AssertionError('archive namespace/stat call forbidden')

    with monkeypatch.context() as patch:
        for name in ('stat', 'lstat', 'fstat', 'readlink', 'listdir', 'scandir'):
            patch.setattr(A.os, name, forbidden)
        patch.setattr(A.os, 'open', opening)
        patch.setattr(A, 'no_symlinks', forbidden)
        assert A.file_hash(path, size, archive=True, exact=size) == digest
        result = A.scan_archive(path, spec(), archive_size=size, jobpars_size=len(payload),
                                indices=(1, 3), block_size=2)
        assert result['archive_read_sha256'] == digest
    assert flags == [A.os.O_RDONLY, A.os.O_RDONLY]


def test_measured_count_production_publication(production_shaped):
    config = production_shaped['contract']
    contract = json.loads(config.read_bytes())
    contract['expected_items'] = None
    contract['exact_jobpars_bytes'] = len(document())
    config.write_text(json.dumps(contract))
    production_shaped['expected_contract_sha256'] = A.sha(config.read_bytes())
    result = A.run(**production_shaped)
    assert result['all_blocks_match']
    assert result['jobpars']['filename_count'] == 8
    assert result['contract']['expected_items'] is None


@pytest.mark.parametrize('key,path,value', [
    ('sequencer_path', ['sequencer_from_index'], True),
    ('max_events_path', ['tbs', 'source.maxEvents'], 2147483647),
])
def test_old_control_locations_rejected_even_if_values_present(key, path, value):
    contract = spec()
    contract[key] = path
    target = contract['template']
    for part in path[:-1]:
        target = target[part]
    target[path[-1]] = value
    with pytest.raises(ValueError, match='explicitly bound'):
        A.validate_contract(contract, indices=(1, 3), block_size=2)


def test_plain_strings_use_bounded_regex_not_per_character_decoder(monkeypatch):
    name = names(1)[0]
    parser = A.JSONStream(io.BytesIO(json.dumps(name).encode()))

    def forbidden(*args, **kwargs):
        pytest.fail('plain filename should not need json.loads')

    monkeypatch.setattr(A.json, 'loads', forbidden)
    assert parser.string() == name
    parser.finish()


def test_actual_pax_timestamp_archive(tmp_path):
    payload = document()
    path = tmp_path / 'pax.tar.gz'
    with tarfile.open(path, 'w:gz', format=tarfile.PAX_FORMAT) as tar:
        info = tarfile.TarInfo('jobpars.json')
        info.size = len(payload)
        info.pax_headers = {'mtime': '1763072136.4645'}
        tar.addfile(info, io.BytesIO(payload))
    result = scan(path, payload)
    assert result['members'][0]['timestamps'] == {'mtime': '1763072136.4645'}
    assert result['jobpars']['filename_count'] == 8


@pytest.mark.parametrize('payload', [b'', b'99 mtime=0\n', b'0 mtime=0\n',
    b'13 path=a.art\n', b'11 size=10\n', b'13 mtime=NaN\n',
    b'11 mtime=0\n11 mtime=0\n', b'11 mtime=0!', b'99999 mtime=0\n'])
def test_pax_rejects_semantic_or_malformed_records(payload):
    with pytest.raises(ValueError):
        A.pax_times(payload)


@pytest.mark.parametrize('mode', ['orphan', 'consecutive', 'padding', 'global', 'huge'])
def test_pax_structure_fail_closed(tmp_path, mode):
    payload = document()
    body = b'25 mtime=1763072136.4645\n'
    info = tarfile.TarInfo('././@PaxHeader')
    info.type = tarfile.XGLTYPE if mode == 'global' else tarfile.XHDTYPE
    info.size = 4097 if mode == 'huge' else len(body)
    header = info.tobuf(format=tarfile.USTAR_FORMAT)
    padding = b'X' + b'\0' * (511-len(body)) if mode == 'padding' else b'\0' * (512-len(body))
    prefix = header + body + padding
    if mode == 'consecutive':
        prefix += prefix
    regular = tarfile.TarInfo('jobpars.json')
    regular.size = len(payload)
    suffix = regular.tobuf(format=tarfile.USTAR_FORMAT) + payload + b'\0' * ((-len(payload)) % 512)
    raw = prefix + (b'' if mode == 'orphan' else suffix) + b'\0' * 1024
    path = tmp_path / 'bad.tar.gz'
    path.write_bytes(gzip.compress(raw))
    with pytest.raises(ValueError):
        scan(path, payload)
