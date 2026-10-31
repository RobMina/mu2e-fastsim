"""Bounded documentary Cat assignment audit; never open physics files or run old audits.

The separately pinned small contract supplies the COMPLETE jobpars schema: `template`
is the exact JSON with its one filename array replaced by null; `filenames_path`,
`group_size_path`, `sequencer_path`, `max_events_path` are arrays of object keys or exact nonnegative list indices;
`expected_items` is the exact source-list count, or null only with an explicit
`exact_jobpars_bytes` pin. In the latter case the count is measured while streaming. Unknown/missing keys, changed scalar
controls, duplicate keys and arrays outside the declared template fail closed.
The contract is not a worker-execution receipt. No assignment offset is fitted.
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import io
import json
import os
from pathlib import Path
import re
import stat
import tarfile

# Dataset archive, relative to the experiment's dCache area (CRVFS_DCACHE); ARCHIVE overrides it when set.
ARCHIVE_IN_AREA = 'persistent/datasets/phy-etc/cnf/mu2e/NeutralsFlashCat/MDC2025ad/tar/ad/f4/cnf.mu2e.NeutralsFlashCat.MDC2025ad.0.tar'
ARCHIVE = None


def archive():
    """The pinned archive: ARCHIVE, else <CRVFS_DCACHE>/mu2e/ARCHIVE_IN_AREA (standard library only, like this module)."""
    if ARCHIVE is not None:
        return Path(ARCHIVE)
    mount = os.environ.get('CRVFS_DCACHE')
    if not mount:
        raise RuntimeError('set CRVFS_DCACHE to the dCache mount point (see site.env.example)')
    return Path(mount.rstrip('/')) / 'mu2e' / ARCHIVE_IN_AREA
ARCHIVE_SHA256 = 'c17855fe607c8233b160e1158767bee5b4f9c80241da1196201297c80b3651fb'
PROVENANCE_SHA256 = '3ffe019bf7f72ee20cc14a136df9a11a5ccca157c282bf095b8ea4483f3c35e1'
ARCHIVE_BYTES, JOBPARS_BYTES = 18068743, 360000549
INDICES = (88, 147, 229, 330, 407, 566, 668, 727, 744, 760, 788, 807)
BLOCK_SIZE = 5000
CHUNK = 65536
STRING_CAP = 4096
SMALL_JSON_CAP = 8 * 1024**2
CONTRACT_CAP = 65536
SAM_CAP = 1024**2
OUTPUT_CAP = 1024**2
MAX_ITEMS = 6000000
MAX_MEMBERS = 8
MAX_REGULAR = 8
SMALL_MEMBER_CAP = 65536
TOTAL_TAR_CAP = JOBPARS_BYTES + 1024**2
FLASH = re.compile(r'dts\.mu2e\.NeutralsFlash\.MDC2025ac\.001430_[0-9]{8}\.art\Z')
CAT = 'dts.mu2e.NeutralsFlashCat.MDC2025ad.001430_{:08d}.art'
SCOPE = ('Declared index*5000 filename blocks versus saved SAM metadata only; no ROOT/art/Parquet, '
         'selected tracks, overlap recomputation, prior audit, physics extraction, weighting or uncertainty model.')


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(payload):
    return hashlib.sha256(payload).hexdigest()


def pin(value):
    require(type(value) is str and re.fullmatch('[0-9a-f]{64}', value) is not None, 'invalid SHA256 pin')
    return value


def parent_hash(names):
    """Exactly source_provenance.sam_summary: sorted basenames and a final newline."""
    return sha(('\n'.join(sorted(names)) + '\n').encode('utf-8'))


def checked_path(value):
    path = Path(os.path.abspath(value))
    require('://' not in str(value), 'local exact path required')
    require(not any(p.endswith(('.root', '.art', '.parquet')) for p in path.parts), 'physics path forbidden')
    return path


def no_symlinks(path):
    for part in [*reversed(path.parents), path]:
        require(not stat.S_ISLNK(part.lstat().st_mode), 'symlink forbidden')


class DirectArchiveReader:
    """Known catalog object: os.read only; even buffered FileIO may issue fstat."""
    def __init__(self, path):
        self.fd = os.open(path, os.O_RDONLY)

    def read(self, size):
        require(type(size) is int and 0 <= size <= CHUNK, 'archive read cap exceeded')
        return os.read(self.fd, size)

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        os.close(self.fd)


def open_regular(path, cap, *, archive=False):
    # The known PNFS pathname is opened directly; never stat or discover the PNFS namespace.
    if not archive:
        no_symlinks(path)
    if archive:
        return DirectArchiveReader(path)
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    try:
        info = os.fstat(fd)
        require(stat.S_ISREG(info.st_mode) and info.st_size <= cap, 'regular file byte cap exceeded')
        return os.fdopen(fd, 'rb')
    except BaseException:
        os.close(fd)
        raise


class BoundedReader:
    """Bound cumulative bytes and hash the actual bytes delivered by a binary reader."""
    def __init__(self, stream, cap):
        self.stream, self.cap, self.count = stream, cap, 0
        self.hash = hashlib.sha256()

    def read(self, size=CHUNK):
        require(type(size) is int and 0 <= size <= 1024**2, 'unbounded read forbidden')
        data = self.stream.read(min(size, CHUNK, self.cap - self.count + 1))
        require(self.count + len(data) <= self.cap, 'stream byte cap exceeded')
        self.count += len(data)
        self.hash.update(data)
        return data


class MemberReader:
    def __init__(self, stream, size):
        self.stream, self.left = stream, size

    def read(self, size):
        data = self.stream.read(min(size, self.left))
        require(data or self.left == 0 or size == 0, 'truncated tar member')
        self.left -= len(data)
        return data


def read_exact(stream, count):
    require(0 <= count <= CHUNK, 'exact-read cap exceeded')
    chunks, remaining = [], count
    while remaining:
        data = stream.read(remaining)
        require(data, 'truncated stream')
        chunks.append(data)
        remaining -= len(data)
    return b''.join(chunks)


def file_hash(path, cap, *, archive=False, exact=None):
    with open_regular(path, cap, archive=archive) as stream:
        reader = BoundedReader(stream, cap)
        while reader.read(CHUNK):
            pass
    require(exact is None or reader.count == exact, 'file size differs from pin')
    return reader.hash.hexdigest()


def small_bytes(path, cap, expected):
    require(file_hash(path, cap) == pin(expected), 'input SHA256 mismatch')
    with open_regular(path, cap) as stream:
        payload = stream.read(cap + 1)
    require(len(payload) <= cap and sha(payload) == expected, 'consumed bytes differ from pin')
    return payload


class JSONStream:
    """Strict bounded JSON tokenizer, with no whole-source-list decoding or retention."""
    string_piece = re.compile(rb'[^"\\\x00-\x1f]*')
    whitespace = re.compile(rb'[ \t\r\n]*')
    plain_string = re.compile(rb'"([^"\\\x00-\x1f]*)"')
    surrogate = re.compile('[\ud800-\udfff]')

    def __init__(self, stream):
        self.stream, self.buf, self.pos = stream, b'', 0
        self.nodes = 0

    def peek(self):
        if self.pos == len(self.buf):
            self.buf, self.pos = self.stream.read(CHUNK), 0
        return self.buf[self.pos:self.pos + 1]

    def skip(self):
        while self.peek():
            self.pos = self.whitespace.match(self.buf, self.pos).end()
            if self.pos < len(self.buf):
                break

    def take(self, char):
        self.skip()
        require(self.peek() == char, 'JSON syntax: expected ' + repr(char))
        self.pos += 1

    def string(self):
        self.skip()
        match = self.plain_string.match(self.buf, self.pos)
        if match is not None:
            require(match.end() - self.pos <= STRING_CAP, 'JSON string cap exceeded')
            self.pos = match.end()
            return match[1].decode('utf-8')
        self.take(b'"')
        pieces, size = [b'"'], 1
        while True:
            require(self.peek(), 'unterminated JSON string')
            end = self.string_piece.match(self.buf, self.pos).end()
            piece = self.buf[self.pos:end]
            pieces.append(piece)
            size += len(piece)
            require(size <= STRING_CAP, 'JSON string cap exceeded')
            self.pos = end
            char = self.peek()
            require(char, 'unterminated JSON string')
            if char == b'"':
                self.pos += 1
                pieces.append(char)
                break
            if char == b'\\':
                self.pos += 1
                escaped = self.peek()
                require(escaped, 'truncated JSON escape')
                pieces.append(b'\\' + escaped)
                size += 2
                self.pos += 1
            else:
                require(char[0] >= 32, 'JSON control character')
        require(size + 1 <= STRING_CAP, 'JSON string cap exceeded')
        value = json.loads(b''.join(pieces))
        require(self.surrogate.search(value) is None, 'unpaired Unicode surrogate')
        return value

    def value(self, template=None, *, path=(), capture=None, expected=False, depth=0):
        require(depth <= 12, 'JSON nesting cap exceeded')
        self.nodes += 1
        require(self.nodes <= 100000, 'small JSON node cap exceeded')
        self.skip()
        if capture is not None and path == capture.path:
            require(template is None and expected, 'filename placeholder must be null')
            capture.read(self)
            return None
        char = self.peek()
        if char == b'{':
            require(not expected or type(template) is dict, 'object schema mismatch')
            self.take(b'{')
            result = {}
            self.skip()
            if self.peek() != b'}':
                while True:
                    key = self.string()
                    require(key not in result and len(result) < 128, 'duplicate/excess object keys')
                    require(not expected or key in template, 'unknown schema key: ' + key)
                    self.take(b':')
                    result[key] = self.value(template[key] if expected else None, path=path + (key,),
                                             capture=capture, expected=expected, depth=depth + 1)
                    self.skip()
                    if self.peek() == b'}':
                        break
                    self.take(b',')
            self.take(b'}')
            require(not expected or set(result) == set(template), 'missing schema keys')
        elif char == b'[':
            require(not expected or type(template) is list, 'array schema mismatch')
            self.take(b'[')
            result = []
            self.skip()
            if self.peek() != b']':
                while True:
                    require(len(result) < 128 and (not expected or len(result) < len(template)), 'array cap/schema exceeded')
                    result.append(self.value(template[len(result)] if expected else None,
                                             path=path + (len(result),), capture=capture,
                                             expected=expected, depth=depth + 1))
                    self.skip()
                    if self.peek() == b']':
                        break
                    self.take(b',')
            self.take(b']')
            require(not expected or len(result) == len(template), 'array schema length mismatch')
        elif char == b'"':
            result = self.string()
        else:
            token = bytearray()
            while self.peek() and self.peek() not in b' \t\r\n,]}':
                token.extend(self.peek())
                self.pos += 1
                require(len(token) <= 64, 'JSON scalar cap exceeded')
            require(re.fullmatch(rb'(?:true|false|null|-?(?:0|[1-9][0-9]*))', token) is not None,
                    'only JSON booleans/null/integers admitted; malformed scalar')
            result = json.loads(token)
        if expected and char not in (b'{', b'['):
            require(type(result) is type(template) and result == template, 'scalar control/type mismatch at ' + repr(path))
        return result

    def finish(self):
        self.skip()
        require(not self.peek(), 'trailing JSON content')


def small_json(payload):
    parser = JSONStream(io.BytesIO(payload))
    value = parser.value()
    parser.finish()
    return value


def at(value, path):
    require(type(path) is list and 0 < len(path) <= 8
            and all(type(x) is str or (type(x) is int and x >= 0) for x in path),
            'invalid contract path')
    for key in path:
        if type(key) is str:
            require(type(value) is dict and key in value, 'contract object path missing')
        else:
            require(type(value) is list and key < len(value), 'contract list index missing')
        value = value[key]
    return value


def validate_contract(contract, indices=INDICES, block_size=BLOCK_SIZE):
    keys = {'template', 'filenames_path', 'group_size_path',
            'sequencer_path', 'max_events_path', 'expected_items'}
    require(type(contract) is dict and keys <= set(contract)
            and set(contract) <= keys | {'exact_jobpars_bytes'}, 'invalid contract schema')
    template = contract['template']
    require(type(template) is dict, 'root schema must be object')
    paths = dict(filenames_path=['tbs', 'inputs', 'source.fileNames', 1],
                 group_size_path=['tbs', 'inputs', 'source.fileNames', 0],
                 sequencer_path=['tbs', 'sequencer_from_index'],
                 max_events_path=['tbs', 'event_id', 'source.maxEvents'])
    for key, path in paths.items():
        at(template, contract[key])  # Validate exact path types (bool is not index 0/1).
        require(contract[key] == path, 'assignment control must be explicitly bound: ' + key)
    require(at(template, contract['filenames_path']) is None, 'filename placeholder must be null')
    for path, wanted in (('group_size_path', block_size), ('sequencer_path', True), ('max_events_path', 2147483647)):
        value = at(template, contract[path])
        require(type(value) is type(wanted) and value == wanted, 'invalid pinned assignment control: ' + path)
    exact = contract.get('exact_jobpars_bytes')
    if 'exact_jobpars_bytes' in contract:
        require(type(exact) is int and 0 < exact <= JOBPARS_BYTES, 'invalid exact jobpars byte pin')
    count = contract['expected_items']
    if count is None:
        require(exact is not None, 'null expected_items requires exact_jobpars_bytes pin')
    else:
        require(type(count) is int and (max(indices) + 1) * block_size <= count <= MAX_ITEMS
                and count % block_size == 0, 'invalid expected filename count')


class Blocks:
    def __init__(self, path, count, indices=INDICES, block_size=BLOCK_SIZE):
        self.path, self.count = tuple(path), count
        self.indices, self.block_size = frozenset(indices), block_size
        self.rows, self.seen = {}, False

    def read(self, parser):
        require(not self.seen, 'repeated filename list')
        self.seen = True
        parser.take(b'[')
        block = []
        expected, i = self.count, 0
        parser.skip()
        while parser.peek() != b']':
            require(i < MAX_ITEMS and (expected is None or i < expected), 'filename count cap exceeded')
            if i:
                parser.take(b',')
            name = parser.string()
            require(len(name) <= 128 and FLASH.fullmatch(name) is not None, 'invalid Flash filename/type')
            index = i // self.block_size
            if index in self.indices:
                block.append(name)
                if (i + 1) % self.block_size == 0:
                    self.rows[index] = dict(start=index * self.block_size, stop=(index + 1) * self.block_size,
                        count=len(block), unique_count=len(set(block)), parents_sha256=parent_hash(block))
                    block = []
            i += 1
            parser.skip()
        parser.take(b']')
        require(expected is None or i == expected, 'filename count differs from pin')
        require(i > 0 and i % self.block_size == 0, 'filename list must contain whole blocks')
        require(set(self.rows) == self.indices, 'target blocks missing')
        self.count = i



def scan_jobpars(stream, size, contract, *, indices=INDICES, block_size=BLOCK_SIZE):
    validate_contract(contract, indices, block_size)
    require('exact_jobpars_bytes' not in contract or contract['exact_jobpars_bytes'] == size,
            'jobpars size differs from contract byte pin')
    reader = BoundedReader(stream, size)
    parser = JSONStream(reader)
    blocks = Blocks(contract['filenames_path'], contract['expected_items'], indices, block_size)
    parser.value(contract['template'], expected=True, capture=blocks)
    parser.finish()
    require(reader.count == size and blocks.seen, 'jobpars size/list binding mismatch')
    return dict(bytes=reader.count, sha256=reader.hash.hexdigest(), filename_count=blocks.count, blocks=blocks.rows)


def pax_times(payload):
    result, pos = {}, 0
    while pos < len(payload):
        end = payload.find(b' ', pos)
        require(end > pos and end - pos <= 4, 'invalid PAX length')
        digits = payload[pos:end]
        require(re.fullmatch(rb'[1-9][0-9]*', digits) is not None, 'invalid PAX length')
        stop = pos + int(digits)
        require(end + 1 < stop <= len(payload) and payload[stop-1:stop] == b'\n', 'invalid PAX record boundary')
        record = payload[end+1:stop-1]
        match = re.fullmatch(rb'(mtime|atime|ctime)=(-?[0-9]{1,20}(?:\.[0-9]{1,20})?)', record)
        require(match is not None, 'only bounded PAX timestamps admitted')
        key, value = match[1].decode(), match[2].decode()
        require(key not in result, 'duplicate PAX timestamp')
        result[key] = value
        pos = stop
    require(result, 'empty PAX header')
    return result


def scan_archive(path, contract, *, archive_size=ARCHIVE_BYTES, jobpars_size=JOBPARS_BYTES,
                 indices=INDICES, block_size=BLOCK_SIZE):
    inventory, result, names, regular = [], None, set(), 0
    pending_pax = False
    with open_regular(path, archive_size, archive=True) as raw:
        compressed = BoundedReader(raw, archive_size)
        with gzip.GzipFile(fileobj=compressed, mode='rb') as zipped:
            stream = BoundedReader(zipped, min(TOTAL_TAR_CAP, jobpars_size + 1024**2))
            while True:
                header = read_exact(stream, 512)
                if header == b'\0' * 512:
                    require(not pending_pax, 'orphan PAX header')
                    require(read_exact(stream, 512) == b'\0' * 512, 'tar requires two zero terminators')
                    while True:
                        padding = stream.read(CHUNK)
                        if not padding:
                            break
                        require(not padding.strip(b'\0'), 'nonzero/trailing tar data')
                    break
                require(len(inventory) < MAX_MEMBERS, 'tar member cap exceeded')
                member = tarfile.TarInfo.frombuf(header, 'utf-8', 'strict')
                if member.type == tarfile.XHDTYPE:
                    require(not pending_pax and member.name == '././@PaxHeader' and not member.linkname,
                            'invalid/consecutive PAX header')
                    require(0 < member.size <= 4096, 'PAX byte cap exceeded')
                    payload = read_exact(stream, member.size)
                    times = pax_times(payload)
                    require(not read_exact(stream, (-member.size) % 512).strip(b'\0'), 'nonzero PAX padding')
                    inventory.append(dict(name=member.name, bytes=member.size, sha256=sha(payload), timestamps=times))
                    pending_pax = True
                    continue
                require(member.type in (tarfile.REGTYPE, tarfile.AREGTYPE), 'non-regular/extension tar member forbidden')
                pending_pax = False
                regular += 1
                require(regular <= MAX_REGULAR, 'regular member cap exceeded')
                require(re.fullmatch(r'[A-Za-z0-9_-]+\.(json|fcl|txt|sh)', member.name) is not None,
                        'unsafe or non-documentary tar member name')
                require(member.name not in names and not member.linkname, 'duplicate/link tar member')
                names.add(member.name)
                wanted = member.name == 'jobpars.json'
                require(0 <= member.size <= (jobpars_size if wanted else SMALL_MEMBER_CAP), 'tar member byte cap exceeded')
                require(not wanted or member.size == jobpars_size, 'jobpars declared size differs')
                limited = MemberReader(stream, member.size)
                if wanted:
                    result = scan_jobpars(limited, member.size, contract, indices=indices, block_size=block_size)
                    digest = result['sha256']
                else:
                    reader = BoundedReader(limited, member.size)
                    while reader.read(CHUNK):
                        pass
                    digest = reader.hash.hexdigest()
                require(limited.left == 0, 'member not fully consumed')
                require(not read_exact(stream, (-member.size) % 512).strip(b'\0'), 'nonzero member padding')
                inventory.append(dict(name=member.name, bytes=member.size, sha256=digest))
        require(not compressed.read(1), 'compressed bytes not fully consumed')
    require(compressed.count == archive_size and result is not None, 'archive size/jobpars missing')
    return dict(archive_read_sha256=compressed.hash.hexdigest(), archive_read_bytes=compressed.count,
                tar_bytes=stream.count, members=inventory, jobpars=result)


def parse_sam(payload):
    fields, current = {}, None
    for line in payload.decode('utf-8').splitlines():
        if not line.strip():
            continue
        require(len(line) <= STRING_CAP, 'SAM line cap exceeded')
        match = re.fullmatch(r'\s*([A-Za-z][A-Za-z0-9_. -]*):\s*(.*)', line)
        if match and current == 'Checksum' and match[1].strip() in ('adler32', 'enstore'):
            fields[current].append(line.strip())
        elif match:
            current = match[1].strip()
            require(current not in fields and len(fields) < 128, 'duplicate/excess SAM field')
            fields[current] = [match[2].strip()] if match[2].strip() else []
        else:
            require(current is not None and line[0].isspace(), 'malformed SAM continuation')
            fields[current].append(line.strip())
        require(len(fields[current]) <= BLOCK_SIZE, 'SAM field item cap exceeded')
    parents = fields.pop('Parents', [])
    require(len(parents) == BLOCK_SIZE and len(set(parents)) == BLOCK_SIZE
            and all(FLASH.fullmatch(p) for p in parents), 'invalid SAM parents')
    return {k: '\n'.join(v) for k, v in fields.items()}, parents


def saved_metadata(root, provenance):
    sources = provenance.get('sources')
    names = {CAT.format(i): i for i in INDICES}
    require(type(sources) is dict and set(sources) == set(names), 'exact twelve retained filenames required')
    rows, bindings = {}, {}
    for name, index in names.items():
        record = sources[name]
        require(type(record) is dict and type(record.get('sam')) is dict, 'saved SAM binding missing')
        sam = record['sam']
        require(sam.get('file_name') == name and type(sam.get('parent_count')) is int
                and sam['parent_count'] == BLOCK_SIZE, 'saved filename/count mismatch')
        pin(sam.get('parents_sha256'))
        pin(sam.get('source_sha256'))
        require(type(sam.get('event_count')) is int and sam['event_count'] >= 0
                and type(sam.get('file_id')) is str, 'saved event/file-id type mismatch')
        path = root / f'sam_{index}.txt'
        digest = pin(sam.get('metadata_sha256'))
        fields, parents = parse_sam(small_bytes(path, SAM_CAP, digest))
        require(fields.get('File Name') == name and fields.get('dh.sequencer') == f'001430_{index:08d}', 'SAM filename/sequencer mismatch')
        require(fields.get('Event Count') == str(sam['event_count']) and fields.get('File Id') == sam['file_id']
                and fields.get('dh.sha256') == sam['source_sha256'], 'SAM metadata binding mismatch')
        require(parent_hash(parents) == sam['parents_sha256'], 'SAM parent hash mismatch')
        rows[index] = dict(file_name=name, metadata_sha256=digest, parent_count=BLOCK_SIZE,
                           parents_sha256=sam['parents_sha256'], metadata_pins_verified=True)
        bindings[path] = digest
    return rows, bindings


def compare(blocks, saved):
    require(set(blocks) == set(saved), 'comparison indices differ')
    rows = []
    for index in sorted(saved):
        block, sam = blocks[index], saved[index]
        matches = (block['count'] == sam['parent_count'] and block['unique_count'] == block['count']
                   and block['parents_sha256'] == sam['parents_sha256'])
        rows.append(dict(index=index, assignment=block, saved_sam=sam, matches=matches))
    return rows


def outside(out, excluded):
    require(not os.path.lexists(out), 'output exists; overwrite forbidden')
    no_symlinks(out.parent)
    require(out.parent.is_dir(), 'existing output directory required')
    require(all(not out.is_relative_to(p) for p in excluded), 'output must be outside old evidence and code')


def run(root, contract, out, *, expected_contract_sha256, expected_archive_sha256,
        expected_provenance_sha256):
    require(pin(expected_archive_sha256) == ARCHIVE_SHA256, 'wrong campaign archive pin')
    require(pin(expected_provenance_sha256) == PROVENANCE_SHA256, 'wrong retained provenance pin')
    root, contract, out = map(checked_path, (root, contract, out))
    require(root.name == 'continuation_v1', 'exact continuation_v1 evidence directory required')
    code = Path(__file__).absolute()
    excluded = (root.parent, contract.parent, code.parents[2])
    outside(out, excluded)
    pp = root / 'provenance_final.json'
    source = small_bytes(code, CONTRACT_CAP, file_hash(code, CONTRACT_CAP))
    source_sha = sha(source)
    contract_payload = small_bytes(contract, CONTRACT_CAP, expected_contract_sha256)
    spec = small_json(contract_payload)
    validate_contract(spec, INDICES, BLOCK_SIZE)
    provenance = small_json(small_bytes(pp, SMALL_JSON_CAP, expected_provenance_sha256))
    saved, bindings = saved_metadata(root, provenance)
    bindings.update({pp: expected_provenance_sha256, contract: expected_contract_sha256, code: source_sha})
    archive_path = archive()
    before = file_hash(archive_path, ARCHIVE_BYTES, archive=True, exact=ARCHIVE_BYTES)
    require(before == expected_archive_sha256, 'archive before SHA256 mismatch')
    try:
        scanned = scan_archive(archive_path, spec)
    finally:
        after = file_hash(archive_path, ARCHIVE_BYTES, archive=True, exact=ARCHIVE_BYTES)
        require(after == before, 'archive changed after streaming')
    require(scanned['archive_read_sha256'] == before, 'streamed compressed bytes differ from pin')
    rows = compare(scanned['jobpars']['blocks'], saved)
    result = dict(schema='cat-jobdef-assignment-audit-v1', scope=SCOPE,
        assignment_model={'formula': 'block = filenames[index*5000:(index+1)*5000]',
            'index_origin': 'exact output filename sequencer; zero-based model bound to sequencer_from_index=true',
            'sequencer_path': spec['sequencer_path'], 'group_size_path': spec['group_size_path'],
            'not_established': 'Actual worker job-index/recovery mapping, deployed tool version or execution.'},
        contract=spec, contract_sha256=expected_contract_sha256,
        source_sha256=source_sha, source_snapshot=source.decode('utf-8'),
        inputs={str(p): digest for p, digest in bindings.items()}, archive_path=str(archive_path),
        archive_before_sha256=before, archive_after_sha256=after, **scanned,
        comparisons=rows, mismatch_indices=[r['index'] for r in rows if not r['matches']],
        all_blocks_match=all(r['matches'] for r in rows),
        parent_hash_convention='SHA256(UTF8(newline.join(sorted(exact_basenames))+newline))',
        limits=dict(chunk_bytes=CHUNK, string_bytes=STRING_CAP, source_items=MAX_ITEMS,
                    members=MAX_MEMBERS, regular_members=MAX_REGULAR, tar_bytes=TOTAL_TAR_CAP,
                    small_member_bytes=SMALL_MEMBER_CAP, output_bytes=OUTPUT_CAP,
                    retained_filenames=BLOCK_SIZE, json_depth=12, object_keys=128, small_array_items=128),
        limitations=['No cross-block overlap analysis or global duplicate census.',
                    'Only targeted blocks retain names; duplicates inside those blocks are reported as mismatches.',
                    'Strict integer-only JSON profile; only bounded local PAX timestamps admitted; other extensions, links and paths rejected.',
                    'Exclusive output, not atomic publication or crash-durability certification.'],
        physics_executed=False, prior_audit_executed=False, parent_overlap_recomputed=False)
    payload = (json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + '\n').encode('utf-8')
    require(len(payload) <= OUTPUT_CAP, 'output byte cap exceeded')
    for path, digest in bindings.items():
        cap = SMALL_JSON_CAP if path == pp else SAM_CAP if path.parent == root else CONTRACT_CAP
        require(file_hash(path, cap) == digest, 'input/source changed after read')
    outside(out, excluded)
    with out.open('xb') as stream:
        stream.write(payload)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root')
    for name in ('contract', 'out', 'expected-contract-sha256', 'expected-archive-sha256', 'expected-provenance-sha256'):
        parser.add_argument('--' + name, required=True)
    result = run(**vars(parser.parse_args(argv)))
    print(json.dumps({k: result[k] for k in ('schema', 'all_blocks_match', 'mismatch_indices', 'source_sha256')}))
    return 0 if result['all_blocks_match'] else 2


if __name__ == '__main__':
    raise SystemExit(main())
