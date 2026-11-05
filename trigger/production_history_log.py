"""Bind an original production log to its audited configuration and regeneration manifest."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import regenerate
import source_audit


def _read(path):
    path = Path(path).resolve()
    data = path.read_bytes()
    return data, {'path': str(path), 'bytes': len(data),
                  'sha256': hashlib.sha256(data).hexdigest()}


def verify(config_text, regeneration_manifest_path, production_log_path) -> dict:
    """Recheck original log history locally; no parent payload or physics certification."""
    audit = source_audit.audit(config_text)
    manifest_bytes, manifest_file = _read(regeneration_manifest_path)
    log_bytes, log_file = _read(production_log_path)
    manifest = json.loads(manifest_bytes)
    if not isinstance(manifest, dict):
        raise ValueError('regeneration manifest must be a JSON object')
    config_hash = hashlib.sha256(config_text.encode()).hexdigest()
    if audit['config_sha256'] != config_hash or manifest.get('config_sha256') != config_hash:
        raise ValueError('production history configuration hash differs from regeneration manifest')
    history = regenerate.verify_history(log_bytes.decode('utf-8'), audit)
    if (manifest.get('production_log_sha256') != history['production_log_sha256']
            or history['production_log_sha256'] != log_file['sha256']):
        raise ValueError('original production log hash differs from regeneration manifest')
    count = manifest.get('original_processed_events')
    if type(count) is not int or count != history['original_processed_events']:
        raise ValueError('original processed-event count differs from regeneration manifest')
    if (manifest.get('opened_parent_order_verified') is not True
            or history['opened_parent_order_verified'] is not True):
        raise ValueError('original production opened-parent order is not verified')
    parents = manifest.get('parents')
    if not isinstance(parents, list) or not parents or parents != audit['parents']:
        raise ValueError('regeneration manifest parents differ from audited production order')
    local_paths = manifest.get('local_parent_paths')
    if not isinstance(local_paths, list) or len(local_paths) != len(parents):
        raise ValueError('need one local staged path per original production parent')
    for parent, path in zip(parents, local_paths):
        if (not isinstance(path, str) or not path or '://' in path
                or Path(path).name != parent['did'].split(':', 1)[1]):
            raise ValueError('local staged parent basename differs from corresponding audited DID')
    return {**history, 'config_sha256': config_hash,
            'production_history_log_verified': True,
            'parents': parents, 'local_parent_paths': local_paths,
            'regeneration_manifest_path': manifest_file['path'],
            'production_log_path': log_file['path'],
            'evidence_files': [manifest_file, log_file],
            'scope': 'original production log/configuration binding only; no parent payload or physics certification'}


def main():
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('config', 'regeneration-manifest', 'production-log', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError('refuse production history log receipt overwrite')
    report = verify(args.config.read_text(), args.regeneration_manifest, args.production_log)
    args.output.write_text(json.dumps(report, indent=2) + '\n')
    print(f"original production log verified: {report['original_processed_events']} events")


if __name__ == '__main__':
    main()
