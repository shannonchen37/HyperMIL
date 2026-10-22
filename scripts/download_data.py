#!/usr/bin/env python3
"""Download the versioned public inputs into the configured repository layout."""
import argparse
import csv
import hashlib
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def select_rows(rows, cohort):
    if cohort == 'ALL':
        return rows
    def matches(row):
        parts = Path(row['path']).parts
        return (f'TCGA-{cohort}' in parts or cohort in parts or
                any(p.startswith(f'{cohort}_MGAS_') or p.startswith(f'{cohort}_PASG_') for p in parts))
    return [row for row in rows if matches(row)]


def file_sha256(path):
    digest = hashlib.sha256()
    with path.open('rb') as source:
        for block in iter(lambda: source.read(8 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def validate_release(release, cfg, manifest_path, rows):
    revision = release.get('revision', '')
    if (release.get('status') != 'PASS' or release.get('repo_id') != cfg['repo_id'] or
            not re.fullmatch(r'[0-9a-f]{40}', revision) or
            release.get('manifest_sha256') != file_sha256(manifest_path) or
            release.get('files') != len(rows) or
            release.get('bytes') != sum(int(row['bytes']) for row in rows) or
            release.get('remote_sha256_matches') != len(rows)):
        raise RuntimeError('Published data version does not match this code release. Update the code repository before retrying.')
    return revision


def resolve_revision(cfg, manifest_path, rows):
    from huggingface_hub import hf_hub_download
    from huggingface_hub.errors import EntryNotFoundError
    if cfg.get('revision'):
        revision = cfg['revision']
        if not re.fullmatch(r'[0-9a-f]{40}', revision):
            raise RuntimeError('Dataset revision must be an immutable commit hash.')
        return revision
    try:
        marker = hf_hub_download(repo_id=cfg['repo_id'], repo_type=cfg['repo_type'],
                                 filename=cfg['release_file'], revision='main', token=False)
    except EntryNotFoundError:
        raise RuntimeError('The data upload is still in progress. No complete release is published yet; retry this command after publication.') from None
    return validate_release(json.loads(Path(marker).read_text()), cfg, manifest_path, rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cohort', default='ALL', type=str.upper,
                        choices=['ALL', 'STAD', 'THCA', 'CHOL', 'LIHC'])
    parser.add_argument('--output-root', type=Path, default=ROOT)
    parser.add_argument('--verify', action='store_true', help='Also check all downloaded file hashes')
    args = parser.parse_args()
    import yaml
    cfg = yaml.safe_load((ROOT / 'configs/dataset.yaml').read_text())
    manifest_path = ROOT / cfg['manifest']
    with manifest_path.open(newline='') as source:
        all_rows = list(csv.DictReader(source))
    rows = select_rows(all_rows, args.cohort)
    if not rows:
        parser.error('No matching files in the data manifest')
    for row in rows:
        path = Path(row['path'])
        if path.is_absolute() or '..' in path.parts or path.parts[0] != 'data':
            parser.error(f'Unexpected data path: {path}')
    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        sys.exit('Install the download dependency: pip install huggingface_hub')
    try:
        revision = resolve_revision(cfg, manifest_path, all_rows)
    except RuntimeError as error:
        parser.error(str(error))
    print(f'Dataset revision: {revision}', flush=True)
    print(f"Downloading {len(rows):,} files ({sum(int(r['bytes']) for r in rows) / 1e9:.2f} GB) for {args.cohort}", flush=True)
    snapshot_download(repo_id=cfg['repo_id'], repo_type=cfg['repo_type'],
                      revision=revision, token=False,
                      allow_patterns=[r['path'] for r in rows],
                      local_dir=args.output_root.resolve())
    for row in rows:
        path = args.output_root / row['path']
        if not path.is_file() or path.stat().st_size != int(row['bytes']):
            raise RuntimeError(f'Missing or incomplete data file: {path}')
        if args.verify and file_sha256(path) != row['sha256']:
            raise RuntimeError(f'Data checksum mismatch: {path}')
    print(f'Data ready in {args.output_root.resolve() / "data"}', flush=True)


if __name__ == '__main__':
    main()
