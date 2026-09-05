#!/usr/bin/env python3
"""Observe an explicit offline preparation stage; never grant release readiness.

The caller owns the command and its input provenance. No shell or environment
dump is used. Receipts feed the existing release assessment, not another catalog.
"""

import argparse
import hashlib
import json
import os
import platform
import shutil
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path


def now():
    return datetime.now(timezone.utc).isoformat()


def checksum(path):
    sha = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(16 * 1024 * 1024), b''):
            sha.update(block)
    return sha.hexdigest()


def source_state(repo):
    def git(*args):
        return subprocess.check_output(['git', '-C', str(repo), *args])
    # Preserve both history and effective tracked contents across shared commits.
    files = {}
    for name in git('ls-files', '-z').decode().split('\0'):
        if name:
            path = repo / name
            files[name] = checksum(path) if path.is_file() else None
    identity = hashlib.sha256(json.dumps(files, sort_keys=True).encode()).hexdigest()
    return {'head': git('rev-parse', 'HEAD').decode().strip(),
            'tree': git('rev-parse', 'HEAD^{tree}').decode().strip(),
            'tracked_contents_sha256': identity,
            'diff_sha256': hashlib.sha256(git('diff', 'HEAD')).hexdigest()}


def storage(root):
    usage = shutil.disk_usage(root)
    return {'total_bytes': usage.total, 'used_bytes': usage.used,
            'free_bytes': usage.free}


def footprint(roots):
    seen, apparent, allocated = set(), 0, 0
    for root in roots:
        paths = [root] if root.is_file() else root.rglob('*') if root.exists() else []
        for path in paths:
            if path.is_symlink() or not path.is_file():
                continue
            st = path.stat()
            if (st.st_dev, st.st_ino) not in seen:
                seen.add((st.st_dev, st.st_ino))
                apparent += st.st_size
                allocated += st.st_blocks * 512
    return {'apparent_bytes': apparent, 'allocated_bytes': allocated}


def observe(spec, directory, repo, interval=2):
    argv = spec['argv']
    if not argv or any(not isinstance(x, str) for x in argv):
        raise ValueError('explicit string argv required')
    if any('token=' in x.lower() or 'authorization' in x.lower() or
           x.lower() in ('--token', '--password', '--secret') for x in argv):
        raise ValueError('credentials must not enter build arguments or logs')
    outputs = [Path(p) for p in spec['outputs']]
    if any(p.exists() or p.is_symlink() for p in outputs):
        raise ValueError('refusing existing output; each observed build needs new bytes')
    root = Path(spec['storage_root'])
    before = storage(root)
    if before['free_bytes'] < spec['required_free_bytes']:
        raise ValueError('insufficient free-space margin')
    directory.mkdir(parents=True, exist_ok=False)
    roots = [Path(p) for p in spec.get('working_paths', spec['outputs'])]
    executable = Path(shutil.which(argv[0]) or argv[0]).resolve()
    binary_sha = checksum(executable)
    initial = source_state(repo)
    input_refs = []
    for item in spec.get('input_evidence', []):
        if checksum(item['path']) != item['sha256']:
            raise ValueError('input evidence checksum mismatch')
        input_refs.append(item)
    record = {'schema': 'yvex.model.release.build.v1', 'stage': spec['stage'],
              'logical_identity': spec['logical_identity'], 'upstream': spec['upstream'],
              'argv': argv, 'cwd': str(repo), 'input_evidence': input_refs,
              'tool': str(executable), 'tool_sha256': binary_sha, 'source_start': initial,
              'observer_path': str(Path(__file__).resolve()),
              'observer_sha256': checksum(__file__),
              'machine': platform.node(), 'environment': dict(zip(
                  ('system', 'node', 'release', 'version', 'machine', 'processor'), platform.uname())),
              'storage_before': before, 'required_free_bytes': spec['required_free_bytes'],
              'working_paths': [str(p) for p in roots], 'working_before': footprint(roots),
              'peak_working_storage_bytes': None,
              'peak_note': 'Sampled allocated high-water is a lower bound, not exact peak.',
              'started_at': now()}
    (directory / 'request.json').write_text(json.dumps(spec, indent=2) + '\n')
    (directory / 'started.json').write_text(json.dumps(record, indent=2) + '\n')
    start = time.monotonic()
    high = record['working_before']['allocated_bytes']
    samples = 0
    with (directory / 'stdout.log').open('wb') as out, (directory / 'stderr.log').open('wb') as err:
        with (directory / 'storage.jsonl').open('w') as log:
            process = subprocess.Popen(argv, cwd=repo, stdout=out, stderr=err)
            while True:
                sample = {'elapsed_seconds': time.monotonic() - start,
                          'filesystem': storage(root), 'working': footprint(roots)}
                high = max(high, sample['working']['allocated_bytes'])
                log.write(json.dumps(sample) + '\n'); log.flush(); samples += 1
                try:
                    code = process.wait(timeout=interval)
                    break
                except subprocess.TimeoutExpired:
                    continue
    record.update(completed_at=now(), duration_seconds=time.monotonic() - start,
                  exit_code=code, storage_after=storage(root), source_finish=source_state(repo),
                  working_after=footprint(roots), storage_samples=samples)
    record['sampled_peak_allocated_bytes'] = max(high, record['working_after']['allocated_bytes'])
    record['source_stable'] = initial['tracked_contents_sha256'] == record['source_finish']['tracked_contents_sha256']
    record['tool_stable'] = binary_sha == checksum(executable)
    record['inputs_stable'] = all(Path(item['path']).is_file() and
        checksum(item['path']) == item['sha256'] for item in input_refs)
    record['observer_stable'] = record['observer_sha256'] == checksum(__file__)
    hash_start = time.monotonic()
    record['outputs'] = []
    for path in outputs:
        if path.is_file():
            st = path.stat()
            sha = checksum(path)
            after = path.stat()
            fields = ('st_dev', 'st_ino', 'st_size', 'st_mtime_ns', 'st_ctime_ns')
            record['outputs'].append({'path': str(path), 'sha256': sha,
                'size_bytes': st.st_size, 'allocated_bytes': st.st_blocks * 512,
                'stat': {k: getattr(after, k) for k in fields},
                'stable_during_checksum': all(getattr(st, k) == getattr(after, k) for k in fields)})
    record['checksum_duration_seconds'] = time.monotonic() - hash_start
    record['status'] = 'PASS' if (code == 0 and record['source_stable'] and record['tool_stable']
        and record['inputs_stable'] and record['observer_stable']
        and len(record['outputs']) == len(outputs)
        and all(x['stable_during_checksum'] for x in record['outputs'])) else 'BLOCKED'
    for name in ('stdout.log', 'stderr.log', 'storage.jsonl', 'request.json'):
        record.setdefault('evidence', []).append({'path': str(directory / name),
                                                'sha256': checksum(directory / name)})
    (directory / 'receipt.json').write_text(json.dumps(record, indent=2) + '\n')
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--request', required=True)
    parser.add_argument('--out-dir', required=True)
    args = parser.parse_args()
    try:
        result = observe(json.loads(Path(args.request).read_text()), Path(args.out_dir),
                         Path.cwd())
    except (OSError, ValueError, KeyError) as exc:
        parser.exit(2, f'release build: {exc}\n')
    print(result['status'], result['duration_seconds'], flush=True)
    return 0 if result['status'] == 'PASS' else 1


if __name__ == '__main__':
    raise SystemExit(main())
