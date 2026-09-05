#!/usr/bin/env python3
"""Build the exact external terminal dependency into a private staged prefix."""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import tarfile
import tempfile
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
PIN = ROOT / 'config/replai.json'
FILES = ('include/replai.h', 'lib/libreplai_c.a', 'lib/libreplai_c.so',
         'lib/pkgconfig/replai.pc', 'share/licenses/replai/LICENSE')


def digest(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def command(argv, **kwargs):
    return subprocess.check_output(argv, text=True, **kwargs).strip()


def prepare(prefix, source_override):
    pin = json.loads(PIN.read_text())
    prefix.parent.mkdir(parents=True, exist_ok=True)
    with (prefix.parent / (prefix.name + '.lock')).open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if source_override:
            source_override = source_override.resolve()
            if command(['git', 'rev-parse', 'HEAD'], cwd=source_override) != pin['revision']:
                raise RuntimeError('REPLAI_SOURCE must be at the exact pinned revision')
            if command(['git', 'status', '--porcelain'], cwd=source_override):
                raise RuntimeError('REPLAI_SOURCE must be clean')
        receipt = prefix / 'replai-build.json'
        if receipt.exists():
            record = json.loads(receipt.read_text())
            if record['pin'] != pin:
                raise RuntimeError('incompatible REPLAI prefix; choose an empty REPLAI_PREFIX')
            for name in FILES:
                if digest(prefix / name) != record['sha256'][name]:
                    raise RuntimeError(f'REPLAI artifact integrity failure: {name}')
            return
        if prefix.exists():
            raise RuntimeError('REPLAI_PREFIX exists without a verified build receipt')
        cargo = shutil.which('cargo')
        if not cargo:
            raise RuntimeError('building YVEX chat requires stable Rust/Cargo; source your Cargo environment')
        cache = Path(os.environ.get('XDG_CACHE_HOME', str(Path.home() / '.cache'))) / 'yvex/replai'
        cache.mkdir(parents=True, exist_ok=True)
        # Downloaded source is an external producer workspace, never a tracked vendor tree.
        with tempfile.TemporaryDirectory(prefix='producer-', dir=cache) as temp:
            work = Path(temp)
            if source_override:
                source = source_override
            else:
                archive = work / 'source.tar.gz'
                url = f"https://codeload.github.com/mothx9/replai/tar.gz/{pin['revision']}"
                print(f'REPLAI: fetching {pin["revision"]}', flush=True)
                with urllib.request.urlopen(url, timeout=60) as incoming, archive.open('wb') as output:
                    shutil.copyfileobj(incoming, output)
                if digest(archive) != pin['archive_sha256']:
                    raise RuntimeError('REPLAI source archive checksum mismatch')
                with tarfile.open(archive) as bundle:
                    bundle.extractall(work, filter='data')
                source = work / ('replai-' + pin['revision'])
            env = os.environ.copy()
            for name in ('MAKEFLAGS', 'MFLAGS', 'CARGO_MAKEFLAGS'):
                env.pop(name, None)
            env['CARGO_TARGET_DIR'] = str(work / 'target')
            subprocess.run([cargo, 'build', '--locked', '--release', '-p', 'replai-c'], cwd=source, env=env, check=True)
            # Producer owns staging. Its source and the adjacent override are never edited.
            staged = work / 'install'
            subprocess.run(['python3', str(source / 'tools/stage_c.py'), '--prefix', str(staged),
                            '--artifacts', str(work / 'target/release')], check=True)
            header = (staged / 'include/replai.h').read_text()
            if f'#define REPLAI_C_ABI_VERSION {pin["abi"]}\n' not in header:
                raise RuntimeError('REPLAI header ABI mismatch')
            record = {'pin': pin, 'rustc': command(['rustc', '--version']), 'rustflags': env.get('RUSTFLAGS', ''),
                      'sha256': {name: digest(staged / name) for name in FILES}}
            (staged / 'replai-build.json').write_text(json.dumps(record, indent=2) + '\n')
            # Copy into a sibling first so a failed producer never publishes a partial prefix.
            with tempfile.TemporaryDirectory(prefix=prefix.name + '-', dir=prefix.parent) as transfer:
                ready = Path(transfer) / 'install'
                shutil.copytree(staged, ready)
                ready.rename(prefix)
        print(f'REPLAI: ABI {pin["abi"]}, verified installation {prefix}', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prefix', required=True, type=Path)
    parser.add_argument('--source', type=Path)
    args = parser.parse_args()
    try:
        prepare(args.prefix.resolve(), args.source)
    except (OSError, RuntimeError, subprocess.CalledProcessError) as error:
        raise SystemExit(f'REPLAI dependency: {error}') from error
