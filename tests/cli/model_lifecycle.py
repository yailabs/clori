"""Exercise exact byte acquisition/eviction with a deterministic provider fixture."""
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

exe = str(Path(os.environ.get("YVEX_BIN", "./yvex")).resolve())
root = Path(sys.argv[1]).resolve() / "lifecycle"
root.mkdir(parents=True, exist_ok=True)
models = root / "models"
payload = root / "remote.gguf"
payload.write_bytes(b"artifact" * 8192)
digest = hashlib.sha256(payload.read_bytes()).hexdigest()
local = models / "representations" / digest / "model.gguf"
registry = root / "models.local.json"
logical = "family:deepseek4/model:v4-flash"
revision = "1111111111111111111111111111111111111111"
remote = dict(schema_version=1, status="PUBLISHED", logical_identity=logical,
              artifact_identity=digest, provider="huggingface", repository="fixture/release",
              revision=revision, filename="model.gguf", remote_sha256=digest,
              manifest_filename="release.json", manifest_sha256=digest,
              size_bytes=payload.stat().st_size)
entry = dict(alias="deepseek4-v4-flash-lifecycle-fixture", family="deepseek4", model="v4-flash",
             path=str(local), sha256=digest, file_size=payload.stat().st_size, format="gguf")
registry.write_text(json.dumps(dict(schema="yvex.models.local.v8", working_set=[],
                                    models=[entry], publications=[remote])))
original_registry = registry.read_bytes()
log = root / "provider.log"
env = {**os.environ, "YVEX_MODELS_REGISTRY": str(registry),
       "XDG_CONFIG_HOME": str(root / "config"), "XDG_DATA_HOME": str(root / "data"),
       "YVEX_DATA_DIR": str(root / "yvex-data"), "YVEX_FAKE_HF_LOG": str(log),
       "YVEX_FAKE_HF_RELEASE_FILE": str(payload), "YVEX_FAKE_HF_FAIL_AT_STEP": "0",
       "YVEX_HF_CLI": str(Path("tests/fixtures/bin/fake-hf").resolve())}
pull = ["model", "pull", f"hf://fixture/release@{revision}", "--models-root", str(models), "--json"]
evict = ["model", "evict", "v4-flash", "--variant", digest, "--models-root", str(models), "--json"]

def run(args, *, success=True, extra=None):
    result = subprocess.run([exe, *args], env={**env, **(extra or {})}, capture_output=True, timeout=30)
    assert (result.returncode == 0) == success, (args, result.returncode, result.stdout, result.stderr)
    return json.loads(result.stdout) if success else result

def transfers():
    return log.read_text().count("  --local-dir\n") if log.exists() else 0

run(pull, success=False, extra={"YVEX_FAKE_HF_FAIL_AT_STEP": "1"})
assert not local.exists() and registry.read_bytes() == original_registry
assert list((models / "tmp").rglob("*.incomplete")), "interrupted bytes remain resumable transient state"
before = transfers()
workers = [subprocess.Popen([exe, *pull], env={**env, "YVEX_FAKE_HF_STEP_DELAY": "0.2"},
                            stdout=subprocess.PIPE, stderr=subprocess.PIPE) for _ in range(2)]
for worker in workers:
    out, error = worker.communicate(timeout=30)
    assert worker.returncode == 0, (out, error)
assert transfers() == before + 1, "racing acquisition must invoke only one complete transfer"
assert hashlib.sha256(local.read_bytes()).hexdigest() == digest
before = transfers()
assert run(pull, extra={"YVEX_FAKE_HF_FAIL": "1"})["changed"] is False
assert transfers() == before, "verified pull hit must not contact the provider"
assert run([*evict, "--dry-run"])["changed"] is False and local.exists()
with local.open("rb") as pin:
    fcntl.flock(pin, fcntl.LOCK_SH)
    run(evict, success=False)
    assert local.exists(), "live artifact pins prevent eviction"
assert run(evict)["changed"] is True and not local.exists()
assert run(evict)["changed"] is False
assert registry.read_bytes() == original_registry, "remote identity must survive eviction"
assert run([*pull, "--auth", "never"], extra={"HF_TOKEN": "fixture-only-not-a-credential",
    "YVEX_FAKE_HF_REQUIRE_ANONYMOUS": "1"})["changed"] is True
assert hashlib.sha256(local.read_bytes()).hexdigest() == digest
assert transfers() == before + 1
# Restoring mtime must not revive a verification receipt after a same-size mutation.
old = local.stat()
with local.open("r+b") as stream:
    stream.write(b"X")
os.utime(local, ns=(old.st_atime_ns, old.st_mtime_ns))
before = transfers()
run(pull, success=False)
run(evict, success=False)
assert transfers() == before and local.exists()
# Replace only this deliberately corrupted disposable fixture, then rehydrate from cache.
local.unlink()
before = transfers()
assert run(pull, extra={"YVEX_FAKE_HF_CACHE_FILE": str(payload)})["changed"] is True
assert transfers() == before, "provider cache adoption must not invoke a network transfer"
assert hashlib.sha256(local.read_bytes()).hexdigest() == digest
print("model lifecycle: interrupted/concurrent acquisition, pinned eviction, rehydration and stale identity: ok")

# Local intake through an HF-style snapshot resolves file links without losing names.
cache = root / "hf-cache"
(cache / "blobs").mkdir(parents=True)
snapshot = cache / "snapshots" / revision
snapshot.mkdir(parents=True)
(cache / "blobs" / digest).write_bytes(payload.read_bytes())
(snapshot / "model.gguf").symlink_to("../../blobs/" + digest)
(snapshot / "config.json").write_text('{"model_type":"fixture"}')
adopt_models = root / "adopt-models"
adopt = ["model", "pull", str(snapshot), "--managed", "--models-root", str(adopt_models), "--json"]
first = run(adopt)
second = run(adopt)
assert first["digest"] == second["digest"] and first["location"] == second["location"]
assert (Path(first["location"]) / "model.gguf").read_bytes() == payload.read_bytes()
assert not (Path(first["location"]) / "model.gguf").is_symlink(), "managed placement must survive provider eviction"
# File link spelling preserves GGUF identity despite a content-addressed blob basename.
file_adopt = ["model", "pull", str(snapshot / "model.gguf"), "--managed", "--models-root", str(adopt_models), "--json"]
linked = run(file_adopt)
assert linked["digest"] == digest
alias = root / "renamed.gguf"
alias.write_bytes(payload.read_bytes())
renamed = run(["model", "pull", str(alias), "--managed", "--models-root", str(adopt_models), "--json"])
assert linked["location"] == renamed["location"] and linked["digest"] == renamed["digest"]
# Explicit accounting deduplicates physical blobs shared by a snapshot symlink and a hard link.
(cache / "blobs" / "alias").hardlink_to(cache / "blobs" / digest)
accounting = run(["model", "storage", "v4-flash", "--models-root", str(models), "--include-caches", "--json"],
                 extra={"HF_HUB_CACHE": str(cache), "HF_XET_CACHE": str(root / "no-xet")})
cache_row = next(row for row in accounting["rows"] if row["path"] == str(cache))
assert cache_row["shared_files"] >= 2
assert cache_row["attributable_to_model"] is False
assert accounting["historical_peak_bytes"] is None
print("model lifecycle: HF snapshot and local alias adoption, physical allocation accounting: ok")

# Exact upstream sources may be evicted independently of derived representations.
source_models = root / "source-models"
source_args = ["source", "acquire", "--repo", "fixture/source", "--family", "fixture",
               "--name", "upstream-fixture", "--revision", revision, "--include", "model.gguf",
               "--models-root", str(source_models), "--auth", "never", "--no-native-inventory",
               "--output", "json", "--progress", "off"]
run(source_args)
records = list((source_models / "registry").rglob("*.download.json"))
assert len(records) == 1
record = json.loads(records[0].read_text())
source_path = Path(record["local_source_dir"])
source_digest = record["source_payload_digest"]
# Native selected acquisition and a manually downloaded copy share one durable representation.
native_adopt = run(["model", "pull", str(payload), "--managed", "--models-root", str(source_models), "--json"])
assert native_adopt["changed"] is False and native_adopt["location"] == str(source_path / "model.gguf")
assert not (source_models / "source" / "local").exists(), "native bytes must not be copied on local adoption"
assert len(list((source_models / "registry").rglob("*.download.json"))) == 1
source_evict = ["model", "evict", "upstream-fixture", "--representation", "source",
                "--models-root", str(source_models), "--variant", source_digest, "--json"]
with (source_path / "model.gguf").open("rb") as pin:
    fcntl.flock(pin, fcntl.LOCK_SH)
    run(source_evict, success=False)
assert source_path.exists()
root_pin = os.open(source_path, os.O_RDONLY | os.O_DIRECTORY)
try:
    fcntl.flock(root_pin, fcntl.LOCK_SH)
    run(source_evict, success=False)
finally:
    os.close(root_pin)
assert run([*source_evict, "--dry-run"])["changed"] is False
record_bytes = records[0].read_bytes()
assert run(source_evict)["changed"] is True and not source_path.exists()
assert run(source_evict)["changed"] is False
assert records[0].read_bytes() == record_bytes
# An ordinary unqualified pull uses the retained revision and file selection, not a new main.
run(["model", "pull", "hf://fixture/source", "--models-root", str(source_models), "--json",
     "--auth", "never"])
restored = json.loads(records[0].read_text())
assert restored["revision"] == revision and restored["source_payload_digest"] == source_digest
assert restored["include_patterns"] == ["model.gguf"]
assert (source_path / "model.gguf").read_bytes() == payload.read_bytes()
before = transfers()
run(["source", "acquire", "resume", "upstream-fixture", "--models-root", str(source_models),
     "--auth", "never", "--output", "json", "--progress", "off"])
assert transfers() == before, "resume of a completed exact selection must not transfer again"
print("model lifecycle: source pins, eviction and immutable selected rehydration: ok")

# Manual acquisition first: authoritative remote LFS identity adds provenance without copying bytes.
reverse_models = root / "reverse-models"
manual = run(["model", "pull", str(payload), "--managed", "--models-root", str(reverse_models), "--json"])
remote_env = {"YVEX_FAKE_HF_DISCOVERY_MODE": "tiny", "YVEX_FAKE_HF_TINY_SHA": digest,
              "YVEX_FAKE_HF_TINY_BYTES": str(payload.stat().st_size), "YVEX_FAKE_HF_RESOLVED_SHA": revision}
reverse_pull = ["model", "pull", "hf://fixture/manual", "--format", "gguf",
                "--models-root", str(reverse_models), "--json", "--auth", "never"]
before = transfers()
matched = run(reverse_pull, extra=remote_env)
assert matched["changed"] is False and matched["location"] == manual["location"]
assert run(reverse_pull, extra={"YVEX_FAKE_HF_FAIL": "1"})["changed"] is False
assert transfers() == before and not (reverse_models / "source" / "hf").exists()
reverse_evict = ["model", "evict", "remote", "--representation", "source", "--variant", digest,
                 "--models-root", str(reverse_models), "--json"]
# A legacy/edited moving reference never authorizes eviction or a verified pull hit.
association = next(p for p in (reverse_models / "registry" / "sources").glob("*.source.json")
                   if json.loads(p.read_text())["provider"] == "huggingface")
association_bytes = association.read_bytes()
moving = json.loads(association_bytes)
moving["revision"] = "main"
association.write_text(json.dumps(moving))
run([*reverse_evict, "--dry-run"], success=False)
run(reverse_evict, success=False)
run(reverse_pull, success=False)
assert Path(manual["location"]).exists()
association.write_bytes(association_bytes)
assert run(reverse_evict)["changed"] is True
assert run(reverse_evict)["changed"] is False
assert run(reverse_pull)["changed"] is True
assert Path(manual["location"]).read_bytes() == payload.read_bytes()
print("model lifecycle: manual/native convergence, remote association, eviction and exact file rehydration: ok")
