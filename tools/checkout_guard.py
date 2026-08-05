#!/usr/bin/env python3
"""Serialize mutation of one shared Git checkout with an explicit local lease."""

from __future__ import annotations

import argparse
import contextlib
import fcntl
import json
import os
import pathlib
import re
import stat
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from typing import Iterator


SCHEMA_VERSION = 1
STATE_DIRECTORY = "yvex"
LEASE_FILENAME = "checkout_lease.json"
MUTEX_FILENAME = "checkout_guard.mutex"
MAX_LEASE_BYTES = 16 * 1024
OWNER_PATTERN = re.compile(r"[a-z0-9][a-z0-9._-]{2,63}")
COMMIT_PATTERN = re.compile(r"[0-9a-f]{40,64}")
LEASE_FIELDS = {
    "schema_version",
    "owner_id",
    "assigned_branch",
    "repository_root",
    "acquired_head",
    "upstream_ref",
    "upstream_head",
}
GIT_OPERATION_MARKERS = (
    "MERGE_HEAD",
    "CHERRY_PICK_HEAD",
    "REVERT_HEAD",
    "BISECT_LOG",
    "rebase-merge",
    "rebase-apply",
    "sequencer",
)


class Refusal(RuntimeError):
    """A fail-closed checkout state or lease transition."""


@dataclass(frozen=True)
class Repository:
    root: pathlib.Path
    git_directory: pathlib.Path
    state_directory: pathlib.Path
    lease_path: pathlib.Path
    mutex_path: pathlib.Path


@dataclass(frozen=True)
class Lease:
    owner_id: str
    assigned_branch: str
    repository_root: str
    acquired_head: str
    upstream_ref: str | None
    upstream_head: str | None

    def as_json(self) -> dict[str, object]:
        return {
            "acquired_head": self.acquired_head,
            "assigned_branch": self.assigned_branch,
            "owner_id": self.owner_id,
            "repository_root": self.repository_root,
            "schema_version": SCHEMA_VERSION,
            "upstream_head": self.upstream_head,
            "upstream_ref": self.upstream_ref,
        }


def git(cwd: pathlib.Path, *arguments: str, allow_failure: bool = False) -> str:
    result = subprocess.run(
        ["git", *arguments],
        cwd=cwd,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if result.returncode != 0 and not allow_failure:
        detail = result.stderr.strip() or result.stdout.strip() or "git command failed"
        raise Refusal(detail)
    if result.returncode != 0:
        return ""
    return result.stdout.strip()


def discover(repository: str) -> Repository:
    requested = pathlib.Path(repository).resolve()
    root_text = git(requested, "rev-parse", "--show-toplevel")
    git_text = git(requested, "rev-parse", "--absolute-git-dir")
    root = pathlib.Path(root_text).resolve()
    git_directory = pathlib.Path(git_text).resolve()
    state_directory = git_directory / STATE_DIRECTORY
    return Repository(
        root=root,
        git_directory=git_directory,
        state_directory=state_directory,
        lease_path=state_directory / LEASE_FILENAME,
        mutex_path=state_directory / MUTEX_FILENAME,
    )


def ensure_state_directory(repository: Repository) -> None:
    path = repository.state_directory
    if path.is_symlink():
        raise Refusal("Git-local checkout guard state is a symlink")
    if path.exists():
        if not path.is_dir():
            raise Refusal("Git-local checkout guard state is not a directory")
    else:
        path.mkdir(mode=0o700)
    os.chmod(path, 0o700)


@contextlib.contextmanager
def exclusive_operation(repository: Repository) -> Iterator[None]:
    ensure_state_directory(repository)
    flags = os.O_RDWR | os.O_CREAT | os.O_CLOEXEC
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        descriptor = os.open(repository.mutex_path, flags, 0o600)
    except OSError as error:
        raise Refusal(f"cannot open checkout guard mutex: {error}") from error
    try:
        if not stat.S_ISREG(os.fstat(descriptor).st_mode):
            raise Refusal("checkout guard mutex is not a regular file")
        os.fchmod(descriptor, 0o600)
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise Refusal("another checkout guard operation is active") from error
        yield
    finally:
        os.close(descriptor)


def validate_owner(owner_id: str) -> None:
    if OWNER_PATTERN.fullmatch(owner_id) is None:
        raise Refusal("owner ID must use 3-64 lowercase letters, digits, dots, underscores, or dashes")


def validate_branch(repository: Repository, branch: str) -> None:
    result = subprocess.run(
        ["git", "check-ref-format", "--branch", branch],
        cwd=repository.root,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if result.returncode != 0:
        raise Refusal("assigned branch is not a valid local branch name")
    reference = f"refs/heads/{branch}"
    if not git(repository.root, "show-ref", "--verify", reference, allow_failure=True):
        raise Refusal(f"assigned branch does not exist locally: {branch}")


def current_branch(repository: Repository) -> str:
    branch = git(repository.root, "symbolic-ref", "--quiet", "--short", "HEAD", allow_failure=True)
    if not branch:
        raise Refusal("checkout has detached HEAD")
    return branch


def current_head(repository: Repository) -> str:
    head = git(repository.root, "rev-parse", "--verify", "HEAD")
    if COMMIT_PATTERN.fullmatch(head) is None:
        raise Refusal("checkout HEAD is not a full commit identity")
    return head


def current_upstream(repository: Repository) -> tuple[str | None, str | None]:
    upstream = git(
        repository.root,
        "rev-parse",
        "--abbrev-ref",
        "--symbolic-full-name",
        "@{upstream}",
        allow_failure=True,
    )
    if not upstream:
        return None, None
    head = git(repository.root, "rev-parse", "--verify", "@{upstream}")
    if COMMIT_PATTERN.fullmatch(head) is None:
        raise Refusal("upstream does not resolve to a full commit identity")
    return upstream, head


def operation_markers(repository: Repository) -> list[str]:
    found: list[str] = []
    for marker in GIT_OPERATION_MARKERS:
        path_text = git(repository.root, "rev-parse", "--git-path", marker)
        path = pathlib.Path(path_text)
        if not path.is_absolute():
            path = repository.root / path
        if path.exists():
            found.append(marker)
    return found


def git_lock_paths(repository: Repository) -> list[pathlib.Path]:
    locks: list[pathlib.Path] = []
    for path in repository.git_directory.rglob("*.lock"):
        try:
            path.relative_to(repository.state_directory)
        except ValueError:
            locks.append(path)
    return sorted(locks)


def ensure_no_git_operation(repository: Repository) -> None:
    markers = operation_markers(repository)
    if markers:
        raise Refusal(f"Git operation is active: {', '.join(markers)}")
    locks = git_lock_paths(repository)
    if locks:
        names = ", ".join(str(path.relative_to(repository.git_directory)) for path in locks)
        raise Refusal(f"Git lock is active: {names}")


def dirty_entries(repository: Repository) -> list[str]:
    output = git(repository.root, "status", "--porcelain=v2", "--untracked-files=normal")
    return output.splitlines() if output else []


def ensure_clean(repository: Repository) -> None:
    dirty = dirty_entries(repository)
    if dirty:
        raise Refusal(f"checkout is dirty ({len(dirty)} entries)")


def parse_optional_commit(value: object, field: str) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str) or COMMIT_PATTERN.fullmatch(value) is None:
        raise Refusal(f"checkout lease has invalid {field}")
    return value


def read_lease(repository: Repository) -> Lease | None:
    path = repository.lease_path
    if not path.exists() and not path.is_symlink():
        return None
    if path.is_symlink():
        raise Refusal("checkout lease is a symlink")
    metadata = path.stat()
    if not stat.S_ISREG(metadata.st_mode):
        raise Refusal("checkout lease is not a regular file")
    if metadata.st_size > MAX_LEASE_BYTES:
        raise Refusal("checkout lease exceeds its size bound")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise Refusal(f"checkout lease is malformed: {error}") from error
    if not isinstance(value, dict) or set(value) != LEASE_FIELDS:
        raise Refusal("checkout lease fields do not match schema version 1")
    if value["schema_version"] != SCHEMA_VERSION:
        raise Refusal("checkout lease schema version is unsupported")
    owner_id = value["owner_id"]
    branch = value["assigned_branch"]
    root = value["repository_root"]
    acquired_head = value["acquired_head"]
    upstream_ref = value["upstream_ref"]
    if not isinstance(owner_id, str):
        raise Refusal("checkout lease has invalid owner ID")
    validate_owner(owner_id)
    if not isinstance(branch, str):
        raise Refusal("checkout lease has invalid assigned branch")
    validate_branch(repository, branch)
    if not isinstance(root, str) or not root:
        raise Refusal("checkout lease has invalid repository root")
    if not isinstance(acquired_head, str) or COMMIT_PATTERN.fullmatch(acquired_head) is None:
        raise Refusal("checkout lease has invalid acquired HEAD")
    if upstream_ref is not None and (not isinstance(upstream_ref, str) or not upstream_ref):
        raise Refusal("checkout lease has invalid upstream reference")
    upstream_head = parse_optional_commit(value["upstream_head"], "upstream HEAD")
    if (upstream_ref is None) != (upstream_head is None):
        raise Refusal("checkout lease upstream fields disagree")
    return Lease(owner_id, branch, root, acquired_head, upstream_ref, upstream_head)


def ensure_live_lease(repository: Repository, lease: Lease) -> None:
    if lease.repository_root != str(repository.root):
        raise Refusal("stale checkout lease: repository root changed")
    branch = current_branch(repository)
    if branch != lease.assigned_branch:
        raise Refusal(
            f"stale checkout lease: assigned branch is {lease.assigned_branch}, current branch is {branch}"
        )


def ensure_holder(lease: Lease, owner_id: str, branch: str) -> None:
    if lease.owner_id != owner_id:
        raise Refusal(f"checkout is owned by another session: {lease.owner_id}")
    if lease.assigned_branch != branch:
        raise Refusal(f"checkout lease belongs to branch {lease.assigned_branch}")


def write_lease(repository: Repository, lease: Lease) -> None:
    payload = json.dumps(lease.as_json(), indent=2, sort_keys=True) + "\n"
    descriptor, temporary_name = tempfile.mkstemp(
        prefix="checkout_lease.", suffix=".tmp", dir=repository.state_directory
    )
    temporary = pathlib.Path(temporary_name)
    try:
        os.fchmod(descriptor, 0o600)
        with os.fdopen(descriptor, "w", encoding="utf-8") as output:
            output.write(payload)
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary, repository.lease_path)
        directory = os.open(repository.state_directory, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        if temporary.exists():
            temporary.unlink()


def remove_lease(repository: Repository) -> None:
    repository.lease_path.unlink()
    directory = os.open(repository.state_directory, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(directory)
    finally:
        os.close(directory)


def new_lease(repository: Repository, owner_id: str, branch: str) -> Lease:
    upstream_ref, upstream_head = current_upstream(repository)
    return Lease(
        owner_id=owner_id,
        assigned_branch=branch,
        repository_root=str(repository.root),
        acquired_head=current_head(repository),
        upstream_ref=upstream_ref,
        upstream_head=upstream_head,
    )


def acquire(repository: Repository, owner_id: str, branch: str) -> None:
    validate_owner(owner_id)
    validate_branch(repository, branch)
    with exclusive_operation(repository):
        ensure_no_git_operation(repository)
        existing = read_lease(repository)
        if existing is not None:
            ensure_live_lease(repository, existing)
            ensure_holder(existing, owner_id, branch)
            print(f"checkout ownership already held: owner={owner_id} branch={branch}")
            return
        ensure_clean(repository)
        actual = current_branch(repository)
        if actual != branch:
            raise Refusal(f"wrong branch: assigned {branch}, current {actual}")
        write_lease(repository, new_lease(repository, owner_id, branch))
        print(f"checkout ownership acquired: owner={owner_id} branch={branch}")


def check(repository: Repository, owner_id: str, branch: str) -> None:
    validate_owner(owner_id)
    validate_branch(repository, branch)
    with exclusive_operation(repository):
        ensure_no_git_operation(repository)
        lease = read_lease(repository)
        if lease is None:
            raise Refusal("checkout is unowned")
        ensure_live_lease(repository, lease)
        ensure_holder(lease, owner_id, branch)
        dirty = len(dirty_entries(repository))
        print(f"checkout ownership valid: owner={owner_id} branch={branch} dirty_entries={dirty}")


def status_command(repository: Repository) -> None:
    with exclusive_operation(repository):
        ensure_no_git_operation(repository)
        lease = read_lease(repository)
        if lease is None:
            ensure_clean(repository)
            print(f"checkout ownership: unowned branch={current_branch(repository)} clean=true")
            return
        ensure_live_lease(repository, lease)
        dirty = len(dirty_entries(repository))
        print(
            "checkout ownership: "
            f"owner={lease.owner_id} branch={lease.assigned_branch} dirty_entries={dirty}"
        )


def release(repository: Repository, owner_id: str, branch: str) -> None:
    validate_owner(owner_id)
    validate_branch(repository, branch)
    with exclusive_operation(repository):
        ensure_no_git_operation(repository)
        lease = read_lease(repository)
        if lease is None:
            raise Refusal("checkout is already unowned")
        ensure_live_lease(repository, lease)
        ensure_holder(lease, owner_id, branch)
        ensure_clean(repository)
        head = current_head(repository)
        if head != lease.acquired_head:
            upstream_ref, upstream_head = current_upstream(repository)
            if upstream_ref is None or upstream_head != head:
                raise Refusal("checkout HEAD changed but is not published to its configured upstream")
        remove_lease(repository)
        print(f"checkout ownership released: owner={owner_id} branch={branch}")


def switch(repository: Repository, owner_id: str, branch: str) -> None:
    validate_owner(owner_id)
    validate_branch(repository, branch)
    with exclusive_operation(repository):
        ensure_no_git_operation(repository)
        existing = read_lease(repository)
        if existing is not None:
            ensure_live_lease(repository, existing)
            raise Refusal(f"checkout is owned by {existing.owner_id}; release before switching")
        ensure_clean(repository)
        actual = current_branch(repository)
        if actual != branch:
            git(repository.root, "switch", "--no-guess", branch)
        if current_branch(repository) != branch:
            raise Refusal("branch switch did not publish the assigned branch")
        write_lease(repository, new_lease(repository, owner_id, branch))
        print(f"checkout switched and acquired: owner={owner_id} branch={branch}")


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--repository", default=".", help="path inside the sole YVEX checkout")
    commands = result.add_subparsers(dest="command", required=True)
    commands.add_parser("status", help="report an unowned or valid owned checkout")
    for name in ("acquire", "check", "release", "switch"):
        command = commands.add_parser(name, help=f"{name} checkout ownership")
        command.add_argument("--owner", required=True, help="unique session owner ID")
        command.add_argument("--branch", required=True, help="branch assigned to this session")
    return result


def main() -> int:
    arguments = parser().parse_args()
    try:
        repository = discover(arguments.repository)
        if arguments.command == "status":
            status_command(repository)
        elif arguments.command == "acquire":
            acquire(repository, arguments.owner, arguments.branch)
        elif arguments.command == "check":
            check(repository, arguments.owner, arguments.branch)
        elif arguments.command == "release":
            release(repository, arguments.owner, arguments.branch)
        elif arguments.command == "switch":
            switch(repository, arguments.owner, arguments.branch)
        else:
            raise Refusal("unknown checkout guard command")
    except Refusal as error:
        print(f"checkout guard refusal: {error}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
