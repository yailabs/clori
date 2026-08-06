#!/usr/bin/env python3
"""Exercise the explicit Git-local lease for one shared YVEX checkout."""

from __future__ import annotations

import json
import pathlib
import subprocess
import tempfile


ROOT = pathlib.Path(__file__).resolve().parents[1]
TOOL = ROOT / "tools/checkout_guard.py"
MINIMAX_BRANCH = "feature/minimax-h3"
DEEPSEEK_BRANCH = "feature/deepseek-v4-flash"


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def git(repository: pathlib.Path, *arguments: str) -> str:
    result = subprocess.run(
        ["git", *arguments],
        cwd=repository,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    require(result.returncode == 0, result.stderr or result.stdout)
    return result.stdout.strip()


def invoke(repository: pathlib.Path, command: str, owner: str | None = None,
           branch: str | None = None) -> subprocess.CompletedProcess[str]:
    arguments = ["python3", str(TOOL), "--repository", str(repository), command]
    if owner is not None:
        arguments.extend(["--owner", owner])
    if branch is not None:
        arguments.extend(["--branch", branch])
    return subprocess.run(
        arguments,
        cwd=repository,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )


def repository_fixture(parent: pathlib.Path, name: str) -> pathlib.Path:
    repository = parent / name
    repository.mkdir()
    git(repository, "init", "--initial-branch=main")
    git(repository, "config", "user.name", "YVEX Checkout Guard Test")
    git(repository, "config", "user.email", "checkout-guard@example.invalid")
    (repository / "tracked.txt").write_text("baseline\n", encoding="utf-8")
    git(repository, "add", "tracked.txt")
    git(repository, "-c", "commit.gpgsign=false", "commit", "-m", "test: establish baseline")
    git(repository, "branch", MINIMAX_BRANCH)
    git(repository, "branch", DEEPSEEK_BRANCH)
    git(repository, "switch", MINIMAX_BRANCH)
    return repository


def expect_refusal(result: subprocess.CompletedProcess[str], text: str) -> None:
    require(result.returncode == 2, f"expected refusal, got {result.returncode}: {result.stdout}")
    require(text in result.stderr, f"missing refusal {text!r}: {result.stderr}")


def test_competing_sessions(parent: pathlib.Path) -> None:
    repository = repository_fixture(parent, "competing")
    first = invoke(repository, "acquire", "minimax-session-a", MINIMAX_BRANCH)
    require(first.returncode == 0, first.stderr)
    expect_refusal(
        invoke(repository, "acquire", "minimax-session-b", MINIMAX_BRANCH),
        "owned by another session",
    )
    expect_refusal(
        invoke(repository, "check", "minimax-session-b", MINIMAX_BRANCH),
        "owned by another session",
    )
    expect_refusal(
        invoke(repository, "release", "minimax-session-b", MINIMAX_BRANCH),
        "owned by another session",
    )
    check = invoke(repository, "check", "minimax-session-a", MINIMAX_BRANCH)
    require(check.returncode == 0 and "ownership valid" in check.stdout, check.stderr)
    release = invoke(repository, "release", "minimax-session-a", MINIMAX_BRANCH)
    require(release.returncode == 0, release.stderr)


def test_wrong_branch_and_dirty_checkout(parent: pathlib.Path) -> None:
    repository = repository_fixture(parent, "branch-dirty")
    expect_refusal(
        invoke(repository, "acquire", "deepseek-session", DEEPSEEK_BRANCH),
        "wrong branch",
    )
    (repository / "tracked.txt").write_text("dirty\n", encoding="utf-8")
    expect_refusal(
        invoke(repository, "acquire", "minimax-session", MINIMAX_BRANCH),
        "checkout is dirty",
    )
    expect_refusal(
        invoke(repository, "switch", "deepseek-session", DEEPSEEK_BRANCH),
        "checkout is dirty",
    )
    git(repository, "restore", "tracked.txt")


def test_active_git_operations(parent: pathlib.Path) -> None:
    repository = repository_fixture(parent, "git-operations")
    git_directory = pathlib.Path(git(repository, "rev-parse", "--absolute-git-dir"))
    head = git(repository, "rev-parse", "HEAD")
    (git_directory / "MERGE_HEAD").write_text(head + "\n", encoding="ascii")
    expect_refusal(
        invoke(repository, "acquire", "minimax-session", MINIMAX_BRANCH),
        "Git operation is active: MERGE_HEAD",
    )
    (git_directory / "MERGE_HEAD").unlink()
    (git_directory / "rebase-merge").mkdir()
    expect_refusal(
        invoke(repository, "acquire", "minimax-session", MINIMAX_BRANCH),
        "Git operation is active: rebase-merge",
    )
    (git_directory / "rebase-merge").rmdir()


def test_safe_release(parent: pathlib.Path) -> None:
    repository = repository_fixture(parent, "release")
    remote = parent / "release-origin.git"
    remote.mkdir()
    git(remote, "init", "--bare")
    git(repository, "remote", "add", "origin", str(remote))
    git(repository, "push", "--set-upstream", "origin", MINIMAX_BRANCH)
    acquire = invoke(repository, "acquire", "minimax-session", MINIMAX_BRANCH)
    require(acquire.returncode == 0, acquire.stderr)
    lease_path = pathlib.Path(git(repository, "rev-parse", "--absolute-git-dir")) / \
        "yvex/checkout_lease.json"
    require(lease_path.is_file(), "lease was not stored in Git-local state")
    require(not (repository / ".yvex").exists(), "lease escaped into the repository tree")
    lease = json.loads(lease_path.read_text(encoding="utf-8"))
    require(lease["owner_id"] == "minimax-session", "lease lost owner identity")
    (repository / "tracked.txt").write_text("dirty\n", encoding="utf-8")
    expect_refusal(
        invoke(repository, "release", "minimax-session", MINIMAX_BRANCH),
        "checkout is dirty",
    )
    git(repository, "restore", "tracked.txt")
    (repository / "tracked.txt").write_text("committed\n", encoding="utf-8")
    git(repository, "add", "tracked.txt")
    git(repository, "-c", "commit.gpgsign=false", "commit", "-m", "test: advance checkout")
    expect_refusal(
        invoke(repository, "release", "minimax-session", MINIMAX_BRANCH),
        "not published to its configured upstream",
    )
    git(repository, "push", "origin", MINIMAX_BRANCH)
    release = invoke(repository, "release", "minimax-session", MINIMAX_BRANCH)
    require(release.returncode == 0, release.stderr)
    require(not lease_path.exists(), "release retained the checkout lease")
    status = invoke(repository, "status")
    require(status.returncode == 0 and "unowned" in status.stdout, status.stderr)


def test_branch_transition_and_stale_refusal(parent: pathlib.Path) -> None:
    repository = repository_fixture(parent, "transition")
    transition = invoke(repository, "switch", "deepseek-session", DEEPSEEK_BRANCH)
    require(transition.returncode == 0, transition.stderr)
    require(git(repository, "branch", "--show-current") == DEEPSEEK_BRANCH,
            "guard did not switch to the assigned branch")
    expect_refusal(
        invoke(repository, "switch", "minimax-session", MINIMAX_BRANCH),
        "release before switching",
    )
    release = invoke(repository, "release", "deepseek-session", DEEPSEEK_BRANCH)
    require(release.returncode == 0, release.stderr)
    transition = invoke(repository, "switch", "minimax-session", MINIMAX_BRANCH)
    require(transition.returncode == 0, transition.stderr)
    git(repository, "switch", DEEPSEEK_BRANCH)
    expect_refusal(invoke(repository, "status"), "stale checkout lease")
    expect_refusal(
        invoke(repository, "acquire", "deepseek-session", DEEPSEEK_BRANCH),
        "stale checkout lease",
    )
    git(repository, "switch", MINIMAX_BRANCH)
    release = invoke(repository, "release", "minimax-session", MINIMAX_BRANCH)
    require(release.returncode == 0, release.stderr)


def test_policy_projection() -> None:
    agents = (ROOT / "AGENTS.md").read_text(encoding="utf-8")
    contributing = (ROOT / "CONTRIBUTING.md").read_text(encoding="utf-8")
    makefile = (ROOT / "Makefile").read_text(encoding="utf-8")
    require("### Single-checkout coordination" in agents, "AGENTS lacks canonical lease policy")
    require("tools/checkout_guard.py" in agents, "AGENTS lacks the executable guard command")
    require("dedicated worktree" not in contributing, "CONTRIBUTING retains worktree policy")
    require("test-checkout-guard" in makefile, "repository guardrails omit checkout lease tests")


def main() -> None:
    test_policy_projection()
    with tempfile.TemporaryDirectory(prefix="yvex-checkout-guard-") as temporary:
        parent = pathlib.Path(temporary)
        test_competing_sessions(parent)
        test_wrong_branch_and_dirty_checkout(parent)
        test_active_git_operations(parent)
        test_safe_release(parent)
        test_branch_transition_and_stale_refusal(parent)
    print("checkout guard: ownership, refusal, release, and transition checks passed")


if __name__ == "__main__":
    main()
