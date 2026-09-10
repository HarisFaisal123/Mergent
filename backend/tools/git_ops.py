"""Local git operations for turning an applied diff into a pushed branch.

Uses subprocess rather than GitPython to match tools/apply.py, which
already shells out for checkout/ls-files — one git mechanism in the
codebase, not two.

The self-heal loop leaves its candidate applied but UNCOMMITTED on the
working tree (see agents/healer.py), so the flow here is: note the branch
we started on, cut a new one (uncommitted changes ride along), stage only
the files the agent touched, commit, push, then return to where we were.
"""

from __future__ import annotations

import re
import subprocess
import uuid
from pathlib import Path

BRANCH_PREFIX = "mergent"
MAX_SLUG_CHARS = 40

# https://github.com/owner/repo(.git) and git@github.com:owner/repo(.git)
_REMOTE_PATTERN = re.compile(
    r"(?:https://[^/]*github\.com/|git@github\.com:)(?P<owner>[^/]+)/(?P<repo>[^/]+?)(?:\.git)?$"
)


class GitError(RuntimeError):
    """A git command failed. Message is safe to print — never contains a token."""


def _run(repo_path: str, args: list[str], *, redact: str | None = None) -> str:
    result = subprocess.run(
        ["git", *args], cwd=repo_path, capture_output=True, text=True
    )
    if result.returncode != 0:
        detail = (result.stderr or result.stdout).strip()
        if redact:
            detail = detail.replace(redact, "***")
        raise GitError(f"git {args[0]} failed: {detail}")
    return result.stdout.strip()


def slugify(text: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")
    return slug[:MAX_SLUG_CHARS].rstrip("-") or "change"


def branch_name(task: str) -> str:
    """Unique per run — reruns of the same task must not collide on the remote."""
    return f"{BRANCH_PREFIX}/{slugify(task)}-{uuid.uuid4().hex[:6]}"


def parse_remote(repo_path: str, remote: str = "origin") -> tuple[str, str]:
    """Return (owner, repo) for a GitHub remote, in either URL form."""
    url = _run(repo_path, ["remote", "get-url", remote])
    m = _REMOTE_PATTERN.search(url)
    if not m:
        raise GitError(f"Remote '{remote}' is not a recognizable GitHub URL: {url}")
    return m.group("owner"), m.group("repo")


def current_branch(repo_path: str) -> str:
    return _run(repo_path, ["rev-parse", "--abbrev-ref", "HEAD"])


def detect_base_branch(repo_path: str) -> str:
    """The branch a PR should target — capture this BEFORE cutting a new one."""
    return current_branch(repo_path)


def create_branch(repo_path: str, name: str) -> None:
    """Cut and switch to `name`, carrying uncommitted changes across.

    Safe here because the agent's edits are modifications to the working
    tree with no counterpart on the new branch, so nothing conflicts.
    """
    _run(repo_path, ["checkout", "-b", name])


def commit(repo_path: str, paths: list[str], message: str) -> str:
    """Stage exactly `paths` and commit. Returns the new commit sha.

    Deliberately not `git add -A`: the repo under test may hold __pycache__
    or sandbox residue, and a PR should contain the agent's edits only.
    """
    if not paths:
        raise GitError("Nothing to commit — no changed files.")
    _run(repo_path, ["add", "--", *paths])

    staged = _run(repo_path, ["diff", "--cached", "--name-only"])
    if not staged:
        raise GitError("Nothing to commit — staged tree is empty.")

    _run(repo_path, ["commit", "-m", message])
    return _run(repo_path, ["rev-parse", "HEAD"])


def push(repo_path: str, branch: str, token: str, owner: str, repo: str) -> None:
    """Push `branch` to origin using a single-use authenticated URL.

    The token goes on the command line rather than through
    `git remote set-url`, which would persist it into .git/config. Any
    git error is redacted before it propagates, since this pipeline prints
    freely and the URL embeds the credential.
    """
    url = f"https://x-access-token:{token}@github.com/{owner}/{repo}.git"
    _run(repo_path, ["push", url, f"HEAD:refs/heads/{branch}"], redact=token)


def restore_branch(repo_path: str, branch: str) -> None:
    _run(repo_path, ["checkout", branch])


def is_same_repo(repo_path: str, other: Path) -> bool:
    """True if repo_path resolves to the same directory as `other`."""
    return Path(repo_path).expanduser().resolve() == other.expanduser().resolve()
