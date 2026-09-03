"""Write generated file changes to disk and verify with git."""

from __future__ import annotations

import subprocess
from pathlib import Path


def write_files(repo_path: str, updated_files: dict[str, str]) -> list[str]:
    """Write each updated file's content to disk. Returns list of paths written."""
    written = []
    for rel_path, content in updated_files.items():
        full_path = Path(repo_path) / rel_path
        full_path.parent.mkdir(parents=True, exist_ok=True)
        full_path.write_text(content, encoding="utf-8")
        written.append(rel_path)
        print(f"Wrote: {rel_path}")
    return written


def verify_with_git(repo_path: str) -> str:
    """Return `git diff` output so you can visually confirm what changed."""
    result = subprocess.run(
        ["git", "diff", "--stat"],
        cwd=repo_path, capture_output=True, text=True,
    )
    return result.stdout


def _is_tracked(repo_path: str, path: str) -> bool:
    result = subprocess.run(
        ["git", "ls-files", "--error-unmatch", path],
        cwd=repo_path, capture_output=True, text=True,
    )
    return result.returncode == 0


def revert(repo_path: str, paths: list[str]) -> None:
    """Discard changes to specific files — your reset hook between test runs.

    A path the coder just created (### NEW FILE) was never committed, so
    `git checkout -- <path>` fails on it ("did not match any file(s)").
    Reverting an untracked file means deleting it — there's no committed
    version to restore to.
    """
    tracked = [p for p in paths if _is_tracked(repo_path, p)]
    untracked = [p for p in paths if p not in tracked]

    if tracked:
        subprocess.run(["git", "checkout", "--"] + tracked, cwd=repo_path)
    for path in untracked:
        full_path = Path(repo_path) / path
        full_path.unlink(missing_ok=True)