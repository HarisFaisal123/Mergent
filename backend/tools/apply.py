"""Write generated file changes to disk and verify with git."""

from __future__ import annotations

import subprocess
from pathlib import Path


def write_files(repo_path: str, updated_files: dict[str, str]) -> list[str]:
    """Write each updated file's content to disk. Returns list of paths written."""
    written = []
    for rel_path, content in updated_files.items():
        full_path = Path(repo_path) / rel_path
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


def revert(repo_path: str, paths: list[str]) -> None:
    """Discard changes to specific files — your reset hook between test runs."""
    subprocess.run(["git", "checkout", "--"] + paths, cwd=repo_path)