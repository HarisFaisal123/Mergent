"""Filesystem helpers for repo exploration (no LLM calls)."""

from __future__ import annotations

import os
from pathlib import Path

SKIP_DIR_NAMES = {
    ".git",
    "__pycache__",
    ".venv",
    "venv",
    "node_modules",
    "dist",
    "build",
    ".mypy_cache",
    ".pytest_cache",
}

MAX_SEARCH_MATCHES = 150
MAX_FILE_BYTES = 512_000


def _repo_root(repo_path: str | Path) -> Path:
    root = Path(repo_path).expanduser().resolve()
    if not root.is_dir():
        raise NotADirectoryError(f"Not a directory: {root}")
    return root

def _resolve_in_repo(repo: Path, relative: str) -> Path:
    rel = relative.lstrip("/").lstrip("\\")
    if not rel or rel == ".":
        raise ValueError("filename must be a non-empty path relative to the repo root")
    target = (repo / rel).resolve()
    try:
        target.relative_to(repo)
    except ValueError as exc:
        raise ValueError(f"Path escapes repo: {relative}") from exc
    return target


def read_file_tree(repo_path: str) -> str:
    """Return a text directory tree for repo_path (skips common vendor/cache dirs)."""
    repo = _repo_root(repo_path)
    lines: list[str] = [f"{repo.name}/"]

    def walk(dir_path: Path, prefix: str) -> None:
        entries: list[Path] = []
        try:
            for entry in sorted(dir_path.iterdir(), key=lambda p: (not p.is_dir(), p.name.lower())):
                if entry.name in SKIP_DIR_NAMES:
                    continue
                entries.append(entry)
        except OSError as exc:
            lines.append(f"{prefix}[cannot read directory: {exc}]")
            return

        for i, entry in enumerate(entries):
            is_last = i == len(entries) - 1
            branch = "└── " if is_last else "├── "
            lines.append(f"{prefix}{branch}{entry.name}{'/' if entry.is_dir() else ''}")
            if entry.is_dir():
                extension = "    " if is_last else "│   "
                walk(entry, prefix + extension)

    walk(repo, "")
    return "\n".join(lines)


def read_file(repo_path: str, filename: str) -> str:
    """Return contents of a file relative to repo_path."""
    repo = _repo_root(repo_path)
    path = _resolve_in_repo(repo, filename)
    if not path.is_file():
        raise FileNotFoundError(f"Not a file: {filename}")
    size = path.stat().st_size
    if size > MAX_FILE_BYTES:
        raise ValueError(
            f"File too large ({size} bytes, max {MAX_FILE_BYTES}): {filename}"
        )
    return path.read_text(encoding="utf-8", errors="replace")


def search_codebase(repo_path: str, query: str) -> str:
    """Simple grep-style search: path:line: content for each match."""
    repo = _repo_root(repo_path)
    if not query.strip():
        return "No matches found (empty query)."

    matches: list[str] = []
    truncated = False

    for dirpath, dirnames, filenames in os.walk(repo):
        dirnames[:] = [d for d in sorted(dirnames) if d not in SKIP_DIR_NAMES]
        for name in sorted(filenames):
            full = Path(dirpath) / name
            try:
                rel = full.relative_to(repo).as_posix()
            except ValueError:
                continue
            if full.stat().st_size > MAX_FILE_BYTES:
                continue
            try:
                text = full.read_text(encoding="utf-8", errors="replace")
            except OSError:
                continue
            if "\x00" in text:
                continue
            for line_no, line in enumerate(text.splitlines(), start=1):
                if query in line:
                    matches.append(f"{rel}:{line_no}: {line.rstrip()}")
                    if len(matches) >= MAX_SEARCH_MATCHES:
                        truncated = True
                        break
            if truncated:
                break
        if truncated:
            break

    if not matches:
        return "No matches found."
    body = "\n".join(matches)
    if truncated:
        body += f"\n... (truncated after {MAX_SEARCH_MATCHES} matches)"
    return body
