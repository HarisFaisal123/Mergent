"""Phase 1 entry point: run the explorer + coder against a local repo, then apply."""

from __future__ import annotations

from pathlib import Path

from dotenv import load_dotenv

from agents.explorer import explore
from agents.coder import generate_diff
from tools.apply import write_files, verify_with_git

# Hardcoded Phase 1 test inputs — change these for your local test repo
TASK = (
    "Add a docstring to the read_file_tree function in backend/tools/filesystem.py "
    "explaining what it does and documenting the repo_path parameter."
)
REPO_PATH = Path(__file__).resolve().parent.parent


def main() -> None:
    load_dotenv(Path(__file__).resolve().parent / ".env")

    print("PR Agent — Phase 1 Explorer + Coder + Apply")
    print(f"Repo: {REPO_PATH}")
    print(f"Task: {TASK}\n")

    summary, files_read = explore(TASK, str(REPO_PATH))

    print("\n" + "=" * 60)
    print("FINAL EXPLORATION SUMMARY")
    print("=" * 60)
    print(summary)
    print(f"\nFiles read during exploration: {list(files_read.keys())}")

    success, diff, changed_files = generate_diff(TASK, summary, files_read, str(REPO_PATH))

    print("\n" + "=" * 60)
    print("DIFF GENERATION RESULT")
    print("=" * 60)
    if not success:
        print("Diff generation failed:")
        print(diff)
        return

    print("Diff generated successfully:\n")
    print(diff)

    print("\n" + "=" * 60)
    print("APPLY STEP")
    print("=" * 60)
    written = write_files(str(REPO_PATH), changed_files)
    print(f"\nFiles written: {written}")

    print("\n=== git diff --stat ===")
    print(verify_with_git(str(REPO_PATH)))


if __name__ == "__main__":
    main()