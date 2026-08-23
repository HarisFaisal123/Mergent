"""Phase 1 entry point: run the explorer against a local repo."""

from __future__ import annotations

from pathlib import Path

from dotenv import load_dotenv

from agents.explorer import explore
from agents.coder import generate_diff
# Hardcoded Phase 1 test inputs — change these for your local test repo
TASK = (
    "Add a type hint for the return value of read_file_tree in tools/filesystem.py, "
    "and add a short docstring describing its parameters."
)
REPO_PATH = Path(__file__).resolve().parent.parent




def main() -> None:
    
    load_dotenv(Path(__file__).resolve().parent / ".env")

    print("PR Agent — Phase 1 Explorer")
    print(f"Repo: {REPO_PATH}")
    print(f"Task: {TASK}\n")

    summary, files_read = explore(TASK, str(REPO_PATH))
    success, diff = generate_diff(TASK, summary, files_read, str(REPO_PATH))
    print("\n" + "=" * 60)
    print("FINAL EXPLORATION SUMMARY")
    print("=" * 60)
    print(summary)
    if success:
        print("Diff generated successfully:")
        print(diff)
    else:
        print("Diff generation failed")


if __name__ == "__main__":
    main()
