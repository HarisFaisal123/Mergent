"""Phase 1 entry point: run the explorer against a local repo."""

from __future__ import annotations

from pathlib import Path

from dotenv import load_dotenv

from agents.explorer import explore

# Hardcoded Phase 1 test inputs — change these for your local test repo
TASK = (
    "Understand the backend layout of this project. "
    "Identify the main entry point, how dependencies are declared, "
    "and where a future coding agent would add new Python modules."
)
REPO_PATH = Path(__file__).resolve().parent.parent


def main() -> None:
    load_dotenv(Path(__file__).resolve().parent / ".env")

    print("PR Agent — Phase 1 Explorer")
    print(f"Repo: {REPO_PATH}")
    print(f"Task: {TASK}\n")

    summary = explore(TASK, str(REPO_PATH))

    print("\n" + "=" * 60)
    print("FINAL EXPLORATION SUMMARY")
    print("=" * 60)
    print(summary)


if __name__ == "__main__":
    main()
