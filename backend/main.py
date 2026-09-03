"""Entry point: self-heal pipeline — task in, passing diff out.

Runs explore -> code -> apply -> test -> fix against a local repo, entirely
from the command line: python main.py "<task>" /path/to/repo
"""

from __future__ import annotations

import argparse
from pathlib import Path

from dotenv import load_dotenv

from agents.healer import MAX_HEAL_RETRIES, self_heal

DEFAULT_TASK = (
    "Add a docstring to the read_file_tree function in backend/tools/filesystem.py "
    "explaining what it does and documenting the repo_path parameter."
)
DEFAULT_REPO_PATH = Path(__file__).resolve().parent.parent


def main() -> None:
    load_dotenv(Path(__file__).resolve().parent / ".env")

    parser = argparse.ArgumentParser(
        description="Self-heal pipeline: generate a diff, run tests, fix failures, "
        "stop once tests pass (or the coder gives up, or retries run out)."
    )
    parser.add_argument(
        "task", nargs="?", default=DEFAULT_TASK,
        help="Natural-language description of the change to make.",
    )
    parser.add_argument(
        "repo_path", nargs="?", default=str(DEFAULT_REPO_PATH),
        help="Path to the target repository.",
    )
    parser.add_argument(
        "--max-retries", type=int, default=MAX_HEAL_RETRIES,
        help=f"Max test-and-fix cycles (default {MAX_HEAL_RETRIES}).",
    )
    args = parser.parse_args()

    repo_path = str(Path(args.repo_path).expanduser().resolve())

    print("PR Agent — Self-Heal Pipeline")
    print(f"Repo: {repo_path}")
    print(f"Task: {args.task}\n")

    result = self_heal(args.task, repo_path, max_retries=args.max_retries)

    print("\n" + "=" * 60)
    print("PIPELINE RESULT")
    print("=" * 60)
    print(f"Success: {result.success}")
    print(f"Attempts: {result.attempts}")

    if result.success:
        print("\nFinal passing diff:\n")
        print(result.diff)
    else:
        if result.gave_up:
            print(f"\nCoder judged this unfixable by editing code: {result.reason}")
        else:
            print(f"\nFailed: {result.reason}")
        print("\nLast test report:\n")
        print(result.last_report)

    raise SystemExit(0 if result.success else 1)


if __name__ == "__main__":
    main()
