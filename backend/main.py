"""Entry point: self-heal pipeline — task in, passing diff out.

Runs explore -> code -> apply -> test -> fix against a local repo, entirely
from the command line: python main.py "<task>" /path/to/repo
"""

from __future__ import annotations

import argparse
from pathlib import Path

from dotenv import load_dotenv

from agents.describer import describe
from agents.healer import HealResult, MAX_HEAL_RETRIES, self_heal
from tools import git_ops
from tools import github as gh

DEFAULT_TASK = (
    "Add unit tests for detect_projects in backend/tools/sandbox.py, covering: a repo "
    "with only a Python manifest, only a Node manifest, both in separate subdirectories, "
    "and a manage.py-based Django project."
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
    parser.add_argument(
        "--pr", action="store_true",
        help="Push the resulting change to a branch and open a GitHub pull request.",
    )
    args = parser.parse_args()

    repo_path = str(Path(args.repo_path).expanduser().resolve())

    # Refuse to PR against Mergent itself. repo_path defaults to this repo,
    # so a bare `--pr` run would otherwise have the agent open a pull request
    # on its own source rather than on the repo under test.
    if args.pr and git_ops.is_same_repo(repo_path, DEFAULT_REPO_PATH):
        parser.error(
            "--pr requires an explicit repo_path: refusing to open a pull request "
            "against Mergent's own repository."
        )

    # Validate everything the PR step needs BEFORE the pipeline runs. These
    # checks are free, and the run they precede costs several minutes of model
    # calls and container builds — failing afterward would waste all of it.
    if args.pr:
        try:
            gh.get_token()
            owner, repo = git_ops.parse_remote(repo_path)
            print(f"PR target: {owner}/{repo}\n")
        except (gh.GitHubError, git_ops.GitError) as exc:
            parser.error(str(exc))

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

        # Print the candidate even on failure — a run that ends on NO_FIX or
        # exhausted retries still produced real work, and reverting it off
        # disk is not a reason to make it unrecoverable.
        if result.diff:
            print("\nLast candidate diff:\n")
            print(result.diff)
            if result.applied:
                print("\nThese changes are still applied to the working tree.")
            else:
                print("\nThe working tree was reverted; the diff above is unapplied.")

    if args.pr:
        open_pull_request(args.task, repo_path, result)

    raise SystemExit(0 if result.success else 1)


def open_pull_request(task: str, repo_path: str, result: HealResult) -> None:
    """Branch, commit, push, and open a PR for whatever is applied on disk.

    Runs only when the candidate is still on the working tree. A run that
    passed opens a normal PR; a run that ended on NO_FIX opens a draft,
    since the change is unverified but the coder judged the sandbox — not
    the diff — to be at fault.
    """
    if not result.applied:
        print("\nSkipping PR: the working tree was reverted, nothing to push.")
        return

    print("\n" + "=" * 60)
    print("OPENING PULL REQUEST")
    print("=" * 60)

    branch = None
    base = None
    try:
        token = gh.get_token()
        owner, repo = git_ops.parse_remote(repo_path)
        base = git_ops.detect_base_branch(repo_path)

        description = describe(
            task, result.diff, result.last_report, verified=result.success
        )

        branch = git_ops.branch_name(task)
        git_ops.create_branch(repo_path, branch)
        sha = git_ops.commit(repo_path, result.changed_paths, description.title)
        print(f"Committed {sha[:8]} on {branch} ({len(result.changed_paths)} file(s))")

        git_ops.push(repo_path, branch, token, owner, repo)
        print(f"Pushed {branch} to {owner}/{repo}")

        url = gh.open_pr(
            token, owner, repo,
            head=branch, base=base,
            title=description.title, body=description.body,
            draft=not result.success,
        )
        print(f"\nPull request{' (draft)' if not result.success else ''}: {url}")

    except (git_ops.GitError, gh.GitHubError) as exc:
        print(f"\nPR step failed: {exc}")
        # A push that succeeded before a later failure leaves a live remote
        # branch; name it so the PR can be opened by hand.
        if branch:
            print(f"Branch involved: {branch}")
    finally:
        if base and git_ops.current_branch(repo_path) != base:
            git_ops.restore_branch(repo_path, base)
            print(f"Returned to {base}")


if __name__ == "__main__":
    main()
