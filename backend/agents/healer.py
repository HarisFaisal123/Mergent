"""Self-heal loop: explore -> code -> apply -> test -> (fix or give up).

Each retry attempt reverts the working tree to the ORIGINAL file content
(from explore()) before asking the coder for a fix — the same "always
diff against ground truth, use conversation context to remember what went
wrong" pattern coder.py already uses for its own SEARCH/REPLACE-syntax
retries, just extended one level up to cover test failures too.

Whether to keep retrying after a test failure is the coder's call, not a
hardcoded rule here: it either proposes a corrected diff or responds with
NO_FIX (see coder.NO_FIX_INSTRUCTIONS) when it judges the failure is an
environment/infra problem no code edit could address. This loop stops
immediately on NO_FIX rather than burning remaining retries.
"""

from __future__ import annotations

from dataclasses import dataclass

import anthropic

from agents.coder import generate_diff, propose_fix
from agents.explorer import explore
from tools import apply as apply_tools
from tools import sandbox
from tools.report import summarize

MAX_HEAL_RETRIES = 3


@dataclass
class HealResult:
    success: bool
    attempts: int
    diff: str = ""
    reason: str = ""
    gave_up: bool = False
    last_report: str = ""


def self_heal(
    task: str,
    repo_path: str,
    *,
    max_retries: int = MAX_HEAL_RETRIES,
    client: anthropic.Anthropic | None = None,
) -> HealResult:
    """
    Run the full explore -> code -> apply -> test pipeline, retrying the
    coder against real test failures up to max_retries times.

    max_retries counts test attempts (each round writes a candidate diff
    to disk and runs the sandbox once), not model calls — generate_diff
    and propose_fix have their own smaller internal retries for malformed
    SEARCH/REPLACE syntax.
    """
    client = client or anthropic.Anthropic()

    print("\n" + "=" * 60)
    print("SELF-HEAL: EXPLORE")
    print("=" * 60)
    summary, files_read = explore(task, repo_path, client=client)

    ok, diff_or_error, changed_files = generate_diff(
        task, summary, files_read, repo_path, client=client
    )
    if not ok:
        return HealResult(success=False, attempts=0, reason=diff_or_error)

    diff = diff_or_error

    for attempt in range(1, max_retries + 1):
        print("\n" + "=" * 60)
        print(f"SELF-HEAL: ATTEMPT {attempt}/{max_retries} — APPLY + TEST")
        print("=" * 60)

        apply_tools.write_files(repo_path, changed_files)
        results = sandbox.run_tests(repo_path)
        report = summarize(results)
        print(f"\n{report}")

        if all(r.success for r in results):
            return HealResult(success=True, attempts=attempt, diff=diff, last_report=report)

        apply_tools.revert(repo_path, list(changed_files.keys()))

        if attempt == max_retries:
            return HealResult(
                success=False, attempts=attempt,
                reason="Max retries exhausted.", last_report=report,
            )

        print("\n--- Asking coder to fix the failure ---")
        outcome = propose_fix(task, summary, files_read, diff, report, client=client)

        if outcome.status == "no_fix":
            return HealResult(
                success=False, attempts=attempt, gave_up=True,
                reason=outcome.reason, last_report=report,
            )
        if outcome.status == "invalid":
            return HealResult(
                success=False, attempts=attempt,
                reason=f"Coder produced invalid edits: {outcome.reason}", last_report=report,
            )

        diff = outcome.diff
        changed_files = outcome.changed_files or {}

    # Unreachable: the loop above always returns by the final iteration.
    return HealResult(success=False, attempts=max_retries, reason="Max retries exhausted.")
