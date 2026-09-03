"""Coder agent: generates a unified diff for a task, with retry loop.

Claude does NOT write diff syntax or hunk headers by hand — that's the
part LLMs are unreliable at (line-count arithmetic), and it's what caused
"Hunk is shorter than expected" failures even on trivial single-hunk cases.

Instead: Claude proposes edits as SEARCH/REPLACE blocks (exact text in,
exact text out — no counting involved). Python applies each block against
the real file content and uses difflib.unified_diff() to generate the
actual diff text. difflib computes line numbers mechanically, so hunk
headers are always correct by construction — there is no failure mode
where the header disagrees with the body.

The final output is still a standard unified diff, applicable with
`git apply`, matching the original plan.
"""

from __future__ import annotations

import difflib
import json
import re
from dataclasses import dataclass
from typing import Any

import anthropic
from anthropic.types import MessageParam

MAX_RETRIES = 3
MODEL = "claude-haiku-4-5-20251001"
MAX_TOKENS_PER_RESPONSE = 8192

SYSTEM_PROMPT = """You are a coding agent that proposes file edits to accomplish a task.

Output edits using this exact format, one block per change:

### FILE: path/relative/to/repo/root
<<<<<<< SEARCH
exact existing code, copied verbatim including whitespace
=======
the replacement code
>>>>>>> REPLACE

Rules:
- The SEARCH section must match the existing file content EXACTLY —
  same whitespace, same indentation, same line breaks. Copy it verbatim
  from the file content shown to you below. Do not paraphrase, reformat,
  or reindent it.
- Keep SEARCH blocks as small as possible while still being unique in the file.
- You may output multiple ### FILE / SEARCH / REPLACE blocks, including
  multiple blocks for the same file.
- Do NOT write diff syntax, @@ hunk headers, or line numbers anywhere —
  only the SEARCH/REPLACE format above. Line numbers will be computed
  automatically from your edits.
- No explanation, no commentary, no markdown code fences around the blocks.
- Base changes strictly on the file content provided — do not invent files,
  functions, or content that was not shown to you.

To create a file that does NOT already exist (e.g. a new test file), use
this format instead — never SEARCH/REPLACE with an empty SEARCH section:

### NEW FILE: path/relative/to/repo/root
<<<<<<< NEW
full content of the new file
>>>>>>> NEW

Only use ### NEW FILE for a path that doesn't already exist. To edit a
file that exists, always use the SEARCH/REPLACE format above instead."""

NO_FIX_INSTRUCTIONS = """

You may instead be shown a test failure from a previous attempt of yours,
with the diff that caused it and the test output. In that case, propose
corrected SEARCH/REPLACE edits the same way as above — against the
ORIGINAL file content shown to you, not against your previous diff.

If, and only if, the failure is not something an edit to this repository's
code could fix at all — an environment/infrastructure problem such as a
network failure fetching packages, a missing system library, or a sandbox
misconfiguration — respond instead with exactly:

### NO_FIX
<one or two sentence explanation of why no code change could fix this>

A dependency merely missing from requirements.txt/pyproject.toml/package.json
is NOT a NO_FIX case — adding it is a valid code fix. Only give up when you
are confident no code change could plausibly address the failure."""

FIX_SYSTEM_PROMPT = SYSTEM_PROMPT + NO_FIX_INSTRUCTIONS

NO_FIX_PATTERN = re.compile(r"###\s*NO_FIX\s*\n(?P<reason>.*)", re.DOTALL)

TRUNCATION_FEEDBACK = (
    "Your previous response was cut off before it finished — it hit the "
    "output length limit partway through a block, so that block has no "
    "closing marker and was discarded entirely (not just shortened). "
    "Propose fewer files/edits in this response than last time, and make "
    "sure every block you include is fully closed with its "
    ">>>>>>> REPLACE or >>>>>>> NEW marker before the response ends."
)

BLOCK_PATTERN = re.compile(
    r"### FILE:\s*(?P<filename>\S+)\s*\n"
    r"<<<<<<< SEARCH\n"
    r"(?P<search>.*?)\n"
    r"=======\n"
    r"(?P<replace>.*?)\n"
    r">>>>>>> REPLACE",
    re.DOTALL,
)

NEW_FILE_PATTERN = re.compile(
    r"### NEW FILE:\s*(?P<filename>\S+)\s*\n"
    r"<<<<<<< NEW\n"
    r"(?P<content>.*?)\n"
    r">>>>>>> NEW",
    re.DOTALL,
)


@dataclass
class EditBlock:
    filename: str
    search: str
    replace: str


@dataclass
class NewFileBlock:
    filename: str
    content: str


@dataclass
class FixOutcome:
    status: str  # "fixed" | "no_fix" | "invalid"
    diff: str = ""
    changed_files: dict[str, str] | None = None
    reason: str = ""


def _log(title: str, payload: Any) -> None:
    print(f"\n=== {title} ===")
    if isinstance(payload, str):
        print(payload)
    else:
        print(json.dumps(payload, indent=2, default=str))


def _call_model(
    messages: list[MessageParam], client: anthropic.Anthropic, *, system: str = SYSTEM_PROMPT
) -> tuple[str, str | None]:
    """Returns (response_text, stop_reason).

    stop_reason == "max_tokens" means the response was cut off before the
    model finished — any SEARCH/REPLACE or NEW FILE block still "open" at
    that point has no closing marker, so it silently fails to parse rather
    than raising an error. Callers must check this BEFORE parsing, or a
    truncated response reads as "the model just produced fewer edits"
    instead of "the model didn't finish."
    """
    response = client.messages.create(
        model=MODEL,
        max_tokens=MAX_TOKENS_PER_RESPONSE,
        system=system,
        messages=messages,
    )

    if response.usage:
        print(
            f"Tokens this step: {response.usage.input_tokens + response.usage.output_tokens}"
        )
    if response.stop_reason == "max_tokens":
        print("WARNING: response truncated — hit max_tokens before finishing.")

    text_blocks = [block.text for block in response.content if block.type == "text"]
    return "\n".join(text_blocks).strip(), response.stop_reason


def _parse_blocks(response_text: str) -> list[EditBlock]:
    return [
        EditBlock(filename=m.group("filename"), search=m.group("search"), replace=m.group("replace"))
        for m in BLOCK_PATTERN.finditer(response_text)
    ]


def _parse_no_fix(response_text: str) -> str | None:
    m = NO_FIX_PATTERN.search(response_text)
    return m.group("reason").strip() if m else None


def _parse_new_files(response_text: str) -> list[NewFileBlock]:
    return [
        NewFileBlock(filename=m.group("filename"), content=m.group("content"))
        for m in NEW_FILE_PATTERN.finditer(response_text)
    ]


def _validate_and_apply(
    blocks: list[EditBlock], new_files: list[NewFileBlock], files_read: dict[str, str]
) -> tuple[bool, str, dict[str, str]]:
    """
    Check every SEARCH block matches file content exactly and unambiguously,
    apply it in memory, and add any brand-new files. Returns (success,
    error_message, new_file_contents).

    new_file_contents maps filename -> updated content, only for files that
    were actually touched or created.
    """
    if not blocks and not new_files:
        return False, "No SEARCH/REPLACE or NEW FILE blocks found in response.", {}

    working_content = dict(files_read)  # copy — apply blocks against this
    touched: set[str] = set()

    for new_file in new_files:
        if new_file.filename in working_content:
            return False, (
                f"'{new_file.filename}' already exists — use SEARCH/REPLACE to "
                "edit it, not ### NEW FILE."
            ), {}
        working_content[new_file.filename] = new_file.content
        touched.add(new_file.filename)

    for block in blocks:
        if block.filename not in working_content:
            return False, (
                f"File '{block.filename}' was not provided in context. "
                f"Available files: {list(files_read.keys())}. If this file "
                "doesn't exist yet, use ### NEW FILE instead of SEARCH/REPLACE."
            ), {}

        content = working_content[block.filename]
        count = content.count(block.search)

        if count == 0:
            return False, (
                f"SEARCH text not found verbatim in {block.filename}. "
                "Copy the exact text from the provided file content, "
                "including whitespace and indentation."
            ), {}
        if count > 1:
            return False, (
                f"SEARCH text in {block.filename} matches {count} locations — "
                "ambiguous. Make the SEARCH block larger/more specific so it "
                "matches only one location."
            ), {}

        working_content[block.filename] = content.replace(block.search, block.replace, 1)
        touched.add(block.filename)

    changed = {fname: working_content[fname] for fname in touched}
    return True, "", changed


def _build_unified_diffs(
    original: dict[str, str], updated: dict[str, str]
) -> str:
    """
    Build real unified diffs via difflib — headers are always mechanically
    correct since difflib counts the actual lines, Claude never writes
    hunk syntax at all.
    """
    diff_parts = []
    for filename, new_content in updated.items():
        old_content = original.get(filename, "")  # "" for brand-new files
        diff_lines = difflib.unified_diff(
            old_content.splitlines(keepends=True),
            new_content.splitlines(keepends=True),
            fromfile=f"a/{filename}",
            tofile=f"b/{filename}",
        )
        diff_parts.append("".join(diff_lines))
    return "\n".join(diff_parts)


def generate_diff(
    task: str,
    exploration_summary: str,
    files_read: dict[str, str],
    repo_path: str,
    *,
    max_retries: int = MAX_RETRIES,
    client: anthropic.Anthropic | None = None,
) -> tuple[bool, str, dict[str, str]]:
    """
    Generate a unified diff for the task.

    Claude proposes SEARCH/REPLACE edits; Python applies and validates them
    against files_read (exact content captured during exploration), then
    builds the final unified diff text via difflib — never from Claude's
    own hunk-header arithmetic.

    files_read must contain the exact current content of every file Claude
    might need to edit (e.g. returned by explore()).

    Returns (success, diff_text_or_error, changed_files). changed_files
    maps filename -> full updated content, only for files actually edited —
    this is what the apply step writes to disk. Empty dict on failure.
    """
    client = client or anthropic.Anthropic()

    file_context = "\n\n".join(
        f"--- FILE: {path} ---\n{content}" for path, content in files_read.items()
    )

    messages: list[MessageParam] = [
        {
            "role": "user",
            "content": (
                f"Repository path: {repo_path}\n\n"
                f"Task:\n{task}\n\n"
                f"Exploration summary:\n{exploration_summary}\n\n"
                f"Current content of relevant files:\n{file_context}\n\n"
                "Generate the SEARCH/REPLACE edit blocks now."
            ),
        }
    ]

    response_text = ""

    for attempt in range(1, max_retries + 1):
        print(f"\n--- Coder attempt {attempt}/{max_retries} ---")

        response_text, stop_reason = _call_model(messages, client)
        _log("Generated edit blocks", response_text)

        if stop_reason == "max_tokens":
            print("\nResponse truncated — asking for fewer/smaller edits, not parsing this one.")
            messages.append({"role": "assistant", "content": response_text})
            messages.append({"role": "user", "content": TRUNCATION_FEEDBACK})
            continue

        blocks = _parse_blocks(response_text)
        new_files = _parse_new_files(response_text)
        valid, error, changed = _validate_and_apply(blocks, new_files, files_read)
        _log("Validation result", f"valid={valid}" + ("" if valid else f" error={error}"))

        if valid:
            diff_text = _build_unified_diffs(files_read, changed)
            _log("Final unified diff", diff_text)
            print("\nStopping: valid edits produced, unified diff built.")
            return True, diff_text, changed

        messages.append({"role": "assistant", "content": response_text})
        messages.append(
            {
                "role": "user",
                "content": (
                    f"That response had a problem:\n{error}\n\n"
                    "Return corrected SEARCH/REPLACE or NEW FILE blocks. "
                    "Output ONLY the blocks, no explanation, no markdown fences."
                ),
            }
        )

    print(f"\nStopping: max retries ({max_retries}) exhausted without valid edits.")
    return False, response_text or "No edits produced.", {}


def propose_fix(
    task: str,
    exploration_summary: str,
    files_read: dict[str, str],
    failed_diff: str,
    test_report: str,
    *,
    max_retries: int = MAX_RETRIES,
    client: anthropic.Anthropic | None = None,
) -> FixOutcome:
    """
    Ask the coder to fix a diff that applied cleanly but failed the sandbox
    test run. Unlike generate_diff's internal retries (which only correct
    malformed SEARCH/REPLACE syntax), this hands the model real test
    failure output and a genuine choice: propose a corrected diff, or
    respond with ### NO_FIX if it judges the failure isn't something a
    code edit could address at all (see NO_FIX_INSTRUCTIONS). The caller
    (the self-heal loop) treats "no_fix" as a reason to stop retrying,
    not as a failure to retry harder.

    Edits are validated against files_read — the ORIGINAL file content,
    not failed_diff's result — since the caller reverts the working tree
    to that original state before calling this.
    """
    client = client or anthropic.Anthropic()

    file_context = "\n\n".join(
        f"--- FILE: {path} ---\n{content}" for path, content in files_read.items()
    )

    messages: list[MessageParam] = [
        {
            "role": "user",
            "content": (
                f"Task:\n{task}\n\n"
                f"Exploration summary:\n{exploration_summary}\n\n"
                f"Current content of relevant files:\n{file_context}\n\n"
                f"You previously proposed this diff:\n{failed_diff}\n\n"
                f"It applied cleanly, but the test run failed:\n{test_report}\n\n"
                "Propose corrected SEARCH/REPLACE edit blocks against the "
                "original file content above, or respond with ### NO_FIX "
                "if this cannot be fixed by editing code."
            ),
        }
    ]

    for attempt in range(1, max_retries + 1):
        print(f"\n--- Fix attempt {attempt}/{max_retries} ---")

        response_text, stop_reason = _call_model(messages, client, system=FIX_SYSTEM_PROMPT)
        _log("Fix proposal", response_text)

        if stop_reason == "max_tokens":
            print("\nResponse truncated — asking for fewer/smaller edits, not parsing this one.")
            messages.append({"role": "assistant", "content": response_text})
            messages.append({"role": "user", "content": TRUNCATION_FEEDBACK})
            continue

        no_fix_reason = _parse_no_fix(response_text)
        if no_fix_reason is not None:
            print("\nStopping: coder judged this unfixable by editing code.")
            return FixOutcome(status="no_fix", reason=no_fix_reason)

        blocks = _parse_blocks(response_text)
        new_files = _parse_new_files(response_text)
        valid, error, changed = _validate_and_apply(blocks, new_files, files_read)
        _log("Validation result", f"valid={valid}" + ("" if valid else f" error={error}"))

        if valid:
            diff_text = _build_unified_diffs(files_read, changed)
            _log("Final unified diff", diff_text)
            print("\nStopping: valid fix produced.")
            return FixOutcome(status="fixed", diff=diff_text, changed_files=changed)

        messages.append({"role": "assistant", "content": response_text})
        messages.append(
            {
                "role": "user",
                "content": (
                    f"That response had a problem:\n{error}\n\n"
                    "Return corrected SEARCH/REPLACE or NEW FILE blocks, or "
                    "### NO_FIX. Output ONLY that, no explanation, no markdown fences."
                ),
            }
        )

    print(f"\nStopping: max retries ({max_retries}) exhausted without a valid fix.")
    return FixOutcome(status="invalid", reason="Max retries exhausted without valid edits or NO_FIX.")