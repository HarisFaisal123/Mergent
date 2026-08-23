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
MODEL = "claude-sonnet-4-6"
MAX_TOKENS_PER_RESPONSE = 4096

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
  functions, or content that was not shown to you."""

BLOCK_PATTERN = re.compile(
    r"### FILE:\s*(?P<filename>\S+)\s*\n"
    r"<<<<<<< SEARCH\n"
    r"(?P<search>.*?)\n"
    r"=======\n"
    r"(?P<replace>.*?)\n"
    r">>>>>>> REPLACE",
    re.DOTALL,
)


@dataclass
class EditBlock:
    filename: str
    search: str
    replace: str


def _log(title: str, payload: Any) -> None:
    print(f"\n=== {title} ===")
    if isinstance(payload, str):
        print(payload)
    else:
        print(json.dumps(payload, indent=2, default=str))


def _call_model(messages: list[MessageParam], client: anthropic.Anthropic) -> str:
    response = client.messages.create(
        model=MODEL,
        max_tokens=MAX_TOKENS_PER_RESPONSE,
        system=SYSTEM_PROMPT,
        messages=messages,
    )

    if response.usage:
        print(
            f"Tokens this step: {response.usage.input_tokens + response.usage.output_tokens}"
        )

    text_blocks = [block.text for block in response.content if block.type == "text"]
    return "\n".join(text_blocks).strip()


def _parse_blocks(response_text: str) -> list[EditBlock]:
    return [
        EditBlock(filename=m.group("filename"), search=m.group("search"), replace=m.group("replace"))
        for m in BLOCK_PATTERN.finditer(response_text)
    ]


def _validate_and_apply(
    blocks: list[EditBlock], files_read: dict[str, str]
) -> tuple[bool, str, dict[str, str]]:
    """
    Check every SEARCH block matches file content exactly and unambiguously,
    then apply it in memory. Returns (success, error_message, new_file_contents).

    new_file_contents maps filename -> updated content, only for files that
    were actually touched.
    """
    if not blocks:
        return False, "No SEARCH/REPLACE blocks found in response.", {}

    working_content = dict(files_read)  # copy — apply blocks against this

    for block in blocks:
        if block.filename not in working_content:
            return False, (
                f"File '{block.filename}' was not provided in context. "
                f"Available files: {list(files_read.keys())}"
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

    changed = {
        fname: working_content[fname]
        for fname in {b.filename for b in blocks}
    }
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
        old_content = original[filename]
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
) -> tuple[bool, str]:
    """
    Generate a unified diff for the task.

    Claude proposes SEARCH/REPLACE edits; Python applies and validates them
    against files_read (exact content captured during exploration), then
    builds the final unified diff text via difflib — never from Claude's
    own hunk-header arithmetic.

    files_read must contain the exact current content of every file Claude
    might need to edit (e.g. returned by explore()).

    Returns (success, diff_text_or_error).
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

        response_text = _call_model(messages, client)
        _log("Generated edit blocks", response_text)

        blocks = _parse_blocks(response_text)
        valid, error, changed = _validate_and_apply(blocks, files_read)
        _log("Validation result", f"valid={valid}" + ("" if valid else f" error={error}"))

        if valid:
            diff_text = _build_unified_diffs(files_read, changed)
            _log("Final unified diff", diff_text)
            print("\nStopping: valid edits produced, unified diff built.")
            return True, diff_text

        messages.append({"role": "assistant", "content": response_text})
        messages.append(
            {
                "role": "user",
                "content": (
                    f"That response had a problem:\n{error}\n\n"
                    "Return corrected SEARCH/REPLACE blocks. Output ONLY the "
                    "blocks, no explanation, no markdown fences."
                ),
            }
        )

    print(f"\nStopping: max retries ({max_retries}) exhausted without valid edits.")
    return False, response_text or "No edits produced."