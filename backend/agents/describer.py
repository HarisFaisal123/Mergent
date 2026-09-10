"""Describer agent: turn a verified diff into a PR title and description.

Uses the same "### MARKER" response shape as coder.py rather than asking
for JSON — one parsing convention across the agents, and a marker block
survives a stray prose preamble that would break json.loads.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

import anthropic
from anthropic.types import MessageParam

MODEL = "claude-haiku-4-5-20251001"
MAX_TOKENS_PER_RESPONSE = 2048
MAX_TITLE_CHARS = 72

# A large refactor's diff can dwarf the useful signal; the model only needs
# enough to characterize the change, not to reproduce it.
MAX_DIFF_CHARS = 12_000

SYSTEM_PROMPT = """You write pull request descriptions for changes produced by an
automated coding agent. You are given the original task, the unified diff that was
produced, and the sandbox test report.

Respond in exactly this format and nothing else:

### TITLE
<a single imperative line, at most 72 characters, no trailing period>
### BODY
<markdown description>

The body should cover, briefly:
- What changed and why, in one short paragraph.
- A bulleted list of the notable edits, by file.
- What the sandbox test run reported.

Describe only what the diff actually does. Do not invent motivation, issue
numbers, or follow-up work that is not evident from the task and the diff."""

RESPONSE_PATTERN = re.compile(
    r"###\s*TITLE\s*\n(?P<title>.*?)\n###\s*BODY\s*\n(?P<body>.*)",
    re.DOTALL,
)

ATTRIBUTION = (
    "\n\n---\n*Opened automatically by the Mergent self-heal pipeline "
    "(explore → code → apply → test → fix).*"
)


@dataclass
class PRDescription:
    title: str
    body: str


def _truncate(text: str, limit: int) -> str:
    text = text.strip()
    return text if len(text) <= limit else text[:limit] + "\n...(diff truncated)..."


def _fallback(task: str) -> PRDescription:
    """Used when the model's response can't be parsed — a PR with a plain
    description still beats failing the run after tests have already passed."""
    title = task.strip().splitlines()[0]
    if len(title) > MAX_TITLE_CHARS:
        title = title[: MAX_TITLE_CHARS - 3].rstrip() + "..."
    return PRDescription(title=title, body=f"**Task:** {task.strip()}")


def describe(
    task: str,
    diff: str,
    report: str,
    *,
    verified: bool = True,
    client: anthropic.Anthropic | None = None,
) -> PRDescription:
    """Generate a PR title and body. Never raises on a bad response — falls
    back to the task text, since the diff is already built by this point."""
    client = client or anthropic.Anthropic()

    messages: list[MessageParam] = [
        {
            "role": "user",
            "content": (
                f"Task:\n{task}\n\n"
                f"Unified diff:\n{_truncate(diff, MAX_DIFF_CHARS)}\n\n"
                f"Sandbox test report:\n{report or '(no report)'}"
            ),
        }
    ]

    response = client.messages.create(
        model=MODEL,
        max_tokens=MAX_TOKENS_PER_RESPONSE,
        system=SYSTEM_PROMPT,
        messages=messages,
    )
    if response.stop_reason == "max_tokens":
        print("WARNING: PR description truncated — hit max_tokens.")

    text = "\n".join(b.text for b in response.content if b.type == "text").strip()

    m = RESPONSE_PATTERN.search(text)
    if not m:
        print("WARNING: could not parse PR description; using task text instead.")
        described = _fallback(task)
    else:
        title = m.group("title").strip().lstrip("#").strip().rstrip(".")
        if len(title) > MAX_TITLE_CHARS:
            title = title[: MAX_TITLE_CHARS - 3].rstrip() + "..."
        described = PRDescription(title=title or _fallback(task).title, body=m.group("body").strip())

    if not verified:
        described.body = (
            "> **Tests did not pass in the sandbox.** The coder judged the failure "
            "environmental rather than a defect in this change, so it is opened as a "
            "draft for human review.\n\n"
            f"```\n{report.strip()}\n```\n\n" + described.body
        )

    described.body += ATTRIBUTION
    return described
