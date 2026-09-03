"""Turn raw sandbox output into a compact report the coder can act on.

A raw pytest traceback runs to thousands of characters of box-drawing
noise that has nothing to do with the actual bug. This extracts just what
the coder needs to locate and fix it: failing test node ids, their
file:line, and the assertion/exception message — parsed from `--tb=short`
output (tools/sandbox.py's pytest test_cmd), which puts each failure in a
predictable "file:line: in func" + "E   ..." shape.

Anything that doesn't match that shape (install failures, timeouts,
non-pytest runners like `manage.py test`, unparseable output) falls back
to a truncated tail of the raw output rather than silently dropping it.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from tools.sandbox import SandboxResult

_SECTION_RE = re.compile(r"^=+\s*(.+?)\s*=+$")
_DIVIDER_RE = re.compile(r"^_{3,}\s*(.+?)\s*_{3,}$")
_LOCATION_RE = re.compile(r"^(?P<file>\S+):(?P<line>\d+): in (?P<func>\S+)$")
_SUMMARY_LINE_RE = re.compile(r"^(?:FAILED|ERROR) (?P<nodeid>\S+)(?: - (?P<reason>.*))?$")

RAW_OUTPUT_FALLBACK_CHARS = 4000


@dataclass
class TestFailure:
    name: str
    nodeid: str
    location: str | None
    message: str


def _parse_pytest_failures(output: str) -> list[TestFailure]:
    lines = output.splitlines()

    # pytest's "short test summary info" section gives a reliable
    # short-name -> (full nodeid, one-line reason) lookup, since a
    # FAILURES/ERRORS divider header only has the bare function name.
    summary_by_name: dict[str, tuple[str, str]] = {}
    for line in lines:
        m = _SUMMARY_LINE_RE.match(line.strip())
        if m:
            nodeid = m.group("nodeid")
            short = nodeid.rsplit("::", 1)[-1]
            summary_by_name[short] = (nodeid, (m.group("reason") or "").strip())

    failures: list[TestFailure] = []
    in_section = False
    i = 0
    while i < len(lines):
        section = _SECTION_RE.match(lines[i].strip())
        if section:
            in_section = section.group(1).strip().upper() in ("FAILURES", "ERRORS")
            i += 1
            continue

        header = _DIVIDER_RE.match(lines[i].strip()) if in_section else None
        if header:
            test_name = header.group(1).strip()
            location: str | None = None
            message_lines: list[str] = []
            i += 1
            while i < len(lines):
                stripped = lines[i].strip()
                if _DIVIDER_RE.match(stripped) or _SECTION_RE.match(stripped):
                    break
                loc = _LOCATION_RE.match(stripped)
                if loc and location is None:
                    location = f"{loc.group('file')}:{loc.group('line')}"
                if stripped.startswith("E ") or stripped == "E":
                    message_lines.append(stripped[2:].strip())
                i += 1

            nodeid, reason = summary_by_name.get(test_name, (test_name, ""))
            failures.append(TestFailure(
                name=test_name,
                nodeid=nodeid,
                location=location,
                message="\n".join(message_lines).strip() or reason,
            ))
            continue

        i += 1

    return failures


def _format_failures(failures: list[TestFailure]) -> str:
    parts = []
    for idx, f in enumerate(failures, start=1):
        where = f" ({f.location})" if f.location else ""
        message = f"\n   {f.message}" if f.message else ""
        parts.append(f"{idx}. {f.nodeid}{where}{message}")
    return "\n".join(parts)


def _tail(text: str, limit: int = RAW_OUTPUT_FALLBACK_CHARS) -> str:
    text = text.strip()
    return text if len(text) <= limit else "...(truncated)...\n" + text[-limit:]


def summarize_result(result: SandboxResult) -> str:
    """One clean, model-readable report for a single project's sandbox run."""
    if result.project is None:
        return f"Could not run tests: {result.error}"

    label = f"[{result.project.language} project in {result.project.subdir}]"

    if result.install is not None and (result.install.timed_out or result.install.exit_code != 0):
        status = "timed out" if result.install.timed_out else f"exited {result.install.exit_code}"
        return f"{label} Dependency install failed ({status}):\n{_tail(result.install.output)}"

    if result.test is None:
        return f"{label} {result.error or 'Unknown sandbox failure.'}"

    if result.test.timed_out:
        return f"{label} Test run timed out after producing no result."

    if result.success:
        return f"{label} All tests passed."

    if result.project.language == "python":
        failures = _parse_pytest_failures(result.test.output)
        if failures:
            return f"{label} {len(failures)} test(s) failed:\n{_format_failures(failures)}"

    return f"{label} Tests failed (exit {result.test.exit_code}):\n{_tail(result.test.output)}"


def summarize(results: list[SandboxResult]) -> str:
    """Report across every project the sandbox tested — usually just one."""
    return "\n\n".join(summarize_result(r) for r in results)
