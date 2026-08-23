
"""Explorer agent: Claude tool-use loop over local repo filesystem tools."""
 
from __future__ import annotations
 
import json
from typing import Any
 
import anthropic
from anthropic.types import MessageParam, ToolUnionParam
 
from tools import filesystem as fs
 
# Explicit caps — adjust here
MAX_ITERATIONS = 25
MAX_TOTAL_TOKENS = 120_000
MODEL = "claude-haiku-4-5-20251001"
MAX_TOKENS_PER_RESPONSE = 4096
 
TOOLS: list[ToolUnionParam] = [
    {
        "name": "read_file_tree",
        "description": "List the directory structure of the repository (files and folders).",
        "input_schema": {
            "type": "object",
            "properties": {},
            "additionalProperties": False,
        },
    },
    {
        "name": "read_file",
        "description": "Read the full text of a file path relative to the repository root.",
        "input_schema": {
            "type": "object",
            "properties": {
                "filename": {
                    "type": "string",
                    "description": "Path relative to repo root, e.g. src/main.py",
                }
            },
            "required": ["filename"],
            "additionalProperties": False,
        },
    },
    {
        "name": "search_codebase",
        "description": "Search repository file contents for a literal substring (grep-style).",
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Substring to search for (case-sensitive).",
                }
            },
            "required": ["query"],
            "additionalProperties": False,
        },
    },
    {
        "name": "done_exploring",
        "description": "Call when exploration is complete. Provide a concise summary of findings.",
        "input_schema": {
            "type": "object",
            "properties": {
                "summary": {
                    "type": "string",
                    "description": "Final exploration summary for the coding task.",
                }
            },
            "required": ["summary"],
            "additionalProperties": False,
        },
    },
]
 
SYSTEM_PROMPT = """You are a codebase explorer preparing context for a coding agent.
 
Use the tools to inspect the repository: start with the file tree, search for relevant symbols,
read key files, and build an accurate picture of how to implement the task.
 
Always read the FULL content of any file you expect the coding agent will need to edit —
the coding agent will only see files you explicitly read, not just your summary of them.
 
When you have enough context, call done_exploring with a structured summary that includes:
- Relevant files and their roles
- Important functions, classes, or patterns
- Concrete recommendations for where and how to make changes
 
Do not guess file contents — read or search before claiming what is in the code."""
 
 
def _log(title: str, payload: Any) -> None:
    print(f"\n=== {title} ===")
    if isinstance(payload, str):
        print(payload)
    else:
        print(json.dumps(payload, indent=2, default=str))
 
 
def _execute_tool(repo_path: str, name: str, tool_input: dict[str, Any]) -> str:
    if name == "read_file_tree":
        return fs.read_file_tree(repo_path)
    if name == "read_file":
        return fs.read_file(repo_path, tool_input["filename"])
    if name == "search_codebase":
        return fs.search_codebase(repo_path, tool_input["query"])
    if name == "done_exploring":
        return tool_input.get("summary", "")
    raise ValueError(f"Unknown tool: {name}")
 
 
def explore(
    task: str,
    repo_path: str,
    *,
    max_iterations: int = MAX_ITERATIONS,
    max_total_tokens: int = MAX_TOTAL_TOKENS,
    client: anthropic.Anthropic | None = None,
) -> tuple[str, dict[str, str]]:
    """
    Run the explorer loop until done_exploring, iteration cap, or token cap.
 
    Returns (summary, files_read) where files_read maps every file path
    that was actually read during exploration to its exact content. This
    is what downstream agents (e.g. the coder) use as ground truth — they
    have no access to anything the explorer saw beyond what's in these
    two return values.
    """
    client = client or anthropic.Anthropic()
    messages: list[MessageParam] = [
        {
            "role": "user",
            "content": (
                f"Repository path: {repo_path}\n\n"
                f"Task:\n{task}\n\n"
                "Explore the codebase and finish with done_exploring."
            ),
        }
    ]
 
    total_tokens = 0
    summary = ""
    finished = False
    files_read: dict[str, str] = {}
 
    for iteration in range(1, max_iterations + 1):
        print(f"\n--- Explorer iteration {iteration}/{max_iterations} ---")
 
        response = client.messages.create(
            model=MODEL,
            max_tokens=MAX_TOKENS_PER_RESPONSE,
            system=SYSTEM_PROMPT,
            tools=TOOLS,
            messages=messages,
        )
 
        if response.usage:
            step = response.usage.input_tokens + response.usage.output_tokens
            total_tokens += step
            print(
                f"Tokens this step: {step} "
                f"(running total: {total_tokens}/{max_total_tokens})"
            )
 
        if total_tokens >= max_total_tokens:
            summary = (
                summary
                or "Stopped: total token budget exceeded before done_exploring."
            )
            print("\nStopping: token cap reached.")
            break
 
        assistant_content = response.content
        messages.append({"role": "assistant", "content": assistant_content})
 
        tool_uses = [block for block in assistant_content if block.type == "tool_use"]
        if not tool_uses:
            text_blocks = [
                block.text for block in assistant_content if block.type == "text"
            ]
            summary = "\n".join(text_blocks).strip() or "Agent ended without tool use."
            print("\nStopping: model returned end_turn without done_exploring.")
            break
 
        tool_result_blocks: list[dict[str, Any]] = []
        for tool_use in tool_uses:
            _log(f"Tool call: {tool_use.name}", tool_use.input)
            if tool_use.name == "done_exploring":
                summary = str(tool_use.input.get("summary", "")).strip()
                finished = True
                result = summary or "(empty summary)"
            else:
                try:
                    result = _execute_tool(repo_path, tool_use.name, tool_use.input)
                    if tool_use.name == "read_file" and not result.startswith("Error:"):
                        files_read[tool_use.input["filename"]] = result
                except Exception as exc:
                    result = f"Error: {exc}"
 
            _log(f"Tool result: {tool_use.name}", result)
            tool_result_blocks.append(
                {
                    "type": "tool_result",
                    "tool_use_id": tool_use.id,
                    "content": result,
                }
            )
 
        messages.append({"role": "user", "content": tool_result_blocks})
 
        if finished:
            print("\nStopping: done_exploring called.")
            break
    else:
        if not summary:
            summary = f"Stopped: reached max iterations ({max_iterations})."
 
    return summary, files_read