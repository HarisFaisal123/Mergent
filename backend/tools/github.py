"""Open a pull request on GitHub via PyGitHub.

Deliberately thin: everything about branch/commit/push lives in
tools/git_ops.py, so this module only turns an already-pushed branch into
a PR and hands back the URL.
"""

from __future__ import annotations

import os

from github import Github, GithubException

TOKEN_ENV_VAR = "GITHUB_TOKEN"


class GitHubError(RuntimeError):
    """PR creation failed. Message is safe to print."""


def get_token() -> str:
    token = os.environ.get(TOKEN_ENV_VAR, "").strip()
    if not token:
        raise GitHubError(
            f"{TOKEN_ENV_VAR} is not set. Add it to backend/.env — a fine-grained "
            "token with Contents: read/write and Pull requests: read/write on the "
            "target repository."
        )
    return token


def open_pr(
    token: str,
    owner: str,
    repo: str,
    *,
    head: str,
    base: str,
    title: str,
    body: str,
    draft: bool = False,
) -> str:
    """Create a PR and return its URL."""
    try:
        gh = Github(token)
        target = gh.get_repo(f"{owner}/{repo}")
        pr = target.create_pull(title=title, body=body, head=head, base=base, draft=draft)
        return pr.html_url
    except GithubException as exc:
        detail = exc.data.get("message", str(exc)) if isinstance(exc.data, dict) else str(exc)
        errors = exc.data.get("errors") if isinstance(exc.data, dict) else None
        if errors:
            detail += " — " + "; ".join(
                e.get("message", str(e)) if isinstance(e, dict) else str(e) for e in errors
            )
        raise GitHubError(f"Could not open PR ({exc.status}): {detail}") from exc
