"""Sandbox tool: install deps + run tests for a repo inside ephemeral Docker containers.

By the time this runs, the coder's SEARCH/REPLACE edits have already been
written to disk on the host repo (tools/apply.write_files). This module
never mounts that working tree read-write — it binds it read-only and
copies it into each container's own writable layer instead. That means:

- A broken `pip install`/`npm install` can never leave stray .venv/
  node_modules artifacts on the host repo.
- Every attempt (including retries from the self-heal loop) starts from
  an identical clean copy, regardless of what a previous attempt installed.

A repo can contain more than one testable project (e.g. a `backend/` +
`frontend/` split) — detect_projects() returns every manifest it finds at
the repo root or one level of subdirectories, and run_tests() tests each
one in its own container, returning one SandboxResult per project.

A Python project whose requirements name a Postgres driver (psycopg/
asyncpg) gets a `postgres:16-alpine` sidecar container on a private Docker
network, with connection details injected as DATABASE_URL/POSTGRES_* env
vars — this is what lets Django's test runner (which needs a real database
to create its test DB against) and Postgres-backed integration tests run
at all. This assumes the project reads a DATABASE_URL-style convention;
projects wired to a different env var scheme won't pick it up.

Every container and network is always destroyed in a `finally` block,
whatever happens.
"""

from __future__ import annotations

import concurrent.futures
import shlex
import time
import uuid
from dataclasses import dataclass
from pathlib import Path

import docker
from docker.models.containers import Container
from docker.models.networks import Network

from tools.filesystem import SKIP_DIR_NAMES

DEFAULT_TIMEOUT_SECONDS = 300

POSTGRES_IMAGE = "postgres:16-alpine"
POSTGRES_ENV = {
    "POSTGRES_USER": "sandbox",
    "POSTGRES_PASSWORD": "sandbox",
    "POSTGRES_DB": "sandbox",
}
POSTGRES_READY_TIMEOUT_SECONDS = 30
POSTGRES_DRIVER_MARKERS = ("psycopg", "asyncpg")


@dataclass
class ProjectSpec:
    language: str  # "python" | "node"
    subdir: str  # "." or path relative to repo root, e.g. "backend"
    image: str
    install_cmd: str
    test_cmd: str
    needs_postgres: bool = False


@dataclass
class StepResult:
    command: str
    exit_code: int | None
    output: str
    timed_out: bool = False


@dataclass
class SandboxResult:
    success: bool
    project: ProjectSpec | None
    install: StepResult | None = None
    test: StepResult | None = None
    error: str = ""


def _needs_postgres(candidate: Path) -> bool:
    """Best-effort signal that a Python project expects a live Postgres.

    Looks for a Postgres driver package (psycopg2/psycopg/asyncpg) named in
    the manifest — a project pinned to sqlite for tests wouldn't list one.
    """
    for name in ("requirements.txt", "pyproject.toml"):
        manifest = candidate / name
        if not manifest.is_file():
            continue
        try:
            text = manifest.read_text(errors="ignore").lower()
        except OSError:
            continue
        if any(marker in text for marker in POSTGRES_DRIVER_MARKERS):
            return True
    return False


def detect_projects(repo_path: Path) -> list[ProjectSpec]:
    """Look at the repo root, then its immediate subdirectories, for manifests.

    Checking one level of subdirectories (not just the root) matters for
    layouts like this one, where the Python project lives in backend/
    rather than at the repo root. Every matching directory is returned, not
    just the first — a repo with both backend/ and frontend/ manifests
    yields one ProjectSpec each.
    """
    candidates = [repo_path] + sorted(
        p for p in repo_path.iterdir() if p.is_dir() and p.name not in SKIP_DIR_NAMES
    )

    projects: list[ProjectSpec] = []
    for candidate in candidates:
        subdir = "." if candidate == repo_path else candidate.relative_to(repo_path).as_posix()

        if (candidate / "manage.py").is_file():
            # manage.py sets DJANGO_SETTINGS_MODULE itself, so `manage.py test`
            # avoids the extra pytest-django install and settings-module wiring
            # that running pytest directly against a Django project would need.
            projects.append(ProjectSpec(
                language="python",
                subdir=subdir,
                image="python:3.12-slim",
                install_cmd="pip install --no-cache-dir -r requirements.txt",
                test_cmd="python manage.py test",
                needs_postgres=_needs_postgres(candidate),
            ))
        elif (candidate / "requirements.txt").is_file():
            projects.append(ProjectSpec(
                language="python",
                subdir=subdir,
                image="python:3.12-slim",
                install_cmd="pip install --no-cache-dir -r requirements.txt && pip install --no-cache-dir pytest",
                test_cmd="pytest -q --tb=short",
                needs_postgres=_needs_postgres(candidate),
            ))
        elif (candidate / "pyproject.toml").is_file():
            projects.append(ProjectSpec(
                language="python",
                subdir=subdir,
                image="python:3.12-slim",
                install_cmd="pip install --no-cache-dir . && pip install --no-cache-dir pytest",
                test_cmd="pytest -q --tb=short",
                needs_postgres=_needs_postgres(candidate),
            ))
        elif (candidate / "package.json").is_file():
            projects.append(ProjectSpec(
                language="node",
                subdir=subdir,
                image="node:20-slim",
                install_cmd="npm install",
                test_cmd="npm test --silent",
            ))

    return projects


def _copy_command() -> str:
    """Build a tar-pipe copy from /repo to /workspace, skipping vendor/cache dirs.

    A plain `cp -r` has no exclude option and slim base images don't ship
    rsync, but tar is present on every image that has apt (which is all of
    them) — so pipe a tar stream through itself with --exclude flags.
    """
    excludes = " ".join(f"--exclude={shlex.quote(name)}" for name in sorted(SKIP_DIR_NAMES))
    return (
        f"mkdir -p /workspace && "
        f"tar -C /repo {excludes} -cf - . | tar -C /workspace -xf -"
    )


def _exec(
    container: Container,
    command: str,
    workdir: str,
    timeout: int,
    environment: dict[str, str] | None = None,
) -> StepResult:
    """Run `command` via `sh -c` inside the container with a hard timeout.

    docker-py's exec_run takes no timeout argument, so it's run on a
    background thread. There's no way to cancel a single exec in flight,
    so on timeout we kill the whole container and report it rather than
    hang the pipeline forever.
    """
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
        future = pool.submit(
            container.exec_run, ["sh", "-c", command], workdir=workdir, environment=environment
        )
        try:
            exit_code, output = future.result(timeout=timeout)
        except concurrent.futures.TimeoutError:
            container.kill()
            return StepResult(command=command, exit_code=None, output="", timed_out=True)

    text = output.decode("utf-8", errors="replace") if isinstance(output, bytes) else str(output)
    return StepResult(command=command, exit_code=exit_code, output=text)


def _postgres_env(db_host: str) -> dict[str, str]:
    user = POSTGRES_ENV["POSTGRES_USER"]
    password = POSTGRES_ENV["POSTGRES_PASSWORD"]
    db = POSTGRES_ENV["POSTGRES_DB"]
    return {
        "DATABASE_URL": f"postgresql://{user}:{password}@{db_host}:5432/{db}",
        "POSTGRES_HOST": db_host,
        "POSTGRES_PORT": "5432",
        "POSTGRES_USER": user,
        "POSTGRES_PASSWORD": password,
        "POSTGRES_DB": db,
    }


def _wait_for_postgres(container: Container, timeout: int) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        exit_code, _ = container.exec_run(["pg_isready", "-U", POSTGRES_ENV["POSTGRES_USER"]])
        if exit_code == 0:
            return True
        time.sleep(1)
    return False


def _run_project(
    client: docker.DockerClient, repo: Path, project: ProjectSpec, timeout_seconds: int
) -> SandboxResult:
    mount_source = repo if project.subdir == "." else repo / project.subdir
    run_id = uuid.uuid4().hex[:8]

    container: Container | None = None
    db_container: Container | None = None
    network: Network | None = None

    try:
        network_name = None
        test_env: dict[str, str] | None = None

        if project.needs_postgres:
            network = client.networks.create(f"sandbox-net-{run_id}", driver="bridge")
            network_name = network.name
            db_host = f"sandbox-db-{run_id}"
            db_container = client.containers.run(
                POSTGRES_IMAGE,
                detach=True,
                name=db_host,
                network=network_name,
                environment=POSTGRES_ENV,
            )
            if not _wait_for_postgres(db_container, POSTGRES_READY_TIMEOUT_SECONDS):
                return SandboxResult(
                    success=False, project=project,
                    error="Postgres sidecar did not become ready in time.",
                )
            test_env = _postgres_env(db_host)

        container = client.containers.run(
            project.image,
            command="sleep infinity",
            detach=True,
            volumes={str(mount_source): {"bind": "/repo", "mode": "ro"}},
            network=network_name,
        )

        setup = _exec(container, _copy_command(), "/", timeout_seconds)
        if setup.timed_out or setup.exit_code != 0:
            return SandboxResult(
                success=False, project=project,
                error=f"Failed to copy repo into container:\n{setup.output}",
            )

        install = _exec(container, project.install_cmd, "/workspace", timeout_seconds, environment=test_env)
        if install.timed_out:
            return SandboxResult(
                success=False, project=project, install=install,
                error="Dependency install timed out.",
            )
        if install.exit_code != 0:
            return SandboxResult(
                success=False, project=project, install=install,
                error="Dependency install failed.",
            )

        test = _exec(container, project.test_cmd, "/workspace", timeout_seconds, environment=test_env)
        if test.timed_out:
            return SandboxResult(
                success=False, project=project, install=install, test=test,
                error="Test run timed out.",
            )

        return SandboxResult(
            success=test.exit_code == 0, project=project, install=install, test=test,
        )

    finally:
        if container is not None:
            container.remove(force=True)
        if db_container is not None:
            db_container.remove(force=True)
        if network is not None:
            network.remove()


def run_tests(
    repo_path: str,
    *,
    timeout_seconds: int = DEFAULT_TIMEOUT_SECONDS,
    client: docker.DockerClient | None = None,
) -> list[SandboxResult]:
    """
    Detect every testable project in the repo, and for each one: spin up a
    container (plus a Postgres sidecar if the project needs one), copy the
    repo in, install dependencies, run tests, capture output — then always
    destroy the container(s).

    Returns one SandboxResult per detected project. A SandboxResult has
    success=True only if tests were actually run and exited 0. Detection
    failure, install failure, and timeouts are all reported as
    success=False with details in the relevant step. If no project is
    detected at all, returns a single success=False result with no project.
    """
    repo = Path(repo_path).expanduser().resolve()
    projects = detect_projects(repo)
    if not projects:
        return [SandboxResult(
            success=False,
            project=None,
            error=(
                "No recognized manifest (manage.py, requirements.txt, pyproject.toml, "
                "package.json) found in repo root or its immediate subdirectories."
            ),
        )]

    client = client or docker.from_env()
    return [_run_project(client, repo, project, timeout_seconds) for project in projects]
