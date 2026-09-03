"""Unit tests for detect_projects function in backend/tools/sandbox.py."""

from __future__ import annotations

from pathlib import Path

import pytest

from tools.sandbox import ProjectSpec, detect_projects


class TestDetectProjects:
    """Test suite for detect_projects function."""

    def test_detect_projects_python_requirements_only(self, tmp_path: Path) -> None:
        """Test detection of a repo with only requirements.txt at root."""
        (tmp_path / "requirements.txt").write_text("pytest==7.0.0\n")

        projects = detect_projects(tmp_path)

        assert len(projects) == 1
        assert projects[0].language == "python"
        assert projects[0].subdir == "."
        assert projects[0].image == "python:3.12-slim"
        assert "pytest" in projects[0].install_cmd
        assert projects[0].test_cmd == "pytest -q --tb=short"
        assert projects[0].needs_postgres is False

    def test_detect_projects_python_pyproject_only(self, tmp_path: Path) -> None:
        """Test detection of a repo with only pyproject.toml at root."""
        (tmp_path / "pyproject.toml").write_text("[build-system]\nrequires = []\n")

        projects = detect_projects(tmp_path)

        assert len(projects) == 1
        assert projects[0].language == "python"
        assert projects[0].subdir == "."
        assert projects[0].image == "python:3.12-slim"
        assert "pytest" in projects[0].install_cmd
        assert projects[0].test_cmd == "pytest -q --tb=short"
        assert projects[0].needs_postgres is False

    def test_detect_projects_node_only(self, tmp_path: Path) -> None:
        """Test detection of a repo with only package.json at root."""
        (tmp_path / "package.json").write_text('{"name": "test", "scripts": {"test": "jest"}}\n')

        projects = detect_projects(tmp_path)

        assert len(projects) == 1
        assert projects[0].language == "node"
        assert projects[0].subdir == "."
        assert projects[0].image == "node:20-slim"
        assert projects[0].install_cmd == "npm install"
        assert projects[0].test_cmd == "npm test --silent"
        assert projects[0].needs_postgres is False

    def test_detect_projects_django_manage_py(self, tmp_path: Path) -> None:
        """Test detection of Django project with manage.py (prefers manage.py over requirements.txt)."""
        (tmp_path / "manage.py").write_text("#!/usr/bin/env python\n")
        (tmp_path / "requirements.txt").write_text("django==4.0.0\n")

        projects = detect_projects(tmp_path)

        assert len(projects) == 1
        assert projects[0].language == "python"
        assert projects[0].subdir == "."
        assert projects[0].image == "python:3.12-slim"
        assert projects[0].test_cmd == "python manage.py test"
        assert projects[0].needs_postgres is False

    def test_detect_projects_separate_subdirectories(self, tmp_path: Path) -> None:
        """Test detection of Python and Node projects in separate subdirectories."""
        backend_dir = tmp_path / "backend"
        backend_dir.mkdir()
        (backend_dir / "requirements.txt").write_text("flask==2.0.0\n")

        frontend_dir = tmp_path / "frontend"
        frontend_dir.mkdir()
        (frontend_dir / "package.json").write_text('{"name": "frontend"}\n')

        projects = detect_projects(tmp_path)

        assert len(projects) == 2
        
        # Projects should be sorted by subdir
        python_project = next(p for p in projects if p.language == "python")
        node_project = next(p for p in projects if p.language == "node")

        assert python_project.subdir == "backend"
        assert python_project.test_cmd == "pytest -q --tb=short"

        assert node_project.subdir == "frontend"
        assert node_project.test_cmd == "npm test --silent"

    def test_detect_projects_both_python_subdirs(self, tmp_path: Path) -> None:
        """Test detection of two Python projects in separate subdirectories."""
        backend_dir = tmp_path / "backend"
        backend_dir.mkdir()
        (backend_dir / "pyproject.toml").write_text("[build-system]\n")

        frontend_dir = tmp_path / "frontend"
        frontend_dir.mkdir()
        (frontend_dir / "requirements.txt").write_text("requests==2.28.0\n")

        projects = detect_projects(tmp_path)

        assert len(projects) == 2
        assert all(p.language == "python" for p in projects)
        
        subdirs = {p.subdir for p in projects}
        assert subdirs == {"backend", "frontend"}

    def test_detect_projects_skips_vendor_dirs(self, tmp_path: Path) -> None:
        """Test that vendor directories (node_modules, .venv, etc.) are skipped."""
        # Root level package.json should be detected
        (tmp_path / "package.json").write_text('{"name": "root"}\n')

        # node_modules package.json should be skipped
        node_modules_dir = tmp_path / "node_modules"
        node_modules_dir.mkdir()
        (node_modules_dir / "package.json").write_text('{"name": "module"}\n')

        # .venv requirements.txt should be skipped
        venv_dir = tmp_path / ".venv"
        venv_dir.mkdir()
        (venv_dir / "requirements.txt").write_text("pip==23.0.0\n")

        projects = detect_projects(tmp_path)

        assert len(projects) == 1
        assert projects[0].subdir == "."
        assert projects[0].language == "node"

    def test_detect_projects_python_with_postgres_driver(self, tmp_path: Path) -> None:
        """Test that needs_postgres=True when psycopg driver is in requirements."""
        (tmp_path / "requirements.txt").write_text("django==4.0.0\npsycopg2==2.9.0\n")

        projects = detect_projects(tmp_path)

        assert len(projects) == 1
        assert projects[0].needs_postgres is True

    def test_detect_projects_python_with_asyncpg_driver(self, tmp_path: Path) -> None:
        """Test that needs_postgres=True when asyncpg driver is in requirements."""
        (tmp_path / "requirements.txt").write_text("fastapi==0.100.0\nasyncpg==0.27.0\n")

        projects = detect_projects(tmp_path)

        assert len(projects) == 1
        assert projects[0].needs_postgres is True

    def test_detect_projects_python_without_postgres_driver(self, tmp_path: Path) -> None:
        """Test that needs_postgres=False when no Postgres driver is in requirements."""
        (tmp_path / "requirements.txt").write_text("pytest==7.0.0\nrequests==2.28.0\n")

        projects = detect_projects(tmp_path)

        assert len(projects) == 1
        assert projects[0].needs_postgres is False

    def test_detect_projects_django_with_postgres(self, tmp_path: Path) -> None:
        """Test Django project with Postgres driver."""
        (tmp_path / "manage.py").write_text("#!/usr/bin/env python\n")
        (tmp_path / "requirements.txt").write_text("django==4.0.0\npsycopg2-binary==2.9.0\n")

        projects = detect_projects(tmp_path)

        assert len(projects) == 1
        assert projects[0].language == "python"
        assert projects[0].test_cmd == "python manage.py test"
        assert projects[0].needs_postgres is True

    def test_detect_projects_empty_repo(self, tmp_path: Path) -> None:
        """Test that an empty repo returns an empty list."""
        projects = detect_projects(tmp_path)

        assert projects == []

    def test_detect_projects_pyproject_with_postgres_marker(self, tmp_path: Path) -> None:
        """Test that needs_postgres=True when psycopg is mentioned in pyproject.toml."""
        (tmp_path / "pyproject.toml").write_text(
            "[project]\n"
            "dependencies = [\n"
            '    "psycopg[binary]>=3.0",\n'
            "]\n"
        )

        projects = detect_projects(tmp_path)

        assert len(projects) == 1
        assert projects[0].needs_postgres is True

    def test_detect_projects_manifest_priority(self, tmp_path: Path) -> None:
        """Test that manifest detection follows the correct priority order."""
        # Create all four manifest types in the same directory
        (tmp_path / "manage.py").write_text("#!/usr/bin/env python\n")
        (tmp_path / "requirements.txt").write_text("django==4.0.0\n")
        (tmp_path / "pyproject.toml").write_text("[build-system]\n")
        (tmp_path / "package.json").write_text('{"name": "test"}\n')

        projects = detect_projects(tmp_path)

        # Should only detect manage.py project (highest priority)
        assert len(projects) == 1
        assert projects[0].test_cmd == "python manage.py test"

    def test_detect_projects_requirements_priority_over_pyproject(self, tmp_path: Path) -> None:
        """Test that requirements.txt has priority over pyproject.toml."""
        (tmp_path / "requirements.txt").write_text("pytest==7.0.0\n")
        (tmp_path / "pyproject.toml").write_text("[build-system]\n")
        (tmp_path / "package.json").write_text('{"name": "test"}\n')

        projects = detect_projects(tmp_path)

        # Should only detect requirements.txt project
        assert len(projects) == 1
        assert "pip install" in projects[0].install_cmd
        assert "pytest" in projects[0].install_cmd

    def test_detect_projects_pyproject_priority_over_package_json(self, tmp_path: Path) -> None:
        """Test that pyproject.toml has priority over package.json."""
        (tmp_path / "pyproject.toml").write_text("[build-system]\n")
        (tmp_path / "package.json").write_text('{"name": "test"}\n')

        projects = detect_projects(tmp_path)

        # Should only detect pyproject.toml project
        assert len(projects) == 1
        assert projects[0].language == "python"

    def test_detect_projects_multiple_subdirs_mixed_manifests(self, tmp_path: Path) -> None:
        """Test detection with multiple subdirs containing different manifest combinations."""
        # Root: Python only
        (tmp_path / "requirements.txt").write_text("pytest==7.0.0\n")

        # backend: Django project
        backend_dir = tmp_path / "backend"
        backend_dir.mkdir()
        (backend_dir / "manage.py").write_text("#!/usr/bin/env python\n")
        (backend_dir / "requirements.txt").write_text("django==4.0.0\n")

        # frontend: Node project
        frontend_dir = tmp_path / "frontend"
        frontend_dir.mkdir()
        (frontend_dir / "package.json").write_text('{"name": "frontend"}\n')

        projects = detect_projects(tmp_path)

        assert len(projects) == 3

        root_project = next(p for p in projects if p.subdir == ".")
        assert root_project.language == "python"
        assert "pytest" in root_project.test_cmd

        backend_project = next(p for p in projects if p.subdir == "backend")
        assert backend_project.language == "python"
        assert backend_project.test_cmd == "python manage.py test"

        frontend_project = next(p for p in projects if p.subdir == "frontend")
        assert frontend_project.language == "node"

    def test_detect_projects_skips_build_and_cache_dirs(self, tmp_path: Path) -> None:
        """Test that build, dist, and cache directories are properly skipped."""
        # Root level project
        (tmp_path / "requirements.txt").write_text("pytest==7.0.0\n")

        # These should all be skipped
        skip_dirs = ["build", "dist", "__pycache__", ".pytest_cache", ".mypy_cache", ".git"]
        for skip_dir in skip_dirs:
            skip_path = tmp_path / skip_dir
            skip_path.mkdir()
            (skip_path / "requirements.txt").write_text("should-skip==1.0.0\n")

        projects = detect_projects(tmp_path)

        # Should only find the root project
        assert len(projects) == 1
        assert projects[0].subdir == "."

    def test_detect_projects_node_project_no_postgres_support(self, tmp_path: Path) -> None:
        """Test that Node projects never have needs_postgres set to True."""
        (tmp_path / "package.json").write_text('{"name": "test"}\n')

        projects = detect_projects(tmp_path)

        assert len(projects) == 1
        assert projects[0].language == "node"
        # Node projects should not have needs_postgres attribute set to True
        # (it defaults to False in the dataclass)
        assert projects[0].needs_postgres is False

    def test_detect_projects_django_in_subdirectory(self, tmp_path: Path) -> None:
        """Test detection of Django project (manage.py) in a subdirectory."""
        backend_dir = tmp_path / "backend"
        backend_dir.mkdir()
        (backend_dir / "manage.py").write_text("#!/usr/bin/env python\n")
        (backend_dir / "requirements.txt").write_text("django==4.0.0\npsycopg2==2.9.0\n")

        projects = detect_projects(tmp_path)

        assert len(projects) == 1
        assert projects[0].language == "python"
        assert projects[0].subdir == "backend"
        assert projects[0].test_cmd == "python manage.py test"
        assert projects[0].needs_postgres is True

    def test_detect_projects_manage_py_priority_in_subdirectory(self, tmp_path: Path) -> None:
        """Test that manage.py takes priority over requirements.txt in same subdirectory."""
        backend_dir = tmp_path / "backend"
        backend_dir.mkdir()
        (backend_dir / "manage.py").write_text("#!/usr/bin/env python\n")
        (backend_dir / "requirements.txt").write_text("django==4.0.0\n")
        (backend_dir / "pyproject.toml").write_text("[build-system]\n")
        (backend_dir / "package.json").write_text('{"name": "backend"}\n')

        projects = detect_projects(tmp_path)

        assert len(projects) == 1
        assert projects[0].subdir == "backend"
        assert projects[0].test_cmd == "python manage.py test"

    def test_detect_projects_python_manifest_case_insensitivity_in_requirements(self, tmp_path: Path) -> None:
        """Test that Postgres driver detection in requirements is case-insensitive."""
        (tmp_path / "requirements.txt").write_text("Django==4.0.0\nPSYCOPG2==2.9.0\n")

        projects = detect_projects(tmp_path)

        assert len(projects) == 1
        assert projects[0].needs_postgres is True

    def test_detect_projects_multiple_projects_all_detected(self, tmp_path: Path) -> None:
        """Test that all projects are detected even with many subdirectories."""
        # Root: Django
        (tmp_path / "manage.py").write_text("#!/usr/bin/env python\n")
        (tmp_path / "requirements.txt").write_text("django==4.0.0\n")

        # backend: Python with pyproject.toml
        backend_dir = tmp_path / "backend"
        backend_dir.mkdir()
        (backend_dir / "pyproject.toml").write_text("[build-system]\n")

        # frontend: Node
        frontend_dir = tmp_path / "frontend"
        frontend_dir.mkdir()
        (frontend_dir / "package.json").write_text('{"name": "frontend"}\n')

        # api: Python with requirements.txt
        api_dir = tmp_path / "api"
        api_dir.mkdir()
        (api_dir / "requirements.txt").write_text("fastapi==0.100.0\n")

        projects = detect_projects(tmp_path)

        assert len(projects) == 4
        
        subdirs = {p.subdir for p in projects}
        assert subdirs == {".", "backend", "frontend", "api"}

        languages = {(p.subdir, p.language) for p in projects}
        assert (".", "python") in languages  # Django
        assert ("backend", "python") in languages
        assert ("frontend", "node") in languages
        assert ("api", "python") in languages