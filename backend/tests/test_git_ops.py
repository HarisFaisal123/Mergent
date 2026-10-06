"""Unit tests for slugify and branch_name functions in backend/tools/git_ops.py."""

from __future__ import annotations

import re

import pytest

from tools.git_ops import branch_name, slugify


class TestSlugify:
    """Test suite for slugify function."""

    def test_slugify_task_with_spaces_and_punctuation(self) -> None:
        """Test conversion of a task string with spaces and punctuation."""
        result = slugify("Fix: User login issue")
        
        assert result == "fix-user-login-issue"
        assert all(c.isalnum() or c == "-" for c in result)
        assert not result.startswith("-")
        assert not result.endswith("-")

    def test_slugify_only_non_alphanumeric_characters(self) -> None:
        """Test conversion of a string containing only non-alphanumeric characters."""
        result = slugify("!@#$%^&*()")
        
        # Should return the fallback value "change"
        assert result == "change"

    def test_slugify_long_string_truncation(self) -> None:
        """Test truncation of a string exceeding MAX_SLUG_CHARS (40 characters)."""
        # Create a string longer than 40 characters that will have hyphens after truncation
        long_string = "this-is-a-very-long-task-description-that-exceeds-forty-characters"
        result = slugify(long_string)
        
        assert len(result) <= 40
        assert not result.endswith("-")
        assert all(c.isalnum() or c == "-" for c in result)

    def test_slugify_lowercase_conversion(self) -> None:
        """Test that input text is converted to lowercase."""
        result = slugify("UPPERCASE Task")
        
        assert result == "uppercase-task"
        assert result.islower()

    def test_slugify_multiple_consecutive_special_chars(self) -> None:
        """Test that consecutive non-alphanumeric characters are collapsed to single hyphen."""
        result = slugify("task:::with:::colons")
        
        assert result == "task-with-colons"
        assert "--" not in result

    def test_slugify_leading_trailing_special_chars(self) -> None:
        """Test that leading and trailing special characters are stripped."""
        result = slugify("!!!task description!!!")
        
        assert result == "task-description"
        assert not result.startswith("-")
        assert not result.endswith("-")

    def test_slugify_empty_string(self) -> None:
        """Test that an empty string returns the fallback value."""
        result = slugify("")
        
        assert result == "change"

    def test_slugify_only_spaces(self) -> None:
        """Test that a string of only spaces returns the fallback value."""
        result = slugify("     ")
        
        assert result == "change"


class TestBranchName:
    """Test suite for branch_name function."""

    def test_branch_name_has_prefix(self) -> None:
        """Test that branch_name is prefixed with 'mergent/'."""
        result = branch_name("Test task")
        
        assert result.startswith("mergent/")

    def test_branch_name_contains_slugified_task(self) -> None:
        """Test that branch_name contains the slugified task."""
        task = "Fix: User login issue"
        result = branch_name(task)
        
        # Extract the part between "mergent/" and the UUID suffix
        parts = result.split("-")
        # The UUID suffix is the last part (6 hex characters)
        uuid_part = parts[-1]
        assert len(uuid_part) == 6
        assert all(c in "0123456789abcdef" for c in uuid_part)
        
        # Verify the slugified task is in there
        slugified_task = slugify(task)
        assert result.startswith(f"mergent/{slugified_task}-")

    def test_branch_name_uniqueness_across_calls(self) -> None:
        """Test that two calls with the same task produce different branch names."""
        task = "Same task description"
        
        result1 = branch_name(task)
        result2 = branch_name(task)
        
        # Both should start with the same prefix and slugified task
        assert result1.split("-")[0] == result2.split("-")[0]  # Same "mergent/slugified-part"
        
        # But the UUID suffixes should be different
        uuid1 = result1.split("-")[-1]
        uuid2 = result2.split("-")[-1]
        assert uuid1 != uuid2

    def test_branch_name_format(self) -> None:
        """Test the overall format of branch_name output."""
        result = branch_name("Sample task")
        
        # Should match pattern: mergent/{slug}-{6-char-hex-uuid}
        pattern = r"^mergent/[a-z0-9-]+-[0-9a-f]{6}$"
        assert re.match(pattern, result), f"Branch name {result} does not match expected pattern"

    def test_branch_name_with_punctuation_task(self) -> None:
        """Test branch_name with a task containing various punctuation."""
        task = "Fix: Critical bug!!! (priority: HIGH)"
        result = branch_name(task)
        
        assert result.startswith("mergent/")
        # Should not contain spaces or special characters (except hyphens and the UUID suffix)
        main_part = result.split("-")[0]  # Get "mergent/slugified-stuff"
        assert " " not in main_part
        assert "!" not in main_part
        assert "(" not in main_part
        assert ")" not in main_part