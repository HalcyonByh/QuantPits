import pytest
import os
import sys

# conftest.py can be used for globally shared fixtures

@pytest.fixture
def mock_workspace(tmp_path):
    """Fixture providing a temporary workspace directory."""
    workspace_dir = tmp_path / "MockWorkspace"
    workspace_dir.mkdir()
    return str(workspace_dir)
