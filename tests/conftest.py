import pytest
import os


@pytest.fixture()
def n_kern_files(monkeypatch):
    monkeypatch.setenv("N_KERN_FILES", os.environ.get("N_KERN_FILES", "10"))
