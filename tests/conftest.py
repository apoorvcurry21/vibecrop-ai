"""Test-wide defaults."""
import pytest


@pytest.fixture(autouse=True)
def _no_real_relevance_calls(monkeypatch):
    """The clip relevance check calls Gemini; tests must never do that unless they switch it on and mock the client."""
    monkeypatch.setenv("RELEVANCE_CHECK", "0")
