"""Integration tests for SiftqSearch tool (requires SIFTQ_API_KEY env var)."""

import os

import pytest

from langchain_siftq import SiftqSearch


@pytest.mark.skipif(
    not os.environ.get("SIFTQ_API_KEY"),
    reason="SIFTQ_API_KEY environment variable not set",
)
def test_siftq_search_integration():
    tool = SiftqSearch()
    result = tool.invoke({"query": "LangChain framework"})
    assert "webpages" in result or "documents" in result or "scholars" in result
    assert result.get("total", 0) > 0 or len(result.get("webpages", [])) > 0


@pytest.mark.skipif(
    not os.environ.get("SIFTQ_API_KEY"),
    reason="SIFTQ_API_KEY environment variable not set",
)
def test_siftq_search_scholar_scope():
    tool = SiftqSearch(scope="scholar")
    result = tool.invoke({"query": "transformer neural networks"})
    assert "scholars" in result
    if result.get("total", 0) > 0:
        assert len(result["scholars"]) > 0


@pytest.mark.skipif(
    not os.environ.get("SIFTQ_API_KEY"),
    reason="SIFTQ_API_KEY environment variable not set",
)
def test_siftq_search_with_summary():
    tool = SiftqSearch(include_summary=True)
    result = tool.invoke({"query": "Python programming"})
    assert result.get("total", 0) >= 0


@pytest.mark.skipif(
    not os.environ.get("SIFTQ_API_KEY"),
    reason="SIFTQ_API_KEY environment variable not set",
)
def test_siftq_search_size_param():
    tool = SiftqSearch(size=5)
    result = tool.invoke({"query": "machine learning"})
    total_results = sum(
        len(result.get(k, []))
        for k in ["webpages", "documents", "scholars", "images", "videos", "podcasts"]
    )
    assert total_results <= 5 or result.get("total", 0) <= 5
