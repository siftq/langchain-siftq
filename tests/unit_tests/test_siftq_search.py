import os
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from pydantic import SecretStr

from langchain_siftq._utilities import (
    SIFTQ_DEFAULT_API_KEY,
    SiftqAPIWrapper,
    _check_response,
    _parse_error,
)
from langchain_siftq.siftq_search import SiftqSearch


@pytest.fixture(autouse=True)
def set_env():
    if "SIFTQ_API_KEY" not in os.environ:
        os.environ["SIFTQ_API_KEY"] = "test-key"
    yield


def test_tool_name():
    tool = SiftqSearch()
    assert tool.name == "siftq_search"


def test_tool_description():
    tool = SiftqSearch()
    assert "search" in tool.description.lower()
    assert "web pages" in tool.description.lower()


@patch("langchain_siftq.siftq_search.SiftqAPIWrapper.raw_results")
def test_siftq_search_success(mock_raw_results):
    mock_raw_results.return_value = {
        "credits": 100,
        "total": 1,
        "webpages": [
            {
                "title": "Test Result",
                "link": "https://example.com",
                "snippet": "Test snippet",
                "score": 0.95,
            }
        ],
    }
    tool = SiftqSearch()

    result = tool.invoke({"query": "test query"})

    assert result["total"] == 1
    assert result["webpages"][0]["title"] == "Test Result"
    mock_raw_results.assert_called_once_with(
        q="test query",
        scope=None,
        include_summary=None,
        include_raw_content=None,
        concise_snippet=None,
        size=None,
    )


@patch("langchain_siftq.siftq_search.SiftqAPIWrapper.raw_results")
def test_siftq_search_empty_results(mock_raw_results):
    mock_raw_results.return_value = {"credits": 100, "total": 0}
    tool = SiftqSearch()

    result = tool.invoke({"query": "nonexistent query"})

    assert isinstance(result, str)
    assert "no search results" in result.lower()


@patch("langchain_siftq.siftq_search.SiftqAPIWrapper.raw_results")
def test_siftq_search_with_scope(mock_raw_results):
    mock_raw_results.return_value = {
        "credits": 100,
        "total": 1,
        "scholars": [
            {
                "title": "Paper Title",
                "authors": ["Author A"],
                "link": "https://example.com/paper",
                "snippet": "Abstract snippet",
                "year": 2024,
            }
        ],
    }
    tool = SiftqSearch(scope="scholar")

    result = tool.invoke({"query": "machine learning"})

    assert result["scholars"][0]["title"] == "Paper Title"
    mock_raw_results.assert_called_once_with(
        q="machine learning",
        scope="scholar",
        include_summary=None,
        include_raw_content=None,
        concise_snippet=None,
        size=None,
    )


@patch("langchain_siftq.siftq_search.SiftqAPIWrapper.raw_results")
def test_siftq_search_with_raw_content(mock_raw_results):
    mock_raw_results.return_value = {
        "credits": 100,
        "total": 1,
        "webpages": [
            {
                "title": "Page 1",
                "link": "https://example.com/1",
                "snippet": "Snippet 1",
                "content": "Full raw content",
            },
        ],
    }
    tool = SiftqSearch(include_raw_content=True)

    result = tool.invoke({"query": "test"})

    assert result["webpages"][0]["content"] == "Full raw content"
    mock_raw_results.assert_called_once_with(
        q="test",
        scope=None,
        include_summary=None,
        include_raw_content=True,
        concise_snippet=None,
        size=None,
    )


@patch("langchain_siftq.siftq_search.SiftqAPIWrapper.raw_results")
def test_siftq_search_forbidden_param(mock_raw_results):
    tool = SiftqSearch()

    result = tool.invoke({"query": "test", "size": 10})

    assert isinstance(result, str)
    assert "size" in result.lower()


@pytest.mark.asyncio
@patch("langchain_siftq.siftq_search.SiftqAPIWrapper.raw_results_async")
async def test_siftq_search_async_success(mock_raw_results_async):
    mock_raw_results_async.return_value = {
        "credits": 100,
        "total": 1,
        "webpages": [
            {
                "title": "Async Result",
                "link": "https://example.com",
                "snippet": "Async snippet",
            }
        ],
    }
    tool = SiftqSearch()

    result = await tool.ainvoke({"query": "async test"})

    assert result["webpages"][0]["title"] == "Async Result"
    mock_raw_results_async.assert_called_once_with(
        q="async test",
        scope=None,
        include_summary=None,
        include_raw_content=None,
        concise_snippet=None,
        size=None,
    )


@pytest.mark.asyncio
@patch("langchain_siftq.siftq_search.SiftqAPIWrapper.raw_results_async")
async def test_siftq_search_async_empty(mock_raw_results_async):
    mock_raw_results_async.return_value = {"credits": 100, "total": 0}
    tool = SiftqSearch()

    result = await tool.ainvoke({"query": "nothing"})

    assert isinstance(result, str)
    assert "no search results" in result.lower()


def test_siftq_api_key_from_kwargs():
    tool = SiftqSearch(siftq_api_key="custom-key")
    assert tool.api_wrapper.siftq_api_key.get_secret_value() == "custom-key"


def test_siftq_search_api_base_url():
    tool = SiftqSearch(
        siftq_api_key="test-key",
        api_base_url="https://custom.api.com",
    )
    assert tool.api_wrapper.api_base_url == "https://custom.api.com"


def test_parse_error_code_2005():
    err = _parse_error(401, {"code": 2005, "message": "Invalid API key"})
    assert "2005" in str(err)
    assert "API key rejected" in str(err)


def test_parse_error_code_3003():
    err = _parse_error(429, {"code": 3003, "message": "Daily limit exceeded"})
    assert "3003" in str(err)
    assert "Daily search limit reached" in str(err)


def test_parse_error_unknown_code():
    err = _parse_error(500, {"code": 9999, "message": "Internal error"})
    assert "9999" in str(err)
    assert "Internal error" in str(err)


def test_parse_error_non_dict_body():
    err = _parse_error(500, "server error")
    assert "HTTP 500" in str(err)


@patch("langchain_siftq.siftq_search.SiftqAPIWrapper.raw_results")
def test_siftq_search_api_error_2005(mock_raw_results):
    mock_raw_results.side_effect = ValueError("Error 2005: API key rejected")
    tool = SiftqSearch()
    result = tool.invoke({"query": "test"})
    assert isinstance(result, str)
    assert "API key rejected" in result


@patch("langchain_siftq.siftq_search.SiftqAPIWrapper.raw_results")
def test_siftq_search_api_error_3003(mock_raw_results):
    mock_raw_results.side_effect = ValueError("Error 3003: Daily search limit reached")
    tool = SiftqSearch()
    result = tool.invoke({"query": "test"})
    assert isinstance(result, str)
    assert "Daily search limit" in result


@patch.dict(os.environ, {}, clear=True)
def test_default_api_key_used_when_no_key_provided():
    tool = SiftqSearch()
    assert tool.api_wrapper.siftq_api_key.get_secret_value() == SIFTQ_DEFAULT_API_KEY


def test_kwarg_overrides_default_key():
    tool = SiftqSearch(siftq_api_key="custom-key")
    assert tool.api_wrapper.siftq_api_key.get_secret_value() == "custom-key"


@patch.dict(os.environ, {"SIFTQ_API_KEY": "env-key"}, clear=True)
def test_env_var_overrides_default_key():
    tool = SiftqSearch()
    assert tool.api_wrapper.siftq_api_key.get_secret_value() == "env-key"


# --- _check_response tests ---


def test_check_response_passes_on_success():
    _check_response({"webpages": []}, 200)


def test_check_response_raises_on_non_200():
    with pytest.raises(ValueError, match="403"):
        _check_response({"message": "Forbidden"}, 403)


def test_check_response_raises_on_error_code_in_body():
    with pytest.raises(ValueError, match="API key rejected"):
        _check_response({"code": 2005, "message": "bad key"}, 200)


def test_check_response_raises_on_unknown_code():
    with pytest.raises(ValueError, match="1234"):
        _check_response({"code": 1234, "message": "info"}, 200)


# --- Sync raw_results: non-JSON response ---


def test_raw_results_non_json_response():
    wrapper = SiftqAPIWrapper(siftq_api_key=SecretStr("test-key"))
    mock_resp = MagicMock()
    mock_resp.status_code = 502
    mock_resp.json.side_effect = ValueError("no json")
    with patch.object(wrapper._http, "post", return_value=mock_resp):
        with pytest.raises(ValueError, match="Non-JSON response"):
            wrapper.raw_results(q="test")


# --- Async raw_results_async: non-200 with non-JSON body ---


@pytest.mark.asyncio
async def test_raw_results_async_non_200_non_json():
    wrapper = SiftqAPIWrapper(siftq_api_key=SecretStr("test-key"))
    mock_res = MagicMock()
    mock_res.status = 502
    mock_res.text = AsyncMock(return_value="<html>Bad Gateway</html>")

    @asynccontextmanager
    async def mock_post(*args: object, **kwargs: object) -> AsyncGenerator[MagicMock, None]:
        yield mock_res

    mock_session = MagicMock()
    mock_session.post = mock_post
    mock_session.__aenter__ = AsyncMock(return_value=mock_session)
    mock_session.__aexit__ = AsyncMock(return_value=False)

    with patch("langchain_siftq._utilities.aiohttp.ClientSession", return_value=mock_session):
        with pytest.raises(ValueError, match="HTTP 502"):
            await wrapper.raw_results_async(q="test")


# --- Async raw_results_async: 200 with invalid JSON body ---


@pytest.mark.asyncio
async def test_raw_results_async_200_invalid_json():
    wrapper = SiftqAPIWrapper(siftq_api_key=SecretStr("test-key"))
    mock_res = MagicMock()
    mock_res.status = 200
    mock_res.text = AsyncMock(return_value="not valid json{")

    @asynccontextmanager
    async def mock_post(*args: object, **kwargs: object) -> AsyncGenerator[MagicMock, None]:
        yield mock_res

    mock_session = MagicMock()
    mock_session.post = mock_post
    mock_session.__aenter__ = AsyncMock(return_value=mock_session)
    mock_session.__aexit__ = AsyncMock(return_value=False)

    with patch("langchain_siftq._utilities.aiohttp.ClientSession", return_value=mock_session):
        with pytest.raises(ValueError, match="Invalid JSON"):
            await wrapper.raw_results_async(q="test")
