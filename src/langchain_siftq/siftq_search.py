from typing import Any

from langchain_core.callbacks import (
    AsyncCallbackManagerForToolRun,
    CallbackManagerForToolRun,
)
from langchain_core.tools import BaseTool, ToolException
from pydantic import BaseModel, ConfigDict, Field

from langchain_siftq._utilities import Scope, SiftqAPIWrapper

_FORBIDDEN_RUN_PARAMS = frozenset(["size"])
_RESULT_KEYS = ("webpages", "scholars", "documents", "images", "videos", "podcasts")


class SiftqSearchInput(BaseModel):
    model_config = ConfigDict(extra="allow")

    query: str = Field(description="Search query to look up")
    include_summary: bool | None = Field(
        default=None,
        description="Enhance recall using page summaries",
    )
    include_raw_content: bool | None = Field(
        default=None,
        description="Fetch raw content from source pages",
    )
    concise_snippet: bool | None = Field(
        default=None,
        description="Return concise snippet with exact original text match",
    )


def _generate_suggestions(params: dict[str, Any]) -> list[str]:
    suggestions = []
    scope = params.get("scope")
    include_raw_content = params.get("include_raw_content")
    concise_snippet = params.get("concise_snippet")

    if scope and scope != "webpage":
        suggestions.append(f"Try a broader scope like 'webpage' instead of '{scope}'")
    if include_raw_content:
        suggestions.append(
            "Remove include_raw_content since fetching raw content from source pages may return fewer results"
        )
    if concise_snippet:
        suggestions.append(
            "Remove concise_snippet since exact original text matching may be too strict"
        )

    return suggestions


def _validate_params(kwargs: dict[str, Any]) -> None:
    for param in _FORBIDDEN_RUN_PARAMS:
        if param in kwargs:
            raise ValueError(f"The parameter '{param}' can only be set during instantiation.")


def _build_call_kwargs(
    tool: "SiftqSearch",
    include_summary: bool | None,
    include_raw_content: bool | None,
    concise_snippet: bool | None,
    extra_kwargs: dict[str, Any],
) -> dict[str, Any]:
    return {
        "scope": tool.scope,
        "include_summary": (
            tool.include_summary if tool.include_summary is not None else include_summary
        ),
        "include_raw_content": (
            tool.include_raw_content
            if tool.include_raw_content is not None
            else include_raw_content
        ),
        "concise_snippet": (
            tool.concise_snippet if tool.concise_snippet is not None else concise_snippet
        ),
        "size": tool.size,
        **extra_kwargs,
    }


def _check_empty_results(
    raw_results: dict[str, Any],
    query: str,
    scope: Scope | None,
    include_raw_content: bool | None,
    concise_snippet: bool | None,
) -> None:
    if not any(raw_results.get(k) for k in _RESULT_KEYS):
        suggestions = _generate_suggestions(
            {
                "scope": scope,
                "include_raw_content": include_raw_content,
                "concise_snippet": concise_snippet,
            }
        )
        raise ToolException(
            f"No search results found for '{query}'. "
            f"Suggestions: {', '.join(suggestions)}. "
            f"Try modifying your search parameters."
        )


class SiftqSearch(BaseTool):
    name: str = "siftq_search"
    description: str = (
        "A search engine for web pages, documents, scholarly articles, images, "
        "videos, and podcasts. Returns structured results with titles, URLs, "
        "snippets, and optional raw page content. Supports multiple search scopes. "
        "Input should be a search query."
    )

    args_schema: type[BaseModel] = SiftqSearchInput
    handle_tool_error: bool = True

    siftq_api_key: str | None = None
    api_base_url: str | None = None
    scope: Scope | None = None
    include_summary: bool | None = None
    include_raw_content: bool | None = None
    concise_snippet: bool | None = None
    size: int | None = None

    api_wrapper: SiftqAPIWrapper = Field(
        default_factory=lambda: SiftqAPIWrapper()  # pyright: ignore[reportCallIssue]
    )

    def __init__(self, **kwargs: Any) -> None:
        if "siftq_api_key" in kwargs or "api_base_url" in kwargs:
            wrapper_kwargs = {}
            if "siftq_api_key" in kwargs:
                wrapper_kwargs["siftq_api_key"] = kwargs["siftq_api_key"]
            if "api_base_url" in kwargs:
                wrapper_kwargs["api_base_url"] = kwargs["api_base_url"]
            kwargs["api_wrapper"] = SiftqAPIWrapper(**wrapper_kwargs)

        super().__init__(**kwargs)

    def _run(
        self,
        query: str,
        include_summary: bool | None = None,
        include_raw_content: bool | None = None,
        concise_snippet: bool | None = None,
        run_manager: CallbackManagerForToolRun | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        try:
            _validate_params(kwargs)
            call_kwargs = _build_call_kwargs(
                self, include_summary, include_raw_content, concise_snippet, kwargs
            )
            call_kwargs["q"] = query
            raw_results = self.api_wrapper.raw_results(**call_kwargs)
            _check_empty_results(
                raw_results, query, self.scope, include_raw_content, concise_snippet
            )
            return raw_results
        except ToolException:
            raise
        except Exception as e:
            raise ToolException(str(e)) from e

    async def _arun(
        self,
        query: str,
        include_summary: bool | None = None,
        include_raw_content: bool | None = None,
        concise_snippet: bool | None = None,
        run_manager: AsyncCallbackManagerForToolRun | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        try:
            _validate_params(kwargs)
            call_kwargs = _build_call_kwargs(
                self, include_summary, include_raw_content, concise_snippet, kwargs
            )
            call_kwargs["q"] = query
            raw_results = await self.api_wrapper.raw_results_async(**call_kwargs)
            _check_empty_results(
                raw_results, query, self.scope, include_raw_content, concise_snippet
            )
            return raw_results
        except ToolException:
            raise
        except Exception as e:
            raise ToolException(str(e)) from e
