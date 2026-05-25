import json
from typing import Any, Literal

import aiohttp
import requests
from langchain_core.utils import get_from_dict_or_env
from pydantic import BaseModel, ConfigDict, PrivateAttr, SecretStr, model_validator

SIFTQ_API_URL: str = "https://api.siftq.com/v1"
SIFTQ_DEFAULT_API_KEY: str = "mk-1D3D81EFC32A25683B0C2C3B315F8579"

Scope = Literal["webpage", "document", "scholar", "image", "video", "podcast"]

_ERROR_CODES = {
    2005: "API key rejected. Please check your SIFTQ_API_KEY.",
    3003: "Daily search limit reached. Please wait or upgrade your plan.",
}


def _parse_error(status_code: int, body: Any) -> ValueError:
    if isinstance(body, dict):
        code = body.get("code")
        message = body.get("message") or body.get("error", "Unknown error")
        if code in _ERROR_CODES:
            return ValueError(f"Error {code}: {_ERROR_CODES[code]}")
        if message != "Unknown error":
            return ValueError(f"Error {code or status_code}: {message}")
    return ValueError(f"HTTP {status_code}: Unknown error")


def _check_response(body: Any, status_code: int) -> None:
    if status_code != 200:
        raise _parse_error(status_code, body)
    if isinstance(body, dict) and body.get("code"):
        raise _parse_error(status_code, body)


class SiftqAPIWrapper(BaseModel):
    siftq_api_key: SecretStr
    api_base_url: str | None = None

    model_config = ConfigDict(extra="forbid")

    _session: aiohttp.ClientSession | None = PrivateAttr(default=None)
    _http: requests.Session = PrivateAttr(default_factory=requests.Session)

    def _get_session(self) -> aiohttp.ClientSession:
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession()
        return self._session

    async def close(self) -> None:
        if self._session and not self._session.closed:
            await self._session.close()
            self._session = None
        self._http.close()

    @model_validator(mode="before")
    @classmethod
    def validate_environment(cls, values: dict[str, Any]) -> Any:
        if "siftq_api_key" not in values or not values.get("siftq_api_key"):
            try:
                siftq_api_key = get_from_dict_or_env(values, "siftq_api_key", "SIFTQ_API_KEY")
            except ValueError:
                siftq_api_key = SIFTQ_DEFAULT_API_KEY
            values["siftq_api_key"] = siftq_api_key
        return values

    def raw_results(
        self,
        q: str,
        scope: Scope | None = None,
        include_summary: bool | None = None,
        include_raw_content: bool | None = None,
        concise_snippet: bool | None = None,
        size: int | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        params: dict[str, Any] = {
            "q": q,
            "scope": scope,
            "includeSummary": include_summary,
            "includeRawContent": include_raw_content,
            "conciseSnippet": concise_snippet,
            "size": size,
            **kwargs,
        }

        params = {k: v for k, v in params.items() if v is not None}

        headers = {
            "Authorization": f"Bearer {self.siftq_api_key.get_secret_value()}",
            "Content-Type": "application/json",
        }

        base_url = self.api_base_url or SIFTQ_API_URL
        response = self._http.post(
            f"{base_url}/search",
            json=params,
            headers=headers,
        )
        try:
            body = response.json()
        except ValueError:
            raise ValueError(f"HTTP {response.status_code}: Non-JSON response") from None
        _check_response(body, response.status_code)
        return body

    async def raw_results_async(
        self,
        q: str,
        scope: Scope | None = None,
        include_summary: bool | None = None,
        include_raw_content: bool | None = None,
        concise_snippet: bool | None = None,
        size: int | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        params: dict[str, Any] = {
            "q": q,
            "scope": scope,
            "includeSummary": include_summary,
            "includeRawContent": include_raw_content,
            "conciseSnippet": concise_snippet,
            "size": size,
            **kwargs,
        }

        params = {k: v for k, v in params.items() if v is not None}

        headers = {
            "Authorization": f"Bearer {self.siftq_api_key.get_secret_value()}",
            "Content-Type": "application/json",
        }

        base_url = self.api_base_url or SIFTQ_API_URL
        session = self._get_session()
        async with session.post(f"{base_url}/search", json=params, headers=headers) as res:
            text = await res.text()
            if res.status != 200:
                try:
                    body = json.loads(text)
                except json.JSONDecodeError:
                    raise ValueError(f"HTTP {res.status}: {text[:200]}") from None
                raise _parse_error(res.status, body)

        try:
            body = json.loads(text)
        except json.JSONDecodeError:
            raise ValueError("Invalid JSON in response body") from None
        _check_response(body, 200)
        return body
