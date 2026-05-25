# langchain-siftq

LangChain integration for [SiftQ](https://siftq.com) — a retrieval layer for
agents that need precise, fresh, structured web context.

## Quick start

```python
from langchain_siftq import SiftqSearch

# No API key needed — free tier works out of the box
tool = SiftqSearch()
result = tool.invoke({"query": "what is the latest AI research"})
```

## Authentication

The free tier provides 100 searches/day with no sign-up
required. See <https://siftq.com> for details.

Priority (highest to lowest):
1. `siftq_api_key` kwarg passed to `SiftqSearch()`
2. `SIFTQ_API_KEY` environment variable
3. 100 searches/day free tier

## Installation

```bash
pip install langchain-siftq
```

## Development

See [docs/development.md](docs/development.md).
