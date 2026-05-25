from importlib import metadata

from langchain_siftq.siftq_search import SiftqSearch

try:
    __version__: str = metadata.version(__package__ or "langchain-siftq")
except metadata.PackageNotFoundError:
    __version__ = ""
del metadata

__all__ = [
    "SiftqSearch",
    "__version__",
]
