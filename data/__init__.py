"""
Data Access & Ingestion Package
"""

try:
    from .loader import MovieLensDataLoader
except ImportError:
    from data.loader import MovieLensDataLoader

__all__ = ["MovieLensDataLoader"]
