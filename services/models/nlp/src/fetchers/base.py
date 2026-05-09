from abc import ABC, abstractmethod
from typing import List, Dict


class BaseFetcher(ABC):
    """Abstract base class for all news fetchers."""

    @abstractmethod
    def fetch(self) -> List[Dict]:
        """Fetch articles and return a normalized list of article dicts."""
        ...
