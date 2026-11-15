"""Base editor definition for RefusalLens."""

from abc import ABC, abstractmethod
from typing import List
from refusallens.utils import Edit


class BaseEditor(ABC):
    """Base class for all edit operators."""

    @property
    def name(self) -> str:
        """Return the editor name."""
        return self.__class__.__name__

    @abstractmethod
    def generate_candidates(self, prompt: str, budget: int = 3) -> List[Edit]:
        """Generate candidate edits for the given prompt.

        Args:
            prompt: The original prompt text.
            budget: Maximum number of candidate edits to return.

        Returns:
            A list of Edit objects representing candidate rewrites.
        """
        raise NotImplementedError
