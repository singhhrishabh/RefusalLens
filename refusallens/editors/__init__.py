"""Editors package — edit operators that generate candidate prompt rewrites."""

from refusallens.utils import Edit
from refusallens.editors.base import BaseEditor
from refusallens.editors.token_edit import TokenEditor
from refusallens.editors.paraphrase import ParaphraseEditor
from refusallens.editors.framing import FramingEditor

ALL_EDITORS = [TokenEditor, ParaphraseEditor, FramingEditor]

__all__ = [
    "BaseEditor",
    "Edit",
    "TokenEditor",
    "ParaphraseEditor",
    "FramingEditor",
    "ALL_EDITORS",
]
