"""Traduction vers et depuis le français (langue pivot). Implémentations à venir."""

from assistant.languages.base import Translator, UnsupportedLanguageError
from assistant.languages.identity import FrenchPassthrough, get_translator

__all__ = ["Translator", "UnsupportedLanguageError", "FrenchPassthrough", "get_translator"]
