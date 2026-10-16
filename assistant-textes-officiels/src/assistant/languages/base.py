"""Interface commune des traducteurs."""

from abc import ABC, abstractmethod


class UnsupportedLanguageError(ValueError):
    """La langue demandée n'a pas encore de traducteur."""


class Translator(ABC):
    """Traduit entre une langue locale et le français, qui sert de langue pivot.

    Le pipeline RAG travaille toujours en français :
    question (lang) -> to_french -> recherche + génération -> from_french -> réponse (lang).
    """

    #: Code de la langue gérée (ex. "wo" pour le wolof).
    lang: str

    @abstractmethod
    def to_french(self, text: str) -> str:
        """Traduit un texte de `lang` vers le français."""

    @abstractmethod
    def from_french(self, text: str) -> str:
        """Traduit un texte du français vers `lang`."""
