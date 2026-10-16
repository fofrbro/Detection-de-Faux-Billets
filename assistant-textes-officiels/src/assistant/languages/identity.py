"""Traducteur « identité » pour le français, seule langue gérée pour l'instant."""

from assistant.languages.base import Translator, UnsupportedLanguageError


class FrenchPassthrough(Translator):
    """Renvoie le texte tel quel : le français est déjà la langue pivot."""

    lang = "fr"

    def to_french(self, text: str) -> str:
        return text

    def from_french(self, text: str) -> str:
        return text


# Registre des traducteurs disponibles. Wolof, soninké et pulaar viendront ici.
_TRANSLATORS: dict[str, type[Translator]] = {"fr": FrenchPassthrough}


def get_translator(lang: str) -> Translator:
    """Renvoie le traducteur de la langue demandée, ou lève UnsupportedLanguageError."""
    try:
        return _TRANSLATORS[lang]()
    except KeyError:
        raise UnsupportedLanguageError(f"Langue non encore prise en charge : {lang!r}") from None
