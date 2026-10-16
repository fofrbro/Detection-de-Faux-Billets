"""Tests de l'interface de traduction."""

import pytest

from assistant.languages import FrenchPassthrough, UnsupportedLanguageError, get_translator


def test_francais_inchange() -> None:
    traducteur = get_translator("fr")
    assert isinstance(traducteur, FrenchPassthrough)
    assert traducteur.to_french("Bonjour") == "Bonjour"
    assert traducteur.from_french("Bonjour") == "Bonjour"


def test_langue_non_geree() -> None:
    with pytest.raises(UnsupportedLanguageError):
        get_translator("wo")
