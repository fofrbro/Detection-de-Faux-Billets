"""Accès à PostgreSQL (psycopg 3, SQL brut)."""

from collections.abc import Iterator
from contextlib import contextmanager

import psycopg

from assistant.config import get_settings


@contextmanager
def connect(database_url: str | None = None) -> Iterator[psycopg.Connection]:
    """Ouvre une connexion à la base et la ferme à la sortie du bloc."""
    url = database_url or get_settings().database_url
    with psycopg.connect(url) as conn:
        yield conn


def ping(database_url: str | None = None) -> bool:
    """Vérifie que la base répond. Renvoie False au lieu de lever une exception."""
    try:
        with connect(database_url) as conn:
            conn.execute("SELECT 1")
        return True
    except psycopg.Error:
        return False
