"""Configuration de l'application, lue depuis les variables d'environnement et le fichier .env."""

from functools import lru_cache
from pathlib import Path

from pydantic import SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Paramètres de l'application. Aucune valeur secrète n'a de défaut dans le code."""

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # Base de données
    database_url: str = "postgresql://assistant@localhost:5433/assistant"

    # Embeddings
    embedding_model: str = "BAAI/bge-m3"
    embedding_dim: int = 1024

    # LLM
    anthropic_api_key: SecretStr | None = None
    llm_model: str = "claude-sonnet-5-5"
    llm_max_tokens: int = 1500

    # Ingestion
    scraper_user_agent: str = "AssistantTextesOfficielsSN/0.1"
    scraper_min_delay_seconds: float = 2.0
    cache_dir: Path = Path("data/cache")

    # Recherche
    search_top_k: int = 5


@lru_cache
def get_settings() -> Settings:
    """Renvoie l'instance unique des paramètres (chargée une seule fois)."""
    return Settings()
