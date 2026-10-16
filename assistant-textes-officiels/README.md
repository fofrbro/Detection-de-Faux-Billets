# Assistant textes officiels du Sénégal

Prototype d'assistant qui répond aux questions des citoyens à partir des textes
officiels sénégalais (RAG). Premier lot : français uniquement, domaine des
démarches administratives.

> Statut : étape 0 (squelette). Les sections marquées *(à venir)* seront complétées
> au fil des étapes.

## Prérequis

- Python 3.12 et [uv](https://docs.astral.sh/uv/)
- Docker et Docker Compose
- Tesseract avec la langue française (`apt install tesseract-ocr tesseract-ocr-fra`) *(étape 3)*

## Installation

```bash
cp .env.example .env      # puis renseigner POSTGRES_PASSWORD, DATABASE_URL, ANTHROPIC_API_KEY
uv sync                   # crée .venv et installe les dépendances
docker compose up -d      # démarre PostgreSQL 16 + pgvector (port 5433 par défaut)
```

## Lancement

```bash
uv run uvicorn assistant.api:app --reload   # API sur http://localhost:8000
curl http://localhost:8000/health
uv run assistant --help                     # CLI : ingest, ask, eval
```

## Tests

```bash
uv run pytest
```

## Ajouter une source *(à venir, étape 1)*

## Lancer l'évaluation *(à venir, étape 10)*
