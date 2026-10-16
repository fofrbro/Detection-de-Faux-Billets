# CLAUDE.md

## Projet
Prototype RAG qui répond aux questions des citoyens sur les textes officiels
sénégalais. Lot 1 : français, domaine « démarches administratives ».
Plus tard (ne pas coder sans demande) : traduction wolof / soninké / pulaar avec le
français comme pivot, voix (transcription + synthèse), bot WhatsApp.

## Stack
- Python 3.12, dépendances via `uv` (`uv sync`, `uv run …`)
- PostgreSQL 16 + pgvector (`docker compose up -d`, port hôte 5433)
- psycopg 3 en SQL brut, migrations `.sql` numérotées dans `migrations/`
- FastAPI (`src/assistant/api.py`), CLI typer (`src/assistant/cli.py`)
- PyMuPDF + Tesseract (fra) pour l'extraction ; embeddings `BAAI/bge-m3` (configurable)
- LLM : API Anthropic, modèle défini par `LLM_MODEL` dans `.env`
- pytest (`uv run pytest`)

## Architecture
- `src/assistant/config.py` : paramètres (pydantic-settings, `.env`)
- `src/assistant/models.py` : Question (champ `lang`), Answer, Source
- `src/assistant/languages/` : classe abstraite `Translator` ; seul `fr` existe
- `ingestion/`, `extraction/`, `chunking.py`, `search.py`, `generation.py` : pipeline RAG
- `sources.yaml` : liste fermée des sources officielles vérifiées
- `eval/questions.yaml` : questions d'évaluation avec source attendue

## Conventions
- Code, commentaires et docstrings en français ; fonctions courtes ; type hints partout.
- Aucun secret dans le code : tout passe par `.env` (modèle : `.env.example`).
- Ne jamais inventer une URL ou un texte officiel. Une source inaccessible est signalée, pas remplacée.
- Scraping poli : robots.txt respecté, ≥ 2 s entre requêtes, user-agent identifiable, cache local.
- Statut d'un texte (en vigueur / abrogé) : « inconnu » sauf indication explicite de la source.
- Un commit par étape validée par l'utilisateur.
