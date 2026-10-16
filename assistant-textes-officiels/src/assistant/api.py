"""API HTTP (FastAPI). /ask sera branché à l'étape 8."""

from fastapi import FastAPI

from assistant import __version__, db

app = FastAPI(title="Assistant textes officiels du Sénégal", version=__version__)


@app.get("/health")
def health() -> dict[str, str]:
    """Indique si l'API tourne et si la base de données répond."""
    database = "ok" if db.ping() else "indisponible"
    return {"status": "ok", "database": database, "version": __version__}
