"""Ligne de commande : `assistant ingest`, `assistant ask`, `assistant eval`."""

import typer

app = typer.Typer(help="Assistant RAG sur les textes officiels sénégalais.", no_args_is_help=True)


@app.command()
def ingest() -> None:
    """Télécharge et indexe les documents des sources officielles (étape 2)."""
    typer.echo("Pas encore implémenté (étape 2).")
    raise typer.Exit(code=1)


@app.command()
def ask(question: str, lang: str = "fr") -> None:
    """Pose une question et affiche la réponse avec ses sources (étape 9)."""
    typer.echo("Pas encore implémenté (étape 9).")
    raise typer.Exit(code=1)


@app.command(name="eval")
def evaluate() -> None:
    """Mesure le taux de bonne source dans le top 5 (étape 10)."""
    typer.echo("Pas encore implémenté (étape 10).")
    raise typer.Exit(code=1)


if __name__ == "__main__":
    app()
