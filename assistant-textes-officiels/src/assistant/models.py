"""Objets métier échangés entre les modules."""

from pydantic import BaseModel, Field

# Langues prévues. Seul le français est géré dans ce premier lot ;
# les autres passeront par le français comme langue pivot (module languages/).
SUPPORTED_LANGS = ("fr", "wo", "snk", "ff")


class Question(BaseModel):
    """Question posée par un citoyen."""

    question: str = Field(min_length=1, max_length=2000)
    lang: str = Field(default="fr", description="Code langue : fr, wo (wolof), snk (soninké), ff (pulaar)")


class Source(BaseModel):
    """Extrait d'un texte officiel cité dans une réponse."""

    title: str
    article: str | None = None
    date: str | None = None
    url: str
    status: str = "inconnu"  # en_vigueur / abroge / inconnu


class Answer(BaseModel):
    """Réponse renvoyée au citoyen."""

    answer: str
    sources: list[Source] = Field(default_factory=list)
    lang: str = "fr"
