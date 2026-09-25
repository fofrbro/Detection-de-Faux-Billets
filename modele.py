"""Chargement des données, construction et utilisation du modèle de détection.

Le modèle prédit la probabilité qu'un billet soit authentique
(``is_genuine = 1``). Un billet est déclaré authentique lorsque cette
probabilité est supérieure ou égale au seuil de décision, choisi pour
pénaliser davantage un faux billet accepté qu'un vrai billet rejeté.
"""

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

RANDOM_STATE = 42
CIBLE = "is_genuine"
VARIABLES = ["diagonal", "height_left", "height_right", "margin_low", "margin_up", "length"]
COLONNE_ID = "id"

DOSSIER_PROJET = Path(__file__).resolve().parent
FICHIER_APPRENTISSAGE = DOSSIER_PROJET / "donnees_apprentissage.csv"
FICHIER_A_PREDIRE = DOSSIER_PROJET / "donnees_a_predire.csv"
FICHIER_MODELE = DOSSIER_PROJET / "modele" / "modele_billets.joblib"

# Accepter un faux billet est jugé 5 fois plus grave que rejeter un vrai billet.
COUT_FAUX_ACCEPTE = 5.0
COUT_VRAI_REJETE = 1.0


def _verifier_colonnes(df, colonnes, fichier):
    manquantes = [col for col in colonnes if col not in df.columns]
    if manquantes:
        raise ValueError(f"Colonnes manquantes dans {fichier} : {', '.join(manquantes)}")


def charger_donnees_apprentissage(chemin=FICHIER_APPRENTISSAGE):
    """Renvoie les variables explicatives ``X`` et la cible ``y`` (1 = authentique)."""
    df = pd.read_csv(chemin)
    _verifier_colonnes(df, [CIBLE, *VARIABLES], chemin)
    y = df[CIBLE].astype(int)
    return df[VARIABLES], y


def charger_donnees_a_predire(chemin=FICHIER_A_PREDIRE):
    """Renvoie les billets à prédire, indexés par leur identifiant."""
    df = pd.read_csv(chemin)
    _verifier_colonnes(df, [COLONNE_ID, *VARIABLES], chemin)
    return df.set_index(COLONNE_ID)[VARIABLES]


def construire_modele(C=1.0, l1_ratio=0.0):
    """Pipeline standardisation + régression logistique."""
    return Pipeline(
        [
            ("standardisation", StandardScaler()),
            (
                "regression",
                LogisticRegression(
                    C=C,
                    l1_ratio=l1_ratio,
                    solver="saga",
                    max_iter=10_000,
                    random_state=RANDOM_STATE,
                ),
            ),
        ]
    )


def optimiser_modele(X, y, cv=None):
    """Recherche par validation croisée de la régularisation (force et type).

    Le critère est la log-loss : les probabilités renvoyées doivent être
    fiables, pas seulement bien ordonnées (l'AUC est déjà proche de 1).
    """
    grille = {
        "regression__C": np.logspace(-2, 2, 9),
        "regression__l1_ratio": [0.0, 0.5, 1.0],
    }
    cv = cv or StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    recherche = GridSearchCV(construire_modele(), grille, scoring="neg_log_loss", cv=cv)
    return recherche.fit(X, y)


def cout_decision(y, proba_authentique, seuil, cout_faux_accepte=COUT_FAUX_ACCEPTE,
                  cout_vrai_rejete=COUT_VRAI_REJETE):
    """Coût total des erreurs lorsqu'on déclare authentiques les billets au-dessus du seuil."""
    y = np.asarray(y)
    predit_authentique = np.asarray(proba_authentique) >= seuil
    faux_acceptes = np.sum(predit_authentique & (y == 0))
    vrais_rejetes = np.sum(~predit_authentique & (y == 1))
    return cout_faux_accepte * faux_acceptes + cout_vrai_rejete * vrais_rejetes


def choisir_seuil(y, proba_authentique, cout_faux_accepte=COUT_FAUX_ACCEPTE,
                  cout_vrai_rejete=COUT_VRAI_REJETE):
    """Seuil minimisant le coût des erreurs.

    Plusieurs seuils consécutifs atteignent souvent le même coût minimal : on
    retient le milieu de cet intervalle, le plus éloigné des billets observés
    de part et d'autre, plutôt qu'une de ses bornes.
    """
    seuils = np.round(np.arange(0.01, 1.0, 0.01), 2)
    couts = np.array(
        [cout_decision(y, proba_authentique, s, cout_faux_accepte, cout_vrai_rejete)
         for s in seuils]
    )
    candidats = seuils[couts == couts.min()]
    return float(candidats[np.argmin(np.abs(candidats - np.median(candidats)))])


def entrainer(X, y):
    """Optimise le modèle, choisit le seuil puis réentraîne sur toutes les données.

    Le seuil est choisi sur des probabilités obtenues par validation croisée,
    pour ne pas utiliser des prédictions faites sur les données d'entraînement.
    """
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    recherche = optimiser_modele(X, y, cv=cv)
    estimateur = recherche.best_estimator_
    proba_oof = cross_val_predict(estimateur, X, y, cv=cv, method="predict_proba")[:, 1]
    seuil = choisir_seuil(y, proba_oof)
    estimateur.fit(X, y)
    return {"modele": estimateur, "seuil": seuil, "variables": VARIABLES,
            "parametres": {cle: float(valeur) for cle, valeur in recherche.best_params_.items()}}


def predire(artefact, X):
    """Probabilité d'authenticité et statut prédit pour chaque billet."""
    proba = artefact["modele"].predict_proba(X[artefact["variables"]])[:, 1]
    authentique = proba >= artefact["seuil"]
    return pd.DataFrame(
        {
            "proba_authentique": proba.round(4),
            "prediction": authentique.astype(int),
            "statut": np.where(authentique, "Billet authentique", "Faux billet"),
        },
        index=X.index,
    )


def sauvegarder(artefact, chemin=FICHIER_MODELE):
    chemin = Path(chemin)
    chemin.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(artefact, chemin)
    return chemin


def charger(chemin=FICHIER_MODELE):
    chemin = Path(chemin)
    if not chemin.exists():
        raise FileNotFoundError(
            f"Modèle introuvable ({chemin}). Lancez d'abord : python entrainer.py"
        )
    return joblib.load(chemin)
