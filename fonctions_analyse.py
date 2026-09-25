"""Fonctions de visualisation et d'analyse statistique utilisées par le notebook."""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.collections import LineCollection
from scipy import stats
from scipy.cluster.hierarchy import dendrogram

COULEURS_CLASSES = {"Authentique": "#2C92D5", "Faux": "#D62728"}


def libeller_classes(y):
    """Transforme la cible (booléens ou 0/1) en libellés lisibles."""
    return pd.Series(np.where(np.asarray(y).astype(bool), "Authentique", "Faux"),
                     index=getattr(y, "index", None), name="classe")


# --------------------------------------------------------------------------
# Description des données
# --------------------------------------------------------------------------

def synthese_donnees(df):
    """Tableau de synthèse par variable : type, valeurs uniques, manquantes, outliers.

    Les outliers sont détectés par la règle de Tukey (1,5 × écart interquartile)
    sur les variables numériques.
    """
    numeriques = df.select_dtypes("number")
    q1, q3 = numeriques.quantile(0.25), numeriques.quantile(0.75)
    iqr = q3 - q1
    outliers = (numeriques < q1 - 1.5 * iqr) | (numeriques > q3 + 1.5 * iqr)

    synthese = pd.DataFrame(
        {
            "type": df.dtypes.astype(str),
            "valeurs_uniques": df.nunique(),
            "manquantes": df.isna().sum(),
            "taux_manquantes_%": df.isna().mean() * 100,
            "outliers": outliers.sum().reindex(df.columns),
            "taux_outliers_%": (outliers.mean() * 100).reindex(df.columns),
        }
    )
    resume = {
        "lignes": len(df),
        "colonnes": df.shape[1],
        "doublons": int(df.duplicated().sum()),
        "lignes_avec_outlier": int(outliers.any(axis=1).sum()),
    }
    return synthese.round(2), resume


def graphique_repartition(serie, titre=None):
    """Diagramme circulaire de la répartition d'une variable qualitative."""
    effectifs = serie.value_counts()
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.pie(effectifs.values, labels=effectifs.index, autopct="%1.1f%%",
           colors=[COULEURS_CLASSES.get(str(v), None) for v in effectifs.index],
           textprops={"fontsize": 13})
    ax.set_title(titre or f"Répartition de la variable : {serie.name}", size=15)
    return ax


def histogrammes_par_classe(X, classes, n_colonnes=3):
    """Histogrammes de chaque variable, par classe, avec asymétrie et aplatissement."""
    variables = list(X.columns)
    n_lignes = int(np.ceil(len(variables) / n_colonnes))
    fig, axes = plt.subplots(n_lignes, n_colonnes, figsize=(6 * n_colonnes, 4 * n_lignes))
    for ax, variable in zip(axes.ravel(), variables, strict=False):
        sns.histplot(x=X[variable], hue=classes, kde=True, palette=COULEURS_CLASSES,
                     ax=ax, element="step")
        ax.set_title(f"{variable}\nasymétrie = {X[variable].skew():.2f}, "
                     f"aplatissement = {X[variable].kurtosis():.2f}")
    for ax in axes.ravel()[len(variables):]:
        ax.set_visible(False)
    fig.tight_layout()
    return fig


def carte_correlations(df, titre="Carte des corrélations"):
    """Heatmap du triangle inférieur de la matrice de corrélation."""
    corr = df.corr()
    masque = np.triu(np.ones_like(corr, dtype=bool))
    fig, ax = plt.subplots(figsize=(8, 7))
    sns.heatmap(corr, mask=masque, center=0, cmap="RdYlGn", linewidths=1, annot=True,
                fmt=".2f", vmin=-1, vmax=1, ax=ax)
    ax.set_title(titre, fontsize=15, fontweight="bold")
    return ax


# --------------------------------------------------------------------------
# Tests statistiques
# --------------------------------------------------------------------------

def correction_holm(p_values):
    """P-values ajustées par la méthode de Holm-Bonferroni (tests multiples)."""
    p = np.asarray(p_values, dtype=float)
    ordre = np.argsort(p)
    m = len(p)
    ajustees_triees = np.maximum.accumulate((m - np.arange(m)) * p[ordre])
    ajustees = np.empty(m)
    ajustees[ordre] = np.minimum(ajustees_triees, 1.0)
    return ajustees


def tests_normalite(X, alpha=0.05):
    """Test de D'Agostino-Pearson (H0 : la variable suit une loi normale)."""
    lignes = []
    for variable in X.columns:
        statistique, p_value = stats.normaltest(X[variable])
        lignes.append({"variable": variable, "statistique": statistique, "p_value": p_value})
    resultats = pd.DataFrame(lignes).set_index("variable")
    resultats["normalite_rejetee"] = resultats["p_value"] < alpha
    return resultats


def comparer_classes(X, y, alpha=0.05):
    """Compare chaque variable entre billets authentiques et faux.

    - test t de Welch (moyennes, variances non supposées égales) ;
    - test de Mann-Whitney (non paramétrique, sans hypothèse de normalité) ;
    - correction de Holm sur les deux familles de tests ;
    - taille d'effet : d de Cohen.
    """
    y = np.asarray(y).astype(bool)
    lignes = []
    for variable in X.columns:
        vrais, faux = X.loc[y, variable], X.loc[~y, variable]
        _, p_welch = stats.ttest_ind(vrais, faux, equal_var=False)
        _, p_mw = stats.mannwhitneyu(vrais, faux, alternative="two-sided")
        ecart_type_commun = np.sqrt((vrais.var() + faux.var()) / 2)
        lignes.append({
            "variable": variable,
            "moyenne_authentiques": vrais.mean(),
            "moyenne_faux": faux.mean(),
            "d_cohen": (vrais.mean() - faux.mean()) / ecart_type_commun,
            "p_welch": p_welch,
            "p_mann_whitney": p_mw,
        })
    resultats = pd.DataFrame(lignes).set_index("variable")
    resultats["p_welch_holm"] = correction_holm(resultats["p_welch"])
    resultats["p_mann_whitney_holm"] = correction_holm(resultats["p_mann_whitney"])
    resultats["difference_significative"] = (
        (resultats["p_welch_holm"] < alpha) & (resultats["p_mann_whitney_holm"] < alpha)
    )
    return resultats


# --------------------------------------------------------------------------
# Analyse en composantes principales
# --------------------------------------------------------------------------

def _libelle_axe(pca, rang):
    return f"F{rang + 1} ({100 * pca.explained_variance_ratio_[rang]:.1f} %)"


def eboulis_valeurs_propres(pca):
    """Éboulis des valeurs propres avec l'inertie cumulée."""
    inertie = pca.explained_variance_ratio_ * 100
    rangs = np.arange(1, len(inertie) + 1)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(rangs, inertie, color="#2C92D5")
    ax.plot(rangs, inertie.cumsum(), c="red", marker="o", label="Inertie cumulée")
    ax.set_xlabel("Rang de l'axe d'inertie")
    ax.set_ylabel("Pourcentage d'inertie")
    ax.set_title("Éboulis des valeurs propres")
    ax.legend()
    return ax


def correlations_composantes(pca, X_centre_reduit):
    """Corrélations entre chaque variable (lignes) et chaque composante (colonnes)."""
    X = np.asarray(X_centre_reduit)
    projections = pca.transform(X)
    n_var = X.shape[1]
    matrice = np.corrcoef(X.T, projections.T)
    return matrice[:n_var, n_var:]


def cercle_correlations(pca, X_centre_reduit, variables, axes=(0, 1)):
    """Cercle des corrélations sur le plan factoriel ``axes``."""
    d1, d2 = axes
    correlations = correlations_composantes(pca, X_centre_reduit)
    fig, ax = plt.subplots(figsize=(7, 7))
    if len(variables) < 30:
        ax.quiver(np.zeros(len(variables)), np.zeros(len(variables)),
                  correlations[:, d1], correlations[:, d2],
                  angles="xy", scale_units="xy", scale=1, color="grey")
    else:
        lignes = [[[0, 0], [x, y]] for x, y in correlations[:, [d1, d2]]]
        ax.add_collection(LineCollection(lignes, alpha=0.1, color="black"))
    for nom, (x, y) in zip(variables, correlations[:, [d1, d2]], strict=True):
        ax.text(x, y, nom, fontsize=12, ha="center", va="center", color="blue", alpha=0.7)
    ax.add_artist(plt.Circle((0, 0), 1, facecolor="none", edgecolor="b"))
    ax.axhline(0, color="grey", ls="--")
    ax.axvline(0, color="grey", ls="--")
    ax.set_xlim(-1.1, 1.1)
    ax.set_ylim(-1.1, 1.1)
    ax.set_aspect("equal")
    ax.set_xlabel(_libelle_axe(pca, d1))
    ax.set_ylabel(_libelle_axe(pca, d2))
    ax.set_title(f"Cercle des corrélations (F{d1 + 1} et F{d2 + 1})")
    return ax


def plan_factoriel(X_projete, pca, groupes=None, axes=(0, 1), titre=None, palette=None):
    """Projection des individus sur le plan factoriel ``axes``."""
    d1, d2 = axes
    fig, ax = plt.subplots(figsize=(8, 8))
    sns.scatterplot(x=X_projete[:, d1], y=X_projete[:, d2], hue=groupes, palette=palette,
                    alpha=0.8, ax=ax)
    limite = np.max(np.abs(X_projete[:, [d1, d2]])) * 1.1
    ax.set_xlim(-limite, limite)
    ax.set_ylim(-limite, limite)
    ax.axhline(0, color="grey", ls="--")
    ax.axvline(0, color="grey", ls="--")
    ax.set_xlabel(_libelle_axe(pca, d1))
    ax.set_ylabel(_libelle_axe(pca, d2))
    ax.set_title(titre or f"Projection des individus (F{d1 + 1} et F{d2 + 1})")
    return ax


# --------------------------------------------------------------------------
# Classification
# --------------------------------------------------------------------------

def graphique_dendrogramme(Z, nb_feuilles=30, seuil_couleur=None):
    """Dendrogramme tronqué aux ``nb_feuilles`` derniers regroupements."""
    fig, ax = plt.subplots(figsize=(14, 6))
    dendrogram(Z, truncate_mode="lastp", p=nb_feuilles, show_contracted=True,
               color_threshold=seuil_couleur, ax=ax)
    ax.set_title("Classification ascendante hiérarchique (Ward)")
    ax.set_xlabel("Groupes de billets (effectif entre parenthèses)")
    ax.set_ylabel("Distance")
    return ax


def correspondance_clusters(clusters, y):
    """Associe chaque cluster à la classe majoritaire qu'il contient."""
    tableau = pd.crosstab(pd.Series(clusters, name="cluster"), libeller_classes(y).values)
    return tableau.idxmax(axis=1).to_dict()
