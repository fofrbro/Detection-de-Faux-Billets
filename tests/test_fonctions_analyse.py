import matplotlib

matplotlib.use("Agg")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import pytest  # noqa: E402
from sklearn.decomposition import PCA  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402

import fonctions_analyse as fa  # noqa: E402
import modele  # noqa: E402


@pytest.fixture(scope="module")
def donnees():
    return modele.charger_donnees_apprentissage()


def test_libeller_classes():
    assert list(fa.libeller_classes(pd.Series([1, 0, True]))) == ["Authentique", "Faux",
                                                                 "Authentique"]


def test_synthese_donnees():
    df = pd.DataFrame({"a": [1.0, 2.0, 3.0, 100.0, 2.5], "b": [1, 1, None, 1, 1]})
    synthese, resume = fa.synthese_donnees(df)
    assert resume == {"lignes": 5, "colonnes": 2, "doublons": 0, "lignes_avec_outlier": 1}
    assert synthese.loc["a", "outliers"] == 1
    assert synthese.loc["b", "manquantes"] == 1
    assert synthese.loc["b", "taux_manquantes_%"] == 20.0


def test_correction_holm():
    ajustees = fa.correction_holm([0.01, 0.04, 0.03, 0.5])
    np.testing.assert_allclose(ajustees, [0.04, 0.09, 0.09, 0.5])


def test_correction_holm_bornee_a_un():
    assert fa.correction_holm([0.6, 0.7]).max() == 1.0


def test_comparer_classes(donnees):
    X, y = donnees
    resultats = fa.comparer_classes(X, y)
    assert resultats.loc["margin_low", "difference_significative"]
    assert resultats.loc["margin_low", "d_cohen"] < 0  # marges plus petites chez les vrais
    assert (resultats["p_welch_holm"] >= resultats["p_welch"]).all()


def test_correlations_composantes(donnees):
    X, _ = donnees
    X_cr = StandardScaler().fit_transform(X)
    pca = PCA().fit(X_cr)
    correlations = fa.correlations_composantes(pca, X_cr)
    assert correlations.shape == (6, 6)
    # Avec toutes les composantes, la somme des carrés des corrélations d'une variable vaut 1.
    np.testing.assert_allclose((correlations ** 2).sum(axis=1), 1.0)


def test_correspondance_clusters():
    clusters = np.array([1, 1, 2, 2, 2])
    y = np.array([0, 0, 1, 1, 0])
    assert fa.correspondance_clusters(clusters, y) == {1: "Faux", 2: "Authentique"}


def test_graphiques_sans_erreur(donnees):
    X, y = donnees
    X_cr = StandardScaler().fit_transform(X)
    pca = PCA().fit(X_cr)
    fa.histogrammes_par_classe(X, fa.libeller_classes(y))
    fa.carte_correlations(X)
    fa.eboulis_valeurs_propres(pca)
    fa.cercle_correlations(pca, X_cr, X.columns)
    fa.plan_factoriel(pca.transform(X_cr), pca, groupes=fa.libeller_classes(y))
    matplotlib.pyplot.close("all")
