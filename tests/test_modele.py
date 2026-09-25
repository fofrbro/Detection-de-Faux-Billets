import numpy as np
import pandas as pd
import pytest

import entrainer
import modele
import predire


@pytest.fixture(scope="module")
def donnees():
    return modele.charger_donnees_apprentissage()


def test_chargement_apprentissage(donnees):
    X, y = donnees
    assert list(X.columns) == modele.VARIABLES
    assert len(X) == len(y) == 170
    assert set(y.unique()) == {0, 1}
    assert not X.isna().any().any()


def test_chargement_a_predire():
    X = modele.charger_donnees_a_predire()
    assert list(X.columns) == modele.VARIABLES
    assert X.index.name == "id"


def test_colonne_manquante(tmp_path):
    chemin = tmp_path / "incomplet.csv"
    pd.DataFrame({"id": ["A"], "diagonal": [171.0]}).to_csv(chemin, index=False)
    with pytest.raises(ValueError, match="Colonnes manquantes"):
        modele.charger_donnees_a_predire(chemin)


def test_cout_decision():
    y = np.array([0, 0, 1, 1])
    proba = np.array([0.1, 0.6, 0.4, 0.9])
    # seuil 0,5 : le faux à 0,6 est accepté (coût 5), le vrai à 0,4 est rejeté (coût 1)
    assert modele.cout_decision(y, proba, 0.5) == 6
    # seuil 0,7 : le faux est rejeté, le vrai à 0,4 reste rejeté (coût 1)
    assert modele.cout_decision(y, proba, 0.7) == 1


def test_choisir_seuil_penalise_les_faux_acceptes():
    y = np.array([0, 0, 1, 1])
    proba = np.array([0.1, 0.6, 0.4, 0.9])
    seuil = modele.choisir_seuil(y, proba)
    # Rejeter le vrai billet à 0,4 coûte moins cher qu'accepter le faux à 0,6.
    assert 0.6 < seuil <= 0.9
    assert modele.cout_decision(y, proba, seuil) == 1


def test_choisir_seuil_milieu_de_l_intervalle_optimal():
    y = np.array([0, 1])
    proba = np.array([0.2, 0.8])
    assert modele.choisir_seuil(y, proba) == pytest.approx(0.5, abs=0.01)


def test_entrainement_et_prediction(donnees, tmp_path):
    X, y = donnees
    artefact = modele.entrainer(X, y)
    assert 0 < artefact["seuil"] < 1

    chemin = modele.sauvegarder(artefact, tmp_path / "modele.joblib")
    recharge = modele.charger(chemin)
    resultats = modele.predire(recharge, X)

    assert list(resultats.columns) == ["proba_authentique", "prediction", "statut"]
    assert resultats["proba_authentique"].between(0, 1).all()
    assert (resultats["prediction"] == y).mean() > 0.95
    attendu = np.where(resultats["prediction"] == 1, "Billet authentique", "Faux billet")
    assert (resultats["statut"] == attendu).all()


def test_charger_modele_absent(tmp_path):
    with pytest.raises(FileNotFoundError, match="entrainer.py"):
        modele.charger(tmp_path / "absent.joblib")


def test_scripts_en_ligne_de_commande(tmp_path, capsys):
    chemin_modele = tmp_path / "modele.joblib"
    chemin_sortie = tmp_path / "predictions.csv"
    entrainer.main(["--sortie", str(chemin_modele)])
    predire.main(["--modele", str(chemin_modele), "--sortie", str(chemin_sortie)])

    resultats = pd.read_csv(chemin_sortie, index_col="id")
    assert list(resultats.index) == ["A_1", "A_2", "A_3", "A_4", "A_5"]
    assert list(resultats["prediction"]) == [0, 0, 0, 1, 1]
    assert "Seuil de décision" in capsys.readouterr().out
