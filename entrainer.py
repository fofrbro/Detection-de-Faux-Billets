"""Entraîne le modèle sur les données d'apprentissage et le sauvegarde.

Usage : python entrainer.py [--donnees FICHIER.csv] [--sortie MODELE.joblib]
"""

import argparse

import modele


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--donnees", default=modele.FICHIER_APPRENTISSAGE)
    parser.add_argument("--sortie", default=modele.FICHIER_MODELE)
    args = parser.parse_args(argv)

    X, y = modele.charger_donnees_apprentissage(args.donnees)
    artefact = modele.entrainer(X, y)
    chemin = modele.sauvegarder(artefact, args.sortie)
    print(f"Modèle entraîné sur {len(X)} billets, sauvegardé dans {chemin}")
    print(f"Paramètres retenus : {artefact['parametres']}")
    print(f"Seuil de décision : {artefact['seuil']:.2f}")


if __name__ == "__main__":
    main()
