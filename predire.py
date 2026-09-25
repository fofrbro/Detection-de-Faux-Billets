"""Prédit l'authenticité des billets d'un fichier CSV.

Usage : python predire.py [FICHIER.csv] [--modele MODELE.joblib] [--sortie RESULTATS.csv]

Le fichier doit contenir une colonne ``id`` et les six mesures des billets.
"""

import argparse

import modele


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("fichier", nargs="?", default=modele.FICHIER_A_PREDIRE)
    parser.add_argument("--modele", default=modele.FICHIER_MODELE)
    parser.add_argument("--sortie", help="fichier CSV où écrire les prédictions")
    args = parser.parse_args(argv)

    artefact = modele.charger(args.modele)
    X = modele.charger_donnees_a_predire(args.fichier)
    resultats = modele.predire(artefact, X)
    print(resultats.to_string())
    if args.sortie:
        resultats.to_csv(args.sortie)
        print(f"\nPrédictions écrites dans {args.sortie}")


if __name__ == "__main__":
    main()
