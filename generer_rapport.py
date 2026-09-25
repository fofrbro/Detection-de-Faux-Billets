"""Génère le rapport d'exploration automatique des données d'apprentissage.

Usage : python generer_rapport.py [--sortie rapport_exploration_donnees.html]
"""

import argparse

import pandas as pd

import modele


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--sortie", default="rapport_exploration_donnees.html")
    args = parser.parse_args(argv)

    from ydata_profiling import ProfileReport

    donnees = pd.read_csv(modele.FICHIER_APPRENTISSAGE)
    rapport = ProfileReport(donnees, title="Détection de faux billets - exploration des données")
    rapport.to_file(args.sortie)
    print(f"Rapport écrit dans {args.sortie}")


if __name__ == "__main__":
    main()
