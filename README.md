# Projet 6 - Détection de faux billets

Projet d'analyse de données réalisé dans le cadre de la formation OpenClassrooms.
L'objectif est d'explorer les caractéristiques de billets, de rechercher une
structure dans les données et de construire un modèle capable d'estimer la
probabilité qu'un billet soit authentique.

## Contenu

- `analyse_faux_billets.ipynb` : analyse complète, ACP, classification et régression logistique.
- `fonctions_analyse.py` : fonctions réutilisables de visualisation et d'analyse.
- `donnees_apprentissage.csv` : données d'apprentissage contenant 170 billets et 7 variables.
- `donnees_a_predire.csv` : données de billets à prédire.
- `rapport_exploration_donnees.html` : rapport d'exploration généré par pandas-profiling.
- `presentation_projet.pdf` : support de présentation du projet.

## Analyses réalisées

1. Exploration univariée et bivariée des données.
2. Analyse en composantes principales (ACP) et étude des contributions.
3. Classification non supervisée par clustering hiérarchique et K-means.
4. Régression logistique pour prédire `is_genuine`.

## Installation

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

Sous macOS ou Linux, l'activation s'effectue avec :

```bash
source .venv/bin/activate
```

## Exécution

Depuis le dossier du projet :

```bash
jupyter notebook analyse_faux_billets.ipynb
```

Le notebook et les fichiers CSV doivent rester dans le même dossier, car les
données sont chargées avec des chemins relatifs.
"# Detection-de-Faux-Billets" 
