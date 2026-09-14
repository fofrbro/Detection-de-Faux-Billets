# Projet 6 - Détection de faux billets

Projet d'analyse de données réalisé dans le cadre de la formation OpenClassrooms.
L'objectif est d'explorer les caractéristiques de billets, de rechercher une
structure dans les données et de construire un modèle capable d'estimer la
probabilité qu'un billet soit authentique.

## Contenu

- `P6_01_code.ipynb` : analyse complète, ACP, classification et régression logistique.
- `P6_02_functions.py` : fonctions réutilisables de visualisation et d'analyse.
- `notes.csv` : données d'apprentissage contenant 170 billets et 7 variables.
- `example.csv` : exemple de données à prédire.
- `P6_03_Exploration_Donnees.html` : rapport d'exploration généré par pandas-profiling.
- `P6_04_Presentation.pdf` : support de présentation du projet.

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
jupyter notebook P6_01_code.ipynb
```

Le notebook et les fichiers CSV doivent rester dans le même dossier, car les
données sont chargées avec des chemins relatifs.

## Publication sur GitHub

Après avoir créé un dépôt vide sur GitHub, exécuter :

```bash
git init
git add .
git commit -m "Initial commit - projet detection de faux billets"
git branch -M main
git remote add origin https://github.com/UTILISATEUR/NOM-DU-DEPOT.git
git push -u origin main
```

Remplacer `UTILISATEUR/NOM-DU-DEPOT` par l'adresse réelle du dépôt GitHub.
"# Detection-de-Faux-Billets" 
