# Projet 6 - Détection de faux billets

Projet d'analyse de données réalisé dans le cadre de la formation OpenClassrooms.
L'objectif est d'explorer les caractéristiques de billets, de rechercher une
structure dans les données et de construire un modèle capable d'estimer la
probabilité qu'un billet soit authentique.

## Contenu

| Fichier | Rôle |
| --- | --- |
| `analyse_faux_billets.ipynb` | Analyse complète : exploration, tests statistiques, ACP, clustering, modélisation. |
| `modele.py` | Chargement des données, construction, entraînement, choix du seuil et prédiction. |
| `fonctions_analyse.py` | Fonctions de visualisation et de statistiques utilisées par le notebook. |
| `entrainer.py` | Entraîne le modèle final et le sauvegarde dans `modele/modele_billets.joblib`. |
| `predire.py` | Prédit l'authenticité des billets d'un fichier CSV. |
| `generer_rapport.py` | Génère le rapport d'exploration automatique (ydata-profiling). |
| `donnees_apprentissage.csv` | 170 billets étiquetés (`is_genuine`) décrits par 6 mesures. |
| `donnees_a_predire.csv` | Billets à prédire, identifiés par la colonne `id`. |
| `presentation_projet.pdf` | Support de présentation du projet. |
| `tests/` | Tests automatisés (pytest). |

## Démarche

1. **Exploration** : distributions par classe, corrélations, tests de normalité
   (D'Agostino-Pearson), comparaison des classes (Welch, Mann-Whitney, correction
   de Holm, d de Cohen).
2. **ACP** sur les données centrées-réduites : éboulis, cercle des corrélations,
   contributions, projection des individus.
3. **Classification non supervisée** (CAH de Ward et K-means, choix du nombre de
   clusters par le coude et la silhouette) sans utiliser la cible, puis comparaison
   avec la vraie nature des billets.
4. **Modélisation** : comparaison de plusieurs modèles par validation croisée
   stratifiée répétée, régression logistique régularisée retenue, choix du seuil de
   décision selon le coût des erreurs, évaluation sur un jeu de test mis de côté.

## Le modèle

- Standardisation puis régression logistique ; la régularisation (force et type
  L1 / L2 / elastic-net) est choisie par validation croisée sur la log-loss.
- Le modèle renvoie la probabilité qu'un billet soit **authentique**.
- Accepter un faux billet est considéré comme 5 fois plus grave que rejeter un
  vrai billet. Le seuil de décision minimise ce coût sur des probabilités obtenues
  par validation croisée (0,63 sur les données actuelles, au lieu de 0,5 par défaut).
- Les coûts sont modifiables dans `modele.py` (`COUT_FAUX_ACCEPTE`, `COUT_VRAI_REJETE`).

## Installation

Python 3.11 ou 3.12 est requis.

```bash
python -m venv .venv
source .venv/bin/activate        # Windows : .venv\Scripts\activate
pip install -r requirements.txt
```

## Utilisation

Analyse complète :

```bash
jupyter notebook analyse_faux_billets.ipynb
```

Le notebook s'exécute de haut en bas (*Kernel → Restart & Run All*) ; tous les
tirages aléatoires sont fixés.

Entraînement puis prédiction en ligne de commande :

```bash
python entrainer.py
python predire.py                                  # prédit donnees_a_predire.csv
python predire.py mes_billets.csv --sortie predictions.csv
```

Le fichier à prédire doit contenir une colonne `id` et les colonnes `diagonal`,
`height_left`, `height_right`, `margin_low`, `margin_up` et `length`.

Rapport d'exploration automatique (non versionné, car volumineux) :

```bash
python generer_rapport.py                          # écrit rapport_exploration_donnees.html
```

## Développement

```bash
pip install -r requirements-dev.txt
ruff check .
pytest
```

La CI GitHub Actions exécute le lint, les tests, l'entraînement et la prédiction
en ligne de commande, ainsi que le notebook complet.
