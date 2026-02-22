# 🏦 TP2 - Classification : Bank Telemarketing

> **Module** : Introduction à l'Intelligence Artificielle  
> **Formation** : Licence MTQ 3ème année (S6)  
> **Université** : Université Saint Jean (ISJM)  
> **Année académique** : 2025-2026

---

## 📌 Description du projet

Ce projet correspond à la **Partie 3 du TP2** qui consiste à appliquer des méthodes de
Machine Learning sur un jeu de données réel pour effectuer une tâche de **classification binaire**.

L'objectif est de prédire si un client d'une banque va **souscrire ou non** à un dépôt à terme
suite à une campagne de marketing téléphonique.

- **Tâche** : Classification binaire (`yes` / `no`)
- **Variable cible** : `y` (souscription au dépôt à terme)
- **Algorithmes utilisés** : KNN, Arbre de décision, Random Forest, Gradient Boosting

---

## 📂 Structure du projet

```
TP2-Bank-Telemarketing/
│
├── tp2_partie3_bank_telemarketing.py   ← Code Python principal
├── bank-additional.csv                 ← Jeu de données
├── bank_telemarketing.pkl              ← Meilleur modèle sauvegardé
├── README.md                           ← Ce fichier
│
└── images/                             ← Graphiques générés
    ├── distribution_cible.png
    ├── pairplot_variables.png
    ├── heatmap_correlation.png
    ├── variables_categorielles.png
    ├── knn_influence_k.png
    ├── dt_cv_depth.png
    ├── bagging_nb_arbres.png
    ├── comparaison_modeles.png
    ├── matrice_confusion.png
    ├── courbes_roc.png
    └── importance_variables.png
```

---

## 🗃️ Dataset : Bank Telemarketing

| Propriété | Valeur |
|-----------|--------|
| **Source** | UCI Machine Learning Repository |
| **Fichier** | bank-additional.csv |
| **Séparateur** | `;` (point-virgule) |
| **Instances** | 4 119 clients |
| **Features** | 20 variables (numériques + catégorielles) |
| **Cible** | `y` → `yes` (souscription) / `no` (refus) |

### Variables principales

| Variable | Type | Description |
|----------|------|-------------|
| `age` | Numérique | Âge du client |
| `job` | Catégorielle | Type d'emploi |
| `marital` | Catégorielle | Statut marital |
| `education` | Catégorielle | Niveau d'éducation |
| `duration` | Numérique | Durée du dernier appel (secondes) |
| `campaign` | Numérique | Nombre de contacts durant cette campagne |
| `poutcome` | Catégorielle | Résultat de la campagne précédente |
| `y` | Cible | Souscription au dépôt à terme |

---

## ⚙️ Installation

### Prérequis
- Python 3.8 ou supérieur
- pip

### Installer les dépendances

```bash
pip install pandas matplotlib seaborn scikit-learn
```

---

## 🚀 Utilisation

1. Clone le dépôt :
```bash
git clone https://github.com/TonNom/TP2-Bank-Telemarketing.git
cd TP2-Bank-Telemarketing
```

2. Place le fichier `bank-additional.csv` dans le dossier racine

3. Lance le script :
```bash
python tp2_partie3_bank_telemarketing.py
```

---

## 🔬 Méthodologie

### 1. Exploration des données (EDA)
- Analyse des dimensions, types et statistiques descriptives
- Visualisation de la distribution de la variable cible
- Pairplot et heatmap de corrélation
- Distribution des variables catégorielles par classe

### 2. Préparation des données
- Vérification et gestion des valeurs manquantes
- Encodage One-Hot des variables catégorielles (dummy variables)
- Séparation Train/Test : **80% / 20%** (stratifiée)
- Normalisation avec `StandardScaler` (pour KNN)

### 3. Modèles entraînés

| Modèle | Optimisation |
|--------|-------------|
| **Classifieur constant** | Baseline de référence |
| **KNN** | GridSearch sur k (1 à 30) |
| **Arbre de décision (CART)** | GridSearch sur max_depth + criterion |
| **Bagging** | Variation du nombre d'arbres B |
| **Random Forest** | GridSearch sur max_features + OOB score |
| **Gradient Boosting** | GridSearch sur n_estimators, learning_rate, max_depth |

### 4. Évaluation
- Accuracy, Precision, Recall, F1-score
- Matrice de confusion
- Courbes ROC et AUC
- Importance des variables

---

## 📊 Résultats

| Modèle | Accuracy (test) |
|--------|----------------|
| Baseline (constant) | ~0.887 |
| KNN | ~0.895 |
| Arbre de décision | ~0.905 |
| Random Forest | ~0.920 |
| **Gradient Boosting** | **~0.925** ⭐ |

> ⭐ **Meilleur modèle** : Gradient Boosting  
> 📦 Sauvegardé dans `bank_telemarketing.pkl`

---

## 💾 Charger le modèle sauvegardé

```python
import pickle

with open("bank_telemarketing.pkl", "rb") as f:
    pipeline = pickle.load(f)

model = pipeline['model']
scaler = pipeline['scaler']
feature_cols = pipeline['feature_cols']

# Prédiction sur de nouvelles données
# y_pred = model.predict(X_new)
```

---

## 📚 Bibliothèques utilisées

| Bibliothèque | Usage |
|---|---|
| `pandas` | Chargement et manipulation des données |
| `matplotlib` | Visualisation |
| `seaborn` | Visualisation avancée |
| `scikit-learn` | Modèles ML, évaluation, preprocessing |
| `pickle` | Sauvegarde du modèle |

---

## 👤 Auteur

#BAPOWA MERAWA FLORA RAISSA
Étudiant en Licence MTQ 3ème année  
Université Saint Jean School of management (SJM) — 2025/2026

---

## 📄 Références

- [UCI Machine Learning Repository – Bank Marketing Dataset](https://archive.ics.uci.edu/ml/datasets/bank+marketing)
- [Scikit-learn Documentation](https://scikit-learn.org/stable/)
- Moro, S., Cortez, P., & Rita, P. (2014). *A data-driven approach to predict the success of bank telemarketing.*
