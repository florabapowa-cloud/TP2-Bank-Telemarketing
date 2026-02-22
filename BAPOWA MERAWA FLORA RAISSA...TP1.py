
# TP 2 - Partie 3 : Classification sur le dataset Bank Telemarketing
# Dataset : bank-additional.csv
# Objectif : Prédire si un client va souscrire (y = 'yes'/'no') à un dépôt à terme


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import (accuracy_score, confusion_matrix, classification_report,
                             roc_curve, roc_auc_score, ConfusionMatrixDisplay)
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (RandomForestClassifier, GradientBoostingClassifier,
                               BaggingClassifier)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.dummy import DummyClassifier

# PARTIE A : CHARGEMENT ET DESCRIPTION DES DONNÉES


print("=" * 60)
print("   PARTIE A : DESCRIPTION ET VISUALISATION DES DONNÉES")
print("=" * 60)


df = pd.read_csv("bank-additional.csv", sep=";")

print("\n📌 Dimensions du dataset :")
print(f"   {df.shape[0]} instances, {df.shape[1]} colonnes")

print("\n📌 Informations générales :")
print(df.info())

print("\n📌 Statistiques descriptives :")
print(df.describe())

print("\n📌 Premières lignes :")
print(df.head())

print("\n📌 Nombre d'instances par classe (variable cible 'y') :")
print(df['y'].value_counts())
print(f"\n   Proportion :\n{df['y'].value_counts(normalize=True).round(3)}")

# Visualisation de la distribution de la cible
plt.figure(figsize=(6, 4))
df['y'].value_counts().plot(kind='bar', color=['steelblue', 'salmon'], edgecolor='black')
plt.title("Distribution de la variable cible (y)")
plt.xlabel("Souscription au dépôt")
plt.ylabel("Nombre de clients")
plt.xticks(rotation=0)
plt.tight_layout()
plt.savefig("distribution_cible.png", dpi=150)
plt.show()
print("   → Graphique sauvegardé : distribution_cible.png")


# PARTIE B : VISUALISATION DES VARIABLES


print("\n" + "=" * 60)
print("   PARTIE B : VISUALISATION DES VARIABLES")
print("=" * 60)

# Variables numériques
num_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
print(f"\n📌 Variables numériques ({len(num_cols)}) : {num_cols}")

# Pairplot sur un sous-ensemble de variables numériques
subset_cols = ['age', 'duration', 'campaign', 'pdays', 'previous', 'y']
df_subset = df[subset_cols].copy()

plt.figure(figsize=(10, 8))
sns.pairplot(df_subset, hue='y', palette={'yes': 'green', 'no': 'red'}, diag_kind='kde')
plt.suptitle("Pairplot des variables numériques clés", y=1.02)
plt.savefig("pairplot_variables.png", dpi=150, bbox_inches='tight')
plt.show()
print("   → Graphique sauvegardé : pairplot_variables.png")

# Heatmap de corrélation (variables numériques uniquement)
plt.figure(figsize=(10, 7))
corr_matrix = df[num_cols].corr()
sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap="coolwarm", linewidths=0.5)
plt.title("Matrice de corrélation des variables numériques")
plt.tight_layout()
plt.savefig("heatmap_correlation.png", dpi=150)
plt.show()
print("   → Graphique sauvegardé : heatmap_correlation.png")

# Distribution des variables catégorielles
cat_cols = df.select_dtypes(include=['object']).columns.drop('y').tolist()
print(f"\n📌 Variables catégorielles ({len(cat_cols)}) : {cat_cols}")

fig, axes = plt.subplots(3, 3, figsize=(16, 12))
axes = axes.flatten()
for i, col in enumerate(cat_cols[:9]):
    df.groupby([col, 'y']).size().unstack().plot(
        kind='bar', ax=axes[i], color=['salmon', 'steelblue'], edgecolor='black'
    )
    axes[i].set_title(f"Distribution de '{col}' par classe")
    axes[i].set_xlabel("")
    axes[i].tick_params(axis='x', rotation=45)
plt.suptitle("Variables catégorielles vs Cible", fontsize=14)
plt.tight_layout()
plt.savefig("variables_categorielles.png", dpi=150)
plt.show()
print("   → Graphique sauvegardé : variables_categorielles.png")

# PARTIE C : PRÉPARATION DES DONNÉES


print("\n" + "=" * 60)
print("   PARTIE C : PRÉPARATION DES DONNÉES")
print("=" * 60)

# Vérification des valeurs manquantes
print(f"\n📌 Valeurs manquantes par colonne :\n{df.isnull().sum()}")

# Encodage de la variable cible
df['y_encoded'] = (df['y'] == 'yes').astype(int)
print(f"\n📌 Encodage de la cible : 'yes' → 1, 'no' → 0")

# Encodage des variables catégorielles (One-Hot Encoding / Dummy variables)
df_encoded = pd.get_dummies(df.drop(columns=['y', 'y_encoded']), drop_first=True)
print(f"\n📌 Dimensions après encodage one-hot : {df_encoded.shape}")

# Variables X et y
X = df_encoded
y = df['y_encoded']

print(f"\n📌 Nombre de features : {X.shape[1]}")
print(f"📌 Distribution de la cible encodée :\n{y.value_counts()}")

# Séparation Train / Test (80% / 20%)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"\n📌 Taille ensemble d'apprentissage : {X_train.shape[0]}")
print(f"📌 Taille ensemble de test          : {X_test.shape[0]}")

# Normalisation (utile pour KNN)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)


# PARTIE D : CLASSIFIEUR CONSTANT (référentiel de base)


print("\n" + "=" * 60)
print("   PARTIE D : CLASSIFIEUR CONSTANT (BASELINE)")
print("=" * 60)

dummy = DummyClassifier(strategy='most_frequent', random_state=42)
dummy.fit(X_train, y_train)
dummy_score = dummy.score(X_test, y_test)
print(f"\n📌 Meilleur classifieur constant → classe majoritaire ('no')")
print(f"   Accuracy (test) : {dummy_score:.4f}")
print(f"   → C'est le score minimum à battre !")


# PARTIE E : K-NEAREST NEIGHBORS (KNN)


print("\n" + "=" * 60)
print("   PARTIE E : K-NEAREST NEIGHBORS (KNN)")
print("=" * 60)

# Étude de l'influence du paramètre k
k_range = range(1, 31)
train_scores_knn = []
test_scores_knn  = []

for k in k_range:
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train_scaled, y_train)
    train_scores_knn.append(knn.score(X_train_scaled, y_train))
    test_scores_knn.append(knn.score(X_test_scaled, y_test))

plt.figure(figsize=(10, 5))
plt.plot(k_range, train_scores_knn, label='Train', marker='o', color='steelblue')
plt.plot(k_range, test_scores_knn,  label='Test',  marker='s', color='salmon')
plt.xlabel("Valeur de k")
plt.ylabel("Accuracy")
plt.title("Influence du paramètre k sur KNN")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("knn_influence_k.png", dpi=150)
plt.show()
print("   → Graphique sauvegardé : knn_influence_k.png")

best_k = k_range[np.argmax(test_scores_knn)]
print(f"\n📌 Meilleur k = {best_k} → Accuracy test = {max(test_scores_knn):.4f}")

knn_best = KNeighborsClassifier(n_neighbors=best_k)
knn_best.fit(X_train_scaled, y_train)
y_pred_knn = knn_best.predict(X_test_scaled)

print(f"\n📌 Rapport de classification (KNN, k={best_k}) :")
print(classification_report(y_test, y_pred_knn, target_names=['no', 'yes']))


# PARTIE F : ARBRE DE DÉCISION + VALIDATION CROISÉE


print("\n" + "=" * 60)
print("   PARTIE F : ARBRE DE DÉCISION OPTIMISÉ (CART)")
print("=" * 60)

# GridSearchCV pour choisir max_depth et criterion
param_grid_dt = {
    'max_depth': [3, 5, 8, 10, 15, 20, None],
    'criterion': ['gini', 'entropy']
}

dt = DecisionTreeClassifier(random_state=42)
grid_dt = GridSearchCV(dt, param_grid_dt, cv=5, scoring='accuracy', n_jobs=-1)
grid_dt.fit(X_train, y_train)

print(f"\n📌 Meilleurs paramètres Arbre : {grid_dt.best_params_}")
print(f"   Accuracy validation croisée : {grid_dt.best_score_:.4f}")

dt_best = grid_dt.best_estimator_
y_pred_dt = dt_best.predict(X_test)
print(f"   Accuracy test               : {accuracy_score(y_test, y_pred_dt):.4f}")

# Visualisation de l'accuracy selon max_depth
depths = [3, 5, 8, 10, 15, 20]
cv_scores = []
for d in depths:
    dt_tmp = DecisionTreeClassifier(max_depth=d, random_state=42)
    cv_scores.append(cross_val_score(dt_tmp, X_train, y_train, cv=5).mean())

plt.figure(figsize=(8, 4))
plt.plot(depths, cv_scores, marker='o', color='darkorange')
plt.xlabel("max_depth")
plt.ylabel("Accuracy (CV)")
plt.title("Arbre de décision : Accuracy CV selon max_depth")
plt.grid(True)
plt.tight_layout()
plt.savefig("dt_cv_depth.png", dpi=150)
plt.show()
print("   → Graphique sauvegardé : dt_cv_depth.png")


# PARTIE G : BAGGING ET RANDOM FOREST


print("\n" + "=" * 60)
print("   PARTIE G : BAGGING ET RANDOM FOREST")
print("=" * 60)

# --- Bagging ---
print("\n>>> Bagging :")
B_values = [10, 50, 100, 200]
bagging_test_scores = []

for B in B_values:
    bag = RandomForestClassifier(n_estimators=B, max_features=None, random_state=42)
    bag.fit(X_train, y_train)
    bagging_test_scores.append(bag.score(X_test, y_test))
    print(f"   B={B:3d} arbres → Accuracy test : {bag.score(X_test, y_test):.4f}")

plt.figure(figsize=(8, 4))
plt.plot(B_values, bagging_test_scores, marker='o', color='purple')
plt.xlabel("Nombre d'arbres B")
plt.ylabel("Accuracy (test)")
plt.title("Bagging : Accuracy en fonction du nombre d'arbres")
plt.grid(True)
plt.tight_layout()
plt.savefig("bagging_nb_arbres.png", dpi=150)
plt.show()
print("   → Graphique sauvegardé : bagging_nb_arbres.png")

# --- Random Forest optimisé ---
print("\n>>> Random Forest optimisé :")
param_grid_rf = {
    'max_features': ['sqrt', 'log2', 0.3, 0.5]
}

rf_fixed = RandomForestClassifier(n_estimators=100, oob_score=True, random_state=42)
grid_rf = GridSearchCV(rf_fixed, param_grid_rf, cv=5, scoring='accuracy', n_jobs=-1)
grid_rf.fit(X_train, y_train)

print(f"   Meilleur paramètre max_features : {grid_rf.best_params_}")
print(f"   Accuracy validation croisée      : {grid_rf.best_score_:.4f}")

rf_best = grid_rf.best_estimator_
y_pred_rf = rf_best.predict(X_test)
print(f"   Accuracy test                    : {accuracy_score(y_test, y_pred_rf):.4f}")
print(f"   Erreur Out-Of-Bag                : {1 - rf_best.oob_score_:.4f}")


# PARTIE H : GRADIENT BOOSTING


print("\n" + "=" * 60)
print("   PARTIE H : GRADIENT BOOSTING OPTIMISÉ")
print("=" * 60)

param_grid_gb = {
    'n_estimators': [100, 200],
    'learning_rate': [0.05, 0.1, 0.2],
    'max_depth': [3, 5]
}

gb = GradientBoostingClassifier(random_state=42)
grid_gb = GridSearchCV(gb, param_grid_gb, cv=5, scoring='accuracy', n_jobs=-1, verbose=0)
grid_gb.fit(X_train, y_train)

print(f"\n📌 Meilleurs paramètres GBM : {grid_gb.best_params_}")
print(f"   Accuracy validation croisée : {grid_gb.best_score_:.4f}")

gb_best = grid_gb.best_estimator_
y_pred_gb = gb_best.predict(X_test)
print(f"   Accuracy test               : {accuracy_score(y_test, y_pred_gb):.4f}")


# PARTIE I : COMPARAISON DES MODÈLES


print("\n" + "=" * 60)
print("   PARTIE I : COMPARAISON DES MODÈLES")
print("=" * 60)

models = {
    'KNN'              : (knn_best, X_test_scaled),
    'Arbre de décision': (dt_best, X_test),
    'Random Forest'    : (rf_best, X_test),
    'Gradient Boosting': (gb_best, X_test),
}

results = {}
for name, (model, Xte) in models.items():
    y_pred = model.predict(Xte)
    acc    = accuracy_score(y_test, y_pred)
    results[name] = acc
    print(f"   {name:25s} → Accuracy : {acc:.4f}")

plt.figure(figsize=(8, 5))
bars = plt.bar(results.keys(), results.values(),
               color=['steelblue', 'darkorange', 'green', 'red'], edgecolor='black')
plt.axhline(y=dummy_score, color='gray', linestyle='--', label=f'Baseline ({dummy_score:.3f})')
plt.ylim(0.85, 0.97)
plt.ylabel("Accuracy (test)")
plt.title("Comparaison des modèles de classification")
plt.legend()
for bar, val in zip(bars, results.values()):
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001,
             f'{val:.4f}', ha='center', va='bottom', fontsize=10)
plt.tight_layout()
plt.savefig("comparaison_modeles.png", dpi=150)
plt.show()
print("   → Graphique sauvegardé : comparaison_modeles.png")


# PARTIE J : MATRICE DE CONFUSION DU MEILLEUR MODÈLE


print("\n" + "=" * 60)
print("   PARTIE J : MATRICE DE CONFUSION")
print("=" * 60)

best_model_name = max(results, key=results.get)
best_model, best_Xte = models[best_model_name]
y_pred_best = best_model.predict(best_Xte)

print(f"\n📌 Meilleur modèle : {best_model_name}")
print(classification_report(y_test, y_pred_best, target_names=['no', 'yes']))

cm = confusion_matrix(y_test, y_pred_best)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['no', 'yes'])
fig, ax = plt.subplots(figsize=(6, 5))
disp.plot(ax=ax, cmap='Blues', colorbar=False)
ax.set_title(f"Matrice de confusion – {best_model_name}")
plt.tight_layout()
plt.savefig("matrice_confusion.png", dpi=150)
plt.show()
print("   → Graphique sauvegardé : matrice_confusion.png")


# PARTIE K : COURBES ROC ET AUC


print("\n" + "=" * 60)
print("   PARTIE K : COURBES ROC ET AUC")
print("=" * 60)

plt.figure(figsize=(9, 6))

roc_models = {
    'KNN'              : (knn_best, X_test_scaled),
    'Arbre de décision': (dt_best, X_test),
    'Random Forest'    : (rf_best, X_test),
    'Gradient Boosting': (gb_best, X_test),
}

colors = ['steelblue', 'darkorange', 'green', 'red']

for (name, (model, Xte)), color in zip(roc_models.items(), colors):
    if hasattr(model, "predict_proba"):
        y_score = model.predict_proba(Xte)[:, 1]
    else:
        y_score = model.predict(Xte)
    fpr, tpr, _ = roc_curve(y_test, y_score)
    auc = roc_auc_score(y_test, y_score)
    print(f"   {name:25s} → AUC : {auc:.4f}")
    plt.plot(fpr, tpr, label=f"{name} (AUC={auc:.3f})", color=color, lw=2)

plt.plot([0, 1], [0, 1], 'k--', label='Aléatoire')
plt.xlabel("Taux de Faux Positifs (FPR)")
plt.ylabel("Taux de Vrais Positifs (TPR)")
plt.title("Courbes ROC – Comparaison des modèles")
plt.legend(loc='lower right')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("courbes_roc.png", dpi=150)
plt.show()
print("   → Graphique sauvegardé : courbes_roc.png")


# PARTIE L : IMPORTANCE DES VARIABLES (Random Forest)


print("\n" + "=" * 60)
print("   PARTIE L : IMPORTANCE DES VARIABLES")
print("=" * 60)

importances = pd.Series(rf_best.feature_importances_, index=X.columns)
top20 = importances.nlargest(20)

plt.figure(figsize=(10, 6))
top20.sort_values().plot(kind='barh', color='steelblue', edgecolor='black')
plt.title("Top 20 variables importantes (Random Forest)")
plt.xlabel("Importance")
plt.tight_layout()
plt.savefig("importance_variables.png", dpi=150)
plt.show()
print("   → Graphique sauvegardé : importance_variables.png")
print(f"\n📌 Top 5 variables importantes :\n{top20.head()}")


# PARTIE M : SAUVEGARDE DU MEILLEUR MODÈLE (.pkl)


print("\n" + "=" * 60)
print("   PARTIE M : SAUVEGARDE DU MODÈLE")
print("=" * 60)

# Sauvegarde du modèle + scaler
best_pipeline = {
    'model'       : gb_best,      # Gradient Boosting souvent meilleur
    'scaler'      : scaler,
    'feature_cols': list(X.columns)
}

with open("bank_telemarketing.pkl", "wb") as f:
    pickle.dump(best_pipeline, f)

print("\n✅ Modèle sauvegardé → bank_telemarketing.pkl")

# TEST DE RECHARGEMENT DU MODÈLE


with open("bank_telemarketing.pkl", "rb") as f:
    loaded = pickle.load(f)

model_loaded = loaded['model']
y_pred_loaded = model_loaded.predict(X_test)
print(f"✅ Modèle rechargé avec succès → Accuracy : {accuracy_score(y_test, y_pred_loaded):.4f}")


# RÉSUMÉ FINAL


print("\n" + "=" * 60)
print("   RÉSUMÉ FINAL DES PERFORMANCES")
print("=" * 60)
print(f"\n{'Modèle':<25} {'Accuracy':>10}")
print("-" * 37)
print(f"{'Baseline (constant)':<25} {dummy_score:>10.4f}")
for name, acc in results.items():
    marker = " ★" if name == best_model_name else ""
    print(f"{name:<25} {acc:>10.4f}{marker}")

print(f"\n✅ Meilleur modèle global : {best_model_name}")
print("\n📁 Fichiers générés :")
fichiers = [
    "distribution_cible.png", "pairplot_variables.png",
    "heatmap_correlation.png", "variables_categorielles.png",
    "knn_influence_k.png", "dt_cv_depth.png",
    "bagging_nb_arbres.png", "comparaison_modeles.png",
    "matrice_confusion.png", "courbes_roc.png",
    "importance_variables.png", "bank_telemarketing.pkl"
]
for f in fichiers:
    print(f"   → {f}")

print("\n" + "=" * 60)
print("   TP2 - Partie 3 terminée avec succès !")
print("=" * 60)
