# =============================================================================
# DASHBOARD STREAMLIT - Bank Telemarketing Prediction
# TP2 Partie 3 - Licence MTQ S6 - ISJM
# =============================================================================

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (accuracy_score, confusion_matrix, classification_report,
                              roc_curve, roc_auc_score)
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.dummy import DummyClassifier

# =============================================================================
# CONFIGURATION DE LA PAGE
# =============================================================================

st.set_page_config(
    page_title="Bank Telemarketing - ML Dashboard",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS personnalisé
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=DM+Sans:wght@300;400;500&display=swap');

    html, body, [class*="css"] {
        font-family: 'DM Sans', sans-serif;
    }

    .main-title {
        font-family: 'Syne', sans-serif;
        font-size: 2.8rem;
        font-weight: 800;
        color: #0f172a;
        letter-spacing: -1px;
        line-height: 1.1;
    }

    .subtitle {
        font-family: 'DM Sans', sans-serif;
        font-size: 1.1rem;
        color: #64748b;
        margin-top: 0.3rem;
    }

    .metric-card {
        background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%);
        border-radius: 16px;
        padding: 1.5rem;
        color: white;
        border: 1px solid #334155;
    }

    .metric-value {
        font-family: 'Syne', sans-serif;
        font-size: 2.2rem;
        font-weight: 700;
        color: #38bdf8;
    }

    .metric-label {
        font-size: 0.85rem;
        color: #94a3b8;
        text-transform: uppercase;
        letter-spacing: 1px;
    }

    .section-title {
        font-family: 'Syne', sans-serif;
        font-size: 1.4rem;
        font-weight: 700;
        color: #0f172a;
        border-left: 4px solid #38bdf8;
        padding-left: 0.8rem;
        margin: 1.5rem 0 1rem 0;
    }

    .stAlert {
        border-radius: 12px;
    }

    .prediction-yes {
        background: linear-gradient(135deg, #065f46, #047857);
        color: white;
        padding: 1.5rem;
        border-radius: 16px;
        text-align: center;
        font-family: 'Syne', sans-serif;
        font-size: 1.5rem;
        font-weight: 700;
    }

    .prediction-no {
        background: linear-gradient(135deg, #7f1d1d, #991b1b);
        color: white;
        padding: 1.5rem;
        border-radius: 16px;
        text-align: center;
        font-family: 'Syne', sans-serif;
        font-size: 1.5rem;
        font-weight: 700;
    }

    .stSidebar {
        background: #0f172a;
    }

    div[data-testid="metric-container"] {
        background: #f8fafc;
        border: 1px solid #e2e8f0;
        border-radius: 12px;
        padding: 1rem;
    }
</style>
""", unsafe_allow_html=True)


# =============================================================================
# CHARGEMENT ET PRÉPARATION DES DONNÉES (mis en cache)
# =============================================================================

@st.cache_data
def load_and_prepare_data():
    df = pd.read_csv("bank-additional.csv", sep=";")
    df['y_encoded'] = (df['y'] == 'yes').astype(int)
    df_encoded = pd.get_dummies(df.drop(columns=['y', 'y_encoded']), drop_first=True)
    X = df_encoded
    y = df['y_encoded']
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_test_sc  = scaler.transform(X_test)
    return df, X, y, X_train, X_test, y_train, y_test, X_train_sc, X_test_sc, scaler


@st.cache_resource
def train_models(X_train, X_test, y_train, y_test, X_train_sc, X_test_sc):
    models = {}

    # KNN
    knn = KNeighborsClassifier(n_neighbors=15)
    knn.fit(X_train_sc, y_train)
    models['KNN'] = (knn, X_test_sc, knn.score(X_test_sc, y_test))

    # Decision Tree
    dt = DecisionTreeClassifier(max_depth=8, criterion='gini', random_state=42)
    dt.fit(X_train, y_train)
    models['Arbre de décision'] = (dt, X_test, dt.score(X_test, y_test))

    # Random Forest
    rf = RandomForestClassifier(n_estimators=100, max_features='sqrt',
                                 oob_score=True, random_state=42)
    rf.fit(X_train, y_train)
    models['Random Forest'] = (rf, X_test, rf.score(X_test, y_test))

    # Gradient Boosting
    gb = GradientBoostingClassifier(n_estimators=200, learning_rate=0.1,
                                     max_depth=3, random_state=42)
    gb.fit(X_train, y_train)
    models['Gradient Boosting'] = (gb, X_test, gb.score(X_test, y_test))

    return models


# =============================================================================
# CHARGEMENT
# =============================================================================

try:
    df, X, y, X_train, X_test, y_train, y_test, X_train_sc, X_test_sc, scaler = load_and_prepare_data()
    models = train_models(X_train, X_test, y_train, y_test, X_train_sc, X_test_sc)
    data_loaded = True
except FileNotFoundError:
    data_loaded = False


# =============================================================================
# SIDEBAR
# =============================================================================

with st.sidebar:
    st.markdown("""
    <div style='text-align:center; padding: 1rem 0;'>
        <div style='font-family: Syne, sans-serif; font-size: 1.5rem; font-weight: 800; color: #38bdf8;'>🏦 BankML</div>
        <div style='color: #94a3b8; font-size: 0.8rem; margin-top: 0.3rem;'>Dashboard TP2 - ISJM 2025/2026</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")

    page = st.radio(
        "📌 Navigation",
        ["🏠 Accueil", "📊 Exploration des données", "🤖 Modèles & Performances", "🔮 Prédiction en temps réel"],
        label_visibility="collapsed"
    )

    st.markdown("---")
    st.markdown("""
    <div style='color: #64748b; font-size: 0.75rem; text-align: center;'>
        Licence MTQ 3ème année (S6)<br>
        Introduction à l'IA — TP2 Partie 3
    </div>
    """, unsafe_allow_html=True)


# =============================================================================
# PAGE 1 : ACCUEIL
# =============================================================================

if page == "🏠 Accueil":

    st.markdown('<div class="main-title">🏦 Bank Telemarketing<br>ML Dashboard</div>', unsafe_allow_html=True)
    st.markdown('<div class="subtitle">Prédiction de souscription bancaire par Machine Learning — TP2 Partie 3</div>', unsafe_allow_html=True)
    st.markdown("---")

    if not data_loaded:
        st.error("⚠️ Fichier `bank-additional.csv` introuvable. Place-le dans le même dossier que l'application.")
        st.stop()

    # Métriques clés
    st.markdown('<div class="section-title">📈 Aperçu du Dataset</div>', unsafe_allow_html=True)
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("📋 Instances", f"{df.shape[0]:,}")
    with col2:
        st.metric("🔢 Variables", df.shape[1] - 1)
    with col3:
        pct_yes = (df['y'] == 'yes').mean() * 100
        st.metric("✅ Souscriptions", f"{pct_yes:.1f}%")
    with col4:
        best_acc = max(v[2] for v in models.values())
        st.metric("🏆 Meilleure Accuracy", f"{best_acc:.3f}")

    st.markdown("---")

    col_left, col_right = st.columns(2)

    with col_left:
        st.markdown('<div class="section-title">🎯 Objectif</div>', unsafe_allow_html=True)
        st.info("""
        **Classification Binaire**

        Prédire si un client va **souscrire** à un dépôt à terme bancaire (`yes` / `no`)
        suite à une campagne de marketing téléphonique.

        **Variable cible :** `y` → `yes` (1) ou `no` (0)
        """)

        st.markdown('<div class="section-title">📚 Modèles utilisés</div>', unsafe_allow_html=True)
        for name, (_, _, acc) in models.items():
            st.markdown(f"- **{name}** → Accuracy : `{acc:.4f}`")

    with col_right:
        st.markdown('<div class="section-title">📊 Distribution de la cible</div>', unsafe_allow_html=True)
        fig, ax = plt.subplots(figsize=(6, 4))
        counts = df['y'].value_counts()
        colors = ['#ef4444', '#22c55e']
        bars = ax.bar(counts.index, counts.values, color=colors, edgecolor='white', linewidth=2, width=0.5)
        ax.set_title("Répartition des classes", fontsize=13, fontweight='bold', pad=15)
        ax.set_ylabel("Nombre de clients")
        for bar, val in zip(bars, counts.values):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 30,
                    f'{val}\n({val/len(df)*100:.1f}%)', ha='center', fontsize=10, fontweight='bold')
        ax.spines[['top', 'right']].set_visible(False)
        ax.set_facecolor('#f8fafc')
        fig.patch.set_facecolor('#f8fafc')
        st.pyplot(fig)
        plt.close()


# =============================================================================
# PAGE 2 : EXPLORATION DES DONNÉES
# =============================================================================

elif page == "📊 Exploration des données":

    if not data_loaded:
        st.error("⚠️ Fichier `bank-additional.csv` introuvable.")
        st.stop()

    st.markdown('<div class="main-title">📊 Exploration des données</div>', unsafe_allow_html=True)
    st.markdown("---")

    tab1, tab2, tab3 = st.tabs(["📋 Aperçu", "📈 Distributions", "🔗 Corrélations"])

    # --- TAB 1 : Aperçu ---
    with tab1:
        col1, col2 = st.columns(2)
        with col1:
            st.markdown('<div class="section-title">Premières lignes</div>', unsafe_allow_html=True)
            st.dataframe(df.head(10), use_container_width=True)
        with col2:
            st.markdown('<div class="section-title">Statistiques descriptives</div>', unsafe_allow_html=True)
            st.dataframe(df.describe().round(2), use_container_width=True)

        st.markdown('<div class="section-title">Types des variables</div>', unsafe_allow_html=True)
        type_df = pd.DataFrame({
            'Variable': df.columns,
            'Type': df.dtypes.astype(str),
            'Valeurs uniques': [df[c].nunique() for c in df.columns],
            'Valeurs manquantes': [df[c].isnull().sum() for c in df.columns]
        })
        st.dataframe(type_df, use_container_width=True)

    # --- TAB 2 : Distributions ---
    with tab2:
        st.markdown('<div class="section-title">Distribution d\'une variable numérique</div>', unsafe_allow_html=True)
        num_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
        selected_col = st.selectbox("Choisir une variable :", num_cols)

        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        fig.patch.set_facecolor('#f8fafc')

        # Histogramme
        for val, color in zip(['no', 'yes'], ['#ef4444', '#22c55e']):
            axes[0].hist(df[df['y'] == val][selected_col],
                         alpha=0.7, label=val, color=color, bins=30, edgecolor='white')
        axes[0].set_title(f"Distribution de '{selected_col}' par classe", fontweight='bold')
        axes[0].legend()
        axes[0].spines[['top', 'right']].set_visible(False)
        axes[0].set_facecolor('#f8fafc')

        # Boxplot
        df.boxplot(column=selected_col, by='y', ax=axes[1],
                   patch_artist=True,
                   boxprops=dict(facecolor='#38bdf8', alpha=0.7))
        axes[1].set_title(f"Boxplot de '{selected_col}'", fontweight='bold')
        axes[1].set_xlabel("Souscription (y)")
        plt.suptitle("")
        axes[1].spines[['top', 'right']].set_visible(False)
        axes[1].set_facecolor('#f8fafc')

        st.pyplot(fig)
        plt.close()

        st.markdown('<div class="section-title">Variable catégorielle vs Cible</div>', unsafe_allow_html=True)
        cat_cols = df.select_dtypes(include='object').columns.drop('y').tolist()
        selected_cat = st.selectbox("Choisir une variable catégorielle :", cat_cols)

        fig2, ax2 = plt.subplots(figsize=(10, 4))
        fig2.patch.set_facecolor('#f8fafc')
        ct = df.groupby([selected_cat, 'y']).size().unstack()
        ct.plot(kind='bar', ax=ax2, color=['#ef4444', '#22c55e'], edgecolor='white', width=0.7)
        ax2.set_title(f"'{selected_cat}' selon la souscription", fontweight='bold')
        ax2.set_xlabel("")
        ax2.tick_params(axis='x', rotation=30)
        ax2.legend(title='Souscription')
        ax2.spines[['top', 'right']].set_visible(False)
        ax2.set_facecolor('#f8fafc')
        st.pyplot(fig2)
        plt.close()

    # --- TAB 3 : Corrélations ---
    with tab3:
        st.markdown('<div class="section-title">Matrice de corrélation</div>', unsafe_allow_html=True)
        num_df = df.select_dtypes(include=['int64', 'float64'])
        fig3, ax3 = plt.subplots(figsize=(10, 7))
        fig3.patch.set_facecolor('#f8fafc')
        sns.heatmap(num_df.corr(), annot=True, fmt=".2f", cmap="RdYlBu_r",
                    linewidths=0.5, ax=ax3, center=0,
                    cbar_kws={"shrink": 0.8})
        ax3.set_title("Corrélation entre variables numériques", fontweight='bold', fontsize=13)
        st.pyplot(fig3)
        plt.close()


# =============================================================================
# PAGE 3 : MODÈLES & PERFORMANCES
# =============================================================================

elif page == "🤖 Modèles & Performances":

    if not data_loaded:
        st.error("⚠️ Fichier `bank-additional.csv` introuvable.")
        st.stop()

    st.markdown('<div class="main-title">🤖 Modèles & Performances</div>', unsafe_allow_html=True)
    st.markdown("---")

    tab1, tab2, tab3 = st.tabs(["📊 Comparaison", "📉 Courbes ROC", "🌳 Importance des variables"])

    # --- TAB 1 : Comparaison ---
    with tab1:
        st.markdown('<div class="section-title">Comparaison des modèles</div>', unsafe_allow_html=True)

        results_data = []
        for name, (model, Xte, acc) in models.items():
            y_pred = model.predict(Xte)
            report = classification_report(y_test, y_pred, output_dict=True)
            results_data.append({
                'Modèle': name,
                'Accuracy': round(acc, 4),
                'Précision (yes)': round(report['1']['precision'], 4),
                'Rappel (yes)': round(report['1']['recall'], 4),
                'F1-Score (yes)': round(report['1']['f1-score'], 4),
            })

        results_df = pd.DataFrame(results_data).set_index('Modèle')
        st.dataframe(results_df.style.highlight_max(color='#bbf7d0', axis=0), use_container_width=True)

        fig, ax = plt.subplots(figsize=(10, 5))
        fig.patch.set_facecolor('#f8fafc')
        names = list(results_df.index)
        accs  = results_df['Accuracy'].values
        colors_bar = ['#38bdf8', '#f97316', '#22c55e', '#a855f7']
        bars = ax.bar(names, accs, color=colors_bar, edgecolor='white', linewidth=2, width=0.5)
        ax.set_ylim(0.85, 0.97)
        ax.axhline(y=0.887, color='gray', linestyle='--', linewidth=1.5, label='Baseline')
        ax.set_ylabel("Accuracy", fontsize=12)
        ax.set_title("Comparaison des modèles par Accuracy", fontweight='bold', fontsize=13)
        for bar, val in zip(bars, accs):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001,
                    f'{val:.4f}', ha='center', fontsize=11, fontweight='bold')
        ax.spines[['top', 'right']].set_visible(False)
        ax.set_facecolor('#f8fafc')
        ax.legend()
        st.pyplot(fig)
        plt.close()

        # Matrice de confusion du modèle sélectionné
        st.markdown('<div class="section-title">Matrice de confusion</div>', unsafe_allow_html=True)
        selected_model_name = st.selectbox("Choisir un modèle :", list(models.keys()))
        model_sel, Xte_sel, _ = models[selected_model_name]
        y_pred_sel = model_sel.predict(Xte_sel)
        cm = confusion_matrix(y_test, y_pred_sel)

        fig2, ax2 = plt.subplots(figsize=(6, 4))
        fig2.patch.set_facecolor('#f8fafc')
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax2,
                    xticklabels=['no', 'yes'], yticklabels=['no', 'yes'],
                    linewidths=2, linecolor='white', cbar=False,
                    annot_kws={'size': 16, 'weight': 'bold'})
        ax2.set_xlabel("Prédit", fontsize=12)
        ax2.set_ylabel("Réel", fontsize=12)
        ax2.set_title(f"Matrice de confusion — {selected_model_name}", fontweight='bold')
        st.pyplot(fig2)
        plt.close()

    # --- TAB 2 : Courbes ROC ---
    with tab2:
        st.markdown('<div class="section-title">Courbes ROC & AUC</div>', unsafe_allow_html=True)

        fig3, ax3 = plt.subplots(figsize=(9, 6))
        fig3.patch.set_facecolor('#f8fafc')
        ax3.set_facecolor('#f8fafc')
        colors_roc = ['#38bdf8', '#f97316', '#22c55e', '#a855f7']

        for (name, (model, Xte, _)), color in zip(models.items(), colors_roc):
            if hasattr(model, "predict_proba"):
                y_score = model.predict_proba(Xte)[:, 1]
            else:
                y_score = model.predict(Xte)
            fpr, tpr, _ = roc_curve(y_test, y_score)
            auc = roc_auc_score(y_test, y_score)
            ax3.plot(fpr, tpr, label=f"{name} (AUC = {auc:.3f})", color=color, lw=2.5)

        ax3.plot([0, 1], [0, 1], 'k--', alpha=0.5, label='Aléatoire')
        ax3.fill_between([0, 1], [0, 1], alpha=0.05, color='gray')
        ax3.set_xlabel("Taux de Faux Positifs (FPR)", fontsize=12)
        ax3.set_ylabel("Taux de Vrais Positifs (TPR)", fontsize=12)
        ax3.set_title("Courbes ROC — Comparaison des modèles", fontweight='bold', fontsize=13)
        ax3.legend(loc='lower right', fontsize=10)
        ax3.spines[['top', 'right']].set_visible(False)
        ax3.grid(True, alpha=0.3)
        st.pyplot(fig3)
        plt.close()

    # --- TAB 3 : Importance des variables ---
    with tab3:
        st.markdown('<div class="section-title">Importance des variables — Random Forest</div>', unsafe_allow_html=True)

        rf_model = models['Random Forest'][0]
        importances = pd.Series(rf_model.feature_importances_, index=X.columns)
        top_n = st.slider("Nombre de variables à afficher :", 5, 25, 15)
        top_imp = importances.nlargest(top_n).sort_values()

        fig4, ax4 = plt.subplots(figsize=(10, max(4, top_n * 0.35)))
        fig4.patch.set_facecolor('#f8fafc')
        colors_imp = plt.cm.Blues(np.linspace(0.4, 0.9, len(top_imp)))
        top_imp.plot(kind='barh', ax=ax4, color=colors_imp, edgecolor='white')
        ax4.set_title(f"Top {top_n} variables importantes", fontweight='bold', fontsize=13)
        ax4.set_xlabel("Importance", fontsize=11)
        ax4.spines[['top', 'right']].set_visible(False)
        ax4.set_facecolor('#f8fafc')
        st.pyplot(fig4)
        plt.close()


# =============================================================================
# PAGE 4 : PRÉDICTION EN TEMPS RÉEL
# =============================================================================

elif page == "🔮 Prédiction en temps réel":

    if not data_loaded:
        st.error("⚠️ Fichier `bank-additional.csv` introuvable.")
        st.stop()

    st.markdown('<div class="main-title">🔮 Prédiction en temps réel</div>', unsafe_allow_html=True)
    st.markdown('<div class="subtitle">Remplis les informations d\'un client pour prédire sa réponse</div>', unsafe_allow_html=True)
    st.markdown("---")

    col_form, col_result = st.columns([1.2, 1])

    with col_form:
        st.markdown('<div class="section-title">📝 Informations du client</div>', unsafe_allow_html=True)

        c1, c2 = st.columns(2)
        with c1:
            age        = st.number_input("Âge", 18, 95, 35)
            job        = st.selectbox("Profession", df['job'].unique())
            marital    = st.selectbox("Statut marital", df['marital'].unique())
            education  = st.selectbox("Éducation", df['education'].unique())
            default    = st.selectbox("Défaut de crédit", df['default'].unique())
        with c2:
            housing    = st.selectbox("Prêt immobilier", df['housing'].unique())
            loan       = st.selectbox("Prêt personnel", df['loan'].unique())
            contact    = st.selectbox("Type de contact", df['contact'].unique())
            month      = st.selectbox("Mois", df['month'].unique())
            day_of_week= st.selectbox("Jour", df['day_of_week'].unique())

        c3, c4 = st.columns(2)
        with c3:
            duration   = st.number_input("Durée appel (sec)", 0, 5000, 200)
            campaign   = st.number_input("Nb contacts campagne", 1, 50, 2)
            pdays      = st.number_input("Jours depuis dernier contact", 0, 999, 999)
        with c4:
            previous   = st.number_input("Nb contacts précédents", 0, 20, 0)
            poutcome   = st.selectbox("Résultat campagne précédente", df['poutcome'].unique())
            emp_var    = st.number_input("Taux variation emploi", -3.5, 1.5, -1.8, step=0.1)

        c5, c6 = st.columns(2)
        with c5:
            cons_price = st.number_input("Indice prix consommation", 92.0, 95.0, 93.0, step=0.1)
            cons_conf  = st.number_input("Indice confiance consommation", -51.0, -26.0, -40.0, step=0.5)
        with c6:
            euribor    = st.number_input("Euribor 3 mois", 0.6, 5.1, 4.0, step=0.1)
            nr_employed= st.number_input("Nb employés (milliers)", 4963.0, 5228.0, 5099.0, step=1.0)

        model_choice = st.selectbox("🤖 Modèle de prédiction", list(models.keys()))
        predict_btn  = st.button("🔮 Prédire", type="primary", use_container_width=True)

    with col_result:
        st.markdown('<div class="section-title">📊 Résultat</div>', unsafe_allow_html=True)

        if predict_btn:
            # Construire le dictionnaire du client
            client_dict = {
                'age': age, 'job': job, 'marital': marital, 'education': education,
                'default': default, 'housing': housing, 'loan': loan,
                'contact': contact, 'month': month, 'day_of_week': day_of_week,
                'duration': duration, 'campaign': campaign, 'pdays': pdays,
                'previous': previous, 'poutcome': poutcome,
                'emp.var.rate': emp_var, 'cons.price.idx': cons_price,
                'cons.conf.idx': cons_conf, 'euribor3m': euribor,
                'nr.employed': nr_employed
            }

            client_df = pd.DataFrame([client_dict])
            client_encoded = pd.get_dummies(client_df, drop_first=True)
            client_encoded = client_encoded.reindex(columns=X.columns, fill_value=0)

            model_used, Xte_used, _ = models[model_choice]

            # Normaliser si KNN
            if model_choice == 'KNN':
                client_input = scaler.transform(client_encoded)
            else:
                client_input = client_encoded

            prediction = model_used.predict(client_input)[0]
            proba      = model_used.predict_proba(client_input)[0]

            # Affichage du résultat
            if prediction == 1:
                st.markdown("""
                <div class="prediction-yes">
                    ✅ SOUSCRIPTION<br>
                    <span style='font-size:1rem; font-weight:400;'>Le client va probablement souscrire</span>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown("""
                <div class="prediction-no">
                    ❌ PAS DE SOUSCRIPTION<br>
                    <span style='font-size:1rem; font-weight:400;'>Le client ne va probablement pas souscrire</span>
                </div>
                """, unsafe_allow_html=True)

            st.markdown("---")
            st.markdown("**Probabilités :**")
            col_p1, col_p2 = st.columns(2)
            with col_p1:
                st.metric("❌ Non (no)", f"{proba[0]*100:.1f}%")
            with col_p2:
                st.metric("✅ Oui (yes)", f"{proba[1]*100:.1f}%")

            # Jauge de probabilité
            fig_gauge, ax_gauge = plt.subplots(figsize=(5, 2.5))
            fig_gauge.patch.set_facecolor('#f8fafc')
            ax_gauge.barh([''], [proba[0]], color='#ef4444', label='No', height=0.4)
            ax_gauge.barh([''], [proba[1]], left=[proba[0]], color='#22c55e', label='Yes', height=0.4)
            ax_gauge.set_xlim(0, 1)
            ax_gauge.set_title("Répartition des probabilités", fontweight='bold')
            ax_gauge.legend(loc='upper right')
            ax_gauge.spines[['top', 'right', 'left']].set_visible(False)
            ax_gauge.set_facecolor('#f8fafc')
            st.pyplot(fig_gauge)
            plt.close()

            st.info(f"**Modèle utilisé :** {model_choice}")

        else:
            st.markdown("""
            <div style='text-align:center; padding: 3rem 1rem; color: #94a3b8;'>
                <div style='font-size: 3rem;'>🔮</div>
                <div style='font-size: 1rem; margin-top: 1rem;'>
                    Remplis le formulaire et clique sur<br><strong>Prédire</strong> pour voir le résultat
                </div>
            </div>
            """, unsafe_allow_html=True)
