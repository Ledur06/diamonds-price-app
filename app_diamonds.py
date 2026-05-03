import streamlit as st

# ---------------------------------------------------
# CONFIGURATION (DOIT ÊTRE EN PREMIER)
# ---------------------------------------------------
st.set_page_config(
    page_title="Diamonds Price App",
    page_icon="💎",
    layout="wide"
)

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle

# ---------------------------------------------------
# CHARGEMENT DATASET
# ---------------------------------------------------
@st.cache_data
def load_data():
    data = pd.read_csv("diamonds.csv", index_col=0)
    data = data.drop_duplicates()
    data = data[(data["x"] > 0) & (data["y"] > 0) & (data["z"] > 0)]
    data = data[data["y"] < 20]
    data = data[data["z"] < 10]
    return data

# ---------------------------------------------------
# CHARGEMENT MODELE
# ---------------------------------------------------
@st.cache_resource
def load_model():
    with open("model_diamonds.pkl", "rb") as f:
        model = pickle.load(f)
    return model

# ---------------------------------------------------
# APPLICATION
# ---------------------------------------------------
def main():

    # ── En-tête global ──────────────────────────────────────────
    st.markdown(
        "<h1 style='text-align:center;color:#b5651d;'>💠 Diamonds Price App</h1>",
        unsafe_allow_html=True
    )
    st.markdown(
        "<p style='text-align:center;color:gray;font-size:16px;'>"
        "Prédiction du prix des diamants par XGBoost · Henri Ledoux</p>",
        unsafe_allow_html=True
    )
    st.divider()

    # ── Navigation sidebar ──────────────────────────────────────
    menu_options = {
        "🏠 Home":               "Home",
        "📊 Analysis":           "Analysis",
        "📈 Data Visualisation": "Data Visualisation",
        "🤖 Machine Learning":   "Machine Learning"
    }

    choice_label = st.sidebar.selectbox("Navigation", list(menu_options.keys()))
    choice = menu_options[choice_label]

    data = load_data()

    # ---------------------------------------------------
    # HOME
    # ---------------------------------------------------
    if choice == "Home":

        # ── Présentation + KPIs côte à côte ─────────────────────
        col_text, col_kpis = st.columns([3, 2], gap="large")

        with col_text:
            st.subheader("Présentation du projet")
            st.write("""
            Cette application prédit le prix d'un diamant à partir de ses
            caractéristiques physiques et qualitatives.

            Le modèle utilisé est un **XGBoost Regressor** entraîné sur le dataset
            `diamonds.csv` contenant plus de 53 000 diamants.

            **Objectif :** estimer le prix en USD d'un diamant à partir de ses 9 caractéristiques.
            """)

        with col_kpis:
            st.subheader("Aperçu du dataset")
            st.metric("Nombre de diamants", f"{len(data):,}")
            st.metric("Prix moyen",         f"${data['price'].mean():,.0f}")
            st.metric("Prix médian",        f"${data['price'].median():,.0f}")
            st.metric("Prix maximum",       f"${data['price'].max():,}")

        st.divider()

        # ── Table des variables ──────────────────────────────────
        st.subheader("Variables du dataset")

        st.dataframe(
            pd.DataFrame({
                "Variable": [
                    "carat", "cut", "color", "clarity",
                    "depth", "table", "x", "y", "z", "price"
                ],
                "Type": [
                    "Numérique", "Catégorielle", "Catégorielle",
                    "Catégorielle", "Numérique", "Numérique",
                    "Numérique", "Numérique", "Numérique", "Numérique"
                ],
                "Description": [
                    "Poids du diamant (carat)",
                    "Qualité de taille (Fair → Ideal)",
                    "Couleur (D meilleur → J)",
                    "Clarté (I1 moins bonne → IF)",
                    "Profondeur totale (%)",
                    "Largeur du dessus (%)",
                    "Longueur (mm)",
                    "Largeur (mm)",
                    "Profondeur (mm)",
                    "Prix en USD (variable cible)"
                ]
            }),
            use_container_width=True,
            hide_index=True
        )

    # ---------------------------------------------------
    # ANALYSIS
    # ---------------------------------------------------
    elif choice == "Analysis":

        st.subheader("Analyse du dataset")

        # ── Aperçu + checkboxes côte à côte ─────────────────────
        col_left, col_right = st.columns(2, gap="large")

        with col_left:
            st.markdown("**Aperçu des premières lignes**")
            st.dataframe(data.head(), use_container_width=True)

            if st.checkbox("Résumé statistique"):
                st.dataframe(data.describe(), use_container_width=True)

        with col_right:
            if st.checkbox("Valeurs manquantes"):
                missing = data.isnull().sum().reset_index()
                missing.columns = ["Variable", "Valeurs manquantes"]
                st.dataframe(missing, use_container_width=True, hide_index=True)

            if st.checkbox("Matrice de corrélation"):
                fig, ax = plt.subplots(figsize=(7, 5))
                sns.heatmap(
                    data.select_dtypes(include="number").corr(),
                    annot=True, fmt=".2f",
                    ax=ax, cmap="coolwarm"
                )
                st.pyplot(fig)
                plt.close(fig)

        st.divider()

        # ── Statistiques par catégorie ───────────────────────────
        st.subheader("Statistiques par variable catégorielle")

        cat_choice = st.selectbox(
            "Choisir une variable",
            ["cut", "color", "clarity"]
        )

        stat_table = (
            data.groupby(cat_choice)["price"]
            .agg(["count", "mean", "median", "min", "max"])
            .rename(columns={
                "count":  "Nb diamants",
                "mean":   "Prix moyen ($)",
                "median": "Prix médian ($)",
                "min":    "Prix min ($)",
                "max":    "Prix max ($)"
            })
            .round(0)
        )

        st.dataframe(stat_table, use_container_width=True)

    # ---------------------------------------------------
    # DATA VISUALISATION
    # ---------------------------------------------------
    elif choice == "Data Visualisation":

        st.subheader("Visualisations")

        # ── Ligne 1 : Distribution prix | Carat vs Price ─────────
        col1, col2 = st.columns(2, gap="large")

        with col1:
            st.markdown("**Distribution du prix**")
            fig, ax = plt.subplots(figsize=(6, 4))
            sns.histplot(data["price"], bins=50, ax=ax, color="#b5651d")
            ax.set_xlabel("Prix (USD)")
            ax.set_ylabel("Fréquence")
            st.pyplot(fig)
            plt.close(fig)

        with col2:
            st.markdown("**Carat vs Price**")
            fig, ax = plt.subplots(figsize=(6, 4))
            sns.scatterplot(
                x="carat", y="price",
                data=data, alpha=0.15, s=8,
                ax=ax, color="#b5651d"
            )
            ax.set_xlabel("Carat")
            ax.set_ylabel("Prix (USD)")
            st.pyplot(fig)
            plt.close(fig)

        st.divider()

        # ── Ligne 2 : Barplot Cut | Barplot Clarity ──────────────
        col3, col4 = st.columns(2, gap="large")

        with col3:
            st.markdown("**Prix moyen par Cut**")
            order_cut = ["Fair", "Good", "Very Good", "Premium", "Ideal"]
            fig, ax = plt.subplots(figsize=(6, 4))
            sns.barplot(
                x="cut", y="price",
                data=data, order=order_cut,
                ax=ax, palette="flare"
            )
            ax.set_xlabel("Qualité de taille")
            ax.set_ylabel("Prix moyen (USD)")
            ax.tick_params(axis="x", rotation=15)
            st.pyplot(fig)
            plt.close(fig)

        with col4:
            st.markdown("**Prix moyen par Clarity**")
            order_clarity = ["I1", "SI2", "SI1", "VS2", "VS1", "VVS2", "VVS1", "IF"]
            fig, ax = plt.subplots(figsize=(6, 4))
            sns.barplot(
                x="clarity", y="price",
                data=data, order=order_clarity,
                ax=ax, palette="flare"
            )
            ax.set_xlabel("Clarté")
            ax.set_ylabel("Prix moyen (USD)")
            ax.tick_params(axis="x", rotation=15)
            st.pyplot(fig)
            plt.close(fig)

        st.divider()

        # ── Ligne 3 : Boxplot Cut ────────────────────────────────
        st.markdown("**Distribution des prix par Cut (boxplot)**")
        fig, ax = plt.subplots(figsize=(10, 4))
        sns.boxplot(
            x="cut", y="price",
            data=data, order=order_cut,
            ax=ax, palette="flare"
        )
        ax.set_xlabel("Qualité de taille")
        ax.set_ylabel("Prix (USD)")
        st.pyplot(fig)
        plt.close(fig)

    # ---------------------------------------------------
    # MACHINE LEARNING
    # ---------------------------------------------------
    elif choice == "Machine Learning":

        st.subheader("Prédiction du prix d'un diamant")

        model = load_model()

        # Encodages ordinaux (identiques au notebook)
        cut_map     = {"Fair": 0, "Good": 1, "Very Good": 2, "Premium": 3, "Ideal": 4}
        color_map   = {"J": 0, "I": 1, "H": 2, "G": 3, "F": 4, "E": 5, "D": 6}
        clarity_map = {"I1": 0, "SI2": 1, "SI1": 2, "VS2": 3, "VS1": 4, "VVS2": 5, "VVS1": 6, "IF": 7}

        # ── Formulaire organisé en sections ─────────────────────
        with st.form("prediction_form"):

            st.markdown("#### Caractéristiques qualitatives")
            col_q1, col_q2, col_q3 = st.columns(3)
            cut     = col_q1.selectbox("Cut",     list(cut_map.keys()))
            color   = col_q2.selectbox("Color",   list(color_map.keys()))
            clarity = col_q3.selectbox("Clarity", list(clarity_map.keys()))

            st.markdown("#### Poids et proportions")
            col_p1, col_p2, col_p3 = st.columns(3)
            carat = col_p1.slider("Carat",      0.2,  5.0,  1.0,  step=0.01)
            depth = col_p2.slider("Depth (%)",  43.0, 79.0, 61.5, step=0.1)
            table = col_p3.slider("Table (%)",  43.0, 95.0, 57.0, step=0.5)

            st.markdown("#### Dimensions physiques (mm)")
            col_d1, col_d2, col_d3 = st.columns(3)
            x = col_d1.slider("x — Longueur", 0.0, 10.9, 4.5, step=0.01)
            y = col_d2.slider("y — Largeur",  0.0, 10.9, 4.5, step=0.01)
            z = col_d3.slider("z — Profondeur", 0.0, 6.98, 2.8, step=0.01)

            submit = st.form_submit_button("💎 Prédire le prix", use_container_width=True)

        # ── Résultats après soumission ───────────────────────────
        if submit:

            input_data = pd.DataFrame([{
                "carat":   carat,
                "cut":     cut_map[cut],
                "color":   color_map[color],
                "clarity": clarity_map[clarity],
                "depth":   depth,
                "table":   table,
                "x":       x,
                "y":       y,
                "z":       z
            }])

            prediction = model.predict(input_data)[0]

            st.divider()

            # ── 3 colonnes résultat ──────────────────────────────
            col_pred, col_perc, col_sim = st.columns(3, gap="large")

            with col_pred:
                st.metric("💰 Prix estimé", f"${prediction:,.0f}")

            with col_perc:
                percentile = (data["price"] < prediction).mean() * 100
                st.metric("📊 Percentile dataset", f"{percentile:.0f}%")
                st.caption("Ce diamant est plus cher que X% des diamants du dataset.")

            with col_sim:
                similaires = data[
                    (data["carat"].between(carat - 0.05, carat + 0.05)) &
                    (data["cut"] == cut)
                ]["price"]

                if len(similaires) > 5:
                    st.metric("🔍 Fourchette réelle", f"${similaires.min():,.0f} – ${similaires.max():,.0f}")
                    st.caption(f"Sur {len(similaires)} diamants similaires (±0.05 carat, même cut).")
                else:
                    st.metric("🔍 Fourchette réelle", "Données insuffisantes")
                    st.caption("Pas assez de diamants similaires dans le dataset.")

            st.divider()

            # ── Récapitulatif des caractéristiques saisies ───────
            st.markdown("**Récapitulatif des caractéristiques saisies**")
            st.dataframe(input_data, use_container_width=True, hide_index=True)


# ---------------------------------------------------
# LANCEMENT
# ---------------------------------------------------
if __name__ == "__main__":
    main()
