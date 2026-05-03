import streamlit as st

# ---------------------------------------------------
# CONFIGURATION (DOIT ÊTRE EN PREMIER)
# ---------------------------------------------------
st.set_page_config(
    page_title="Diamonds Price App",
    page_icon="💎",
    layout="wide"          # ← wide permet d'utiliser les colonnes
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

    # Menu avec icônes dans la sidebar
    menu_options = {
        "🏠 Home":              "Home",
        "📊 Analysis":          "Analysis",
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

        st.subheader("Présentation du projet")
        st.write("""
        Cette application prédit le prix d'un diamant à partir de ses
        caractéristiques physiques et qualitatives.
        Le modèle utilisé est un **XGBoost Regressor** entraîné sur le dataset
        `diamonds.csv`.
        """)

        # ── KPIs du dataset ──────────────────────────────────────
        st.subheader("Aperçu du dataset")

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Nombre de diamants", f"{len(data):,}")
        col2.metric("Prix moyen", f"${data['price'].mean():,.0f}")
        col3.metric("Prix médian", f"${data['price'].median():,.0f}")
        col4.metric("Prix maximum", f"${data['price'].max():,}")

        st.divider()

        # ── Description des variables ─────────────────────────────
        st.subheader("Variables du dataset")

        st.dataframe(pd.DataFrame({
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
        }), use_container_width=True, hide_index=True)

    # ---------------------------------------------------
    # ANALYSIS
    # ---------------------------------------------------
    elif choice == "Analysis":

        st.subheader("Analyse du dataset")

        st.write(data.head())

        if st.checkbox("Résumé statistique"):
            st.write(data.describe())

        if st.checkbox("Valeurs manquantes"):
            missing = data.isnull().sum().reset_index()
            missing.columns = ["Variable", "Valeurs manquantes"]
            st.dataframe(missing, use_container_width=True, hide_index=True)

        if st.checkbox("Corrélation"):
            fig, ax = plt.subplots(figsize=(10, 7))
            sns.heatmap(
                data.select_dtypes(include="number").corr(),
                annot=True,
                fmt=".2f",
                ax=ax,
                cmap="coolwarm"   # ← palette plus lisible
            )
            st.pyplot(fig)
            plt.close(fig)

        # ── NOUVEAU : statistiques par catégorie ─────────────────
        st.divider()
        st.subheader("Statistiques par variable catégorielle")

        cat_choice = st.selectbox(
            "Choisir une variable catégorielle",
            ["cut", "color", "clarity"]
        )

        stat_table = (
            data.groupby(cat_choice)["price"]
            .agg(["count", "mean", "median", "min", "max"])
            .rename(columns={
                "count":  "Nb diamants",
                "mean":   "Prix moyen",
                "median": "Prix médian",
                "min":    "Prix min",
                "max":    "Prix max"
            })
            .round(0)
        )

        st.dataframe(stat_table, use_container_width=True)

    # ---------------------------------------------------
    # DATA VISUALISATION
    # ---------------------------------------------------
    elif choice == "Data Visualisation":

        st.subheader("Visualisations")

        if st.checkbox("Distribution du prix"):
            fig, ax = plt.subplots(figsize=(8, 4))
            sns.histplot(data["price"], bins=50, ax=ax, color="#b5651d")
            ax.set_title("Distribution du prix")
            st.pyplot(fig)
            plt.close(fig)

        if st.checkbox("Carat vs Price"):
            fig, ax = plt.subplots(figsize=(8, 5))
            sns.scatterplot(
                x="carat", y="price",
                data=data,
                alpha=0.2, s=10,
                ax=ax,
                color="#b5651d"
            )
            ax.set_title("Carat vs Price")
            st.pyplot(fig)
            plt.close(fig)

        if st.checkbox("Prix moyen par Cut"):
            fig, ax = plt.subplots(figsize=(7, 4))
            order = ["Fair", "Good", "Very Good", "Premium", "Ideal"]
            sns.barplot(
                x="cut", y="price",
                data=data, order=order,
                ax=ax,
                palette="flare"    # ← palette cohérente
            )
            ax.set_title("Prix moyen par qualité de taille")
            st.pyplot(fig)
            plt.close(fig)

        if st.checkbox("Prix moyen par Clarity"):
            fig, ax = plt.subplots(figsize=(8, 4))
            order = ["I1", "SI2", "SI1", "VS2", "VS1", "VVS2", "VVS1", "IF"]
            sns.barplot(
                x="clarity", y="price",
                data=data, order=order,
                ax=ax,
                palette="flare"
            )
            ax.set_title("Prix moyen par clarté")
            st.pyplot(fig)
            plt.close(fig)

        # ── NOUVEAU : Boxplot prix par Cut ────────────────────────
        if st.checkbox("Distribution prix par Cut (boxplot)"):
            order = ["Fair", "Good", "Very Good", "Premium", "Ideal"]
            fig, ax = plt.subplots(figsize=(8, 4))
            sns.boxplot(
                x="cut", y="price",
                data=data, order=order,
                ax=ax,
                palette="flare"
            )
            ax.set_title("Distribution des prix par qualité de taille")
            st.pyplot(fig)
            plt.close(fig)

    # ---------------------------------------------------
    # MACHINE LEARNING
    # ---------------------------------------------------
    elif choice == "Machine Learning":

        st.subheader("Prédiction du prix d'un diamant")

        model = load_model()

        # ── Deux colonnes : formulaire | résultat ─────────────────
        col_form, col_result = st.columns([2, 1])

        with col_form:
            with st.form("prediction_form"):

                carat   = st.slider("Carat", 0.2, 5.0, 1.0, step=0.01)
                cut     = st.selectbox("Cut",     ["Fair", "Good", "Very Good", "Premium", "Ideal"])
                color   = st.selectbox("Color",   ["J", "I", "H", "G", "F", "E", "D"])
                clarity = st.selectbox("Clarity", ["I1", "SI2", "SI1", "VS2", "VS1", "VVS2", "VVS1", "IF"])
                depth   = st.slider("Depth (%)",  43.0, 79.0, 61.5, step=0.1)
                table   = st.slider("Table (%)",  43.0, 95.0, 57.0, step=0.5)
                x       = st.slider("x (mm)",     0.0, 10.9, 4.5, step=0.01)
                y       = st.slider("y (mm)",     0.0, 10.9, 4.5, step=0.01)
                z       = st.slider("z (mm)",     0.0, 6.98, 2.8, step=0.01)

                submit = st.form_submit_button("💎 Prédire le prix")

        # Encodages ordinaux (identiques au notebook)
        cut_map     = {"Fair": 0, "Good": 1, "Very Good": 2, "Premium": 3, "Ideal": 4}
        color_map   = {"J": 0, "I": 1, "H": 2, "G": 3, "F": 4, "E": 5, "D": 6}
        clarity_map = {"I1": 0, "SI2": 1, "SI1": 2, "VS2": 3, "VS1": 4, "VVS2": 5, "VVS1": 6, "IF": 7}

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

            # ── Résultat dans la colonne de droite ────────────────
            with col_result:
                st.metric(label="Prix estimé", value=f"${prediction:,.0f}")

                # Percentile : où se situe ce diamant par rapport au dataset
                percentile = (data["price"] < prediction).mean() * 100
                st.caption(f"Ce diamant est plus cher que **{percentile:.0f}%** des diamants du dataset.")

                # Fourchette de prix réels pour des diamants similaires (±0.05 carat)
                similaires = data[
                    (data["carat"].between(carat - 0.05, carat + 0.05)) &
                    (data["cut"] == cut)
                ]["price"]

                if len(similaires) > 5:
                    st.caption(
                        f"Diamants similaires ({len(similaires)} trouvés) : "
                        f"**${similaires.min():,.0f}** – **${similaires.max():,.0f}**"
                    )

            st.divider()
            st.write("Caractéristiques saisies :")
            st.write(input_data)


# ---------------------------------------------------
# LANCEMENT
# ---------------------------------------------------
if __name__ == "__main__":
    main()
