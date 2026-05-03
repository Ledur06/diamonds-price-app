import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle

# Chargement du dataset (mis en cache pour ne pas recharger à chaque interaction)
@st.cache_data
def load_data():
    data = pd.read_csv('diamonds.csv', index_col=0)
    data = data.drop_duplicates()
    data = data[(data['x'] > 0) & (data['y'] > 0) & (data['z'] > 0)]
    data = data[data['y'] < 20]
    data = data[data['z'] < 10]
    return data

# Chargement du modèle (mis en cache pour ne charger qu'une seule fois)
@st.cache_resource
def load_model():
    with open('model_diamonds.pkl', 'rb') as f:
        model = pickle.load(f)
    return model

# Titre et mise en page
st.set_page_config(page_title="Prédicteur du prix des diamants", page_icon="💎", layout="centered")


def main():
    st.markdown("<h1 style='text-align:center;color:brown;'>💠Diamonds Price App</h1>", unsafe_allow_html=True)
    st.markdown("<h2 style='text-align:center;color:black;'>Prédiction du prix des diamants</h2>", unsafe_allow_html=True)

    menu = ['Home', 'Analysis', 'Data Visualisation', 'Machine Learning']
    choice = st.sidebar.selectbox('Select Menu', menu)

    data = load_data()

    # ─── HOME ────────────────────────────────────────────────────
    if choice == 'Home':
        st.subheader('Présentation du projet')
        st.write("""
            Cette application prédit le prix d'un diamant à partir de ses caractéristiques
            physiques et qualitatives. Le modèle utilisé est un **XGBoost Regressor**
            entraîné sur le dataset `diamonds.csv` (53 940 diamants).
        """)

        st.subheader('Variables du dataset')
        st.write(pd.DataFrame({
            'Variable': ['carat', 'cut', 'color', 'clarity', 'depth', 'table', 'x', 'y', 'z', 'price'],
            'Type': ['Numérique', 'Catégorielle', 'Catégorielle', 'Catégorielle',
                     'Numérique', 'Numérique', 'Numérique', 'Numérique', 'Numérique', 'Numérique'],
            'Description': [
                'Poids du diamant (carat)',
                'Qualité de taille (Fair → Ideal)',
                'Couleur (D=meilleur → J)',
                'Clarté (I1=moins bonne → IF)',
                'Profondeur totale (%)',
                'Largeur du dessus (%)',
                'Longueur (mm)', 'Largeur (mm)', 'Profondeur (mm)',
                'Prix en USD (variable cible)'
            ]
        }))

    # ─── ANALYSIS ────────────────────────────────────────────────
    elif choice == 'Analysis':
        st.subheader('Dataset Diamonds')
        st.write(data.head())

        if st.checkbox('Résumé statistique'):
            st.write(data.describe())

        if st.checkbox('Valeurs manquantes'):
            st.write(data.isnull().sum())

        if st.checkbox('Corrélation'):
            fig = plt.figure(figsize=(10, 7))
            sns.heatmap(data.select_dtypes(include='number').corr(), annot=True, fmt='.2f')
            st.pyplot(fig)

    # ─── DATA VISUALISATION ──────────────────────────────────────
    elif choice == 'Data Visualisation':

        if st.checkbox('Distribution du prix'):
            fig = plt.figure(figsize=(8, 4))
            sns.histplot(data['price'], bins=50)
            plt.title('Distribution du prix')
            st.pyplot(fig)

        if st.checkbox('Carat vs Price'):
            fig = plt.figure(figsize=(8, 5))
            sns.scatterplot(x='carat', y='price', data=data, alpha=0.2, s=10)
            plt.title('Carat vs Price')
            st.pyplot(fig)

        if st.checkbox('Prix moyen par Cut'):
            fig = plt.figure(figsize=(7, 4))
            order = ['Fair', 'Good', 'Very Good', 'Premium', 'Ideal']
            sns.barplot(x='cut', y='price', data=data, order=order)
            plt.title('Prix moyen par qualité de taille')
            st.pyplot(fig)

        if st.checkbox('Prix moyen par Clarity'):
            fig = plt.figure(figsize=(8, 4))
            order = ['I1', 'SI2', 'SI1', 'VS2', 'VS1', 'VVS2', 'VVS1', 'IF']
            sns.barplot(x='clarity', y='price', data=data, order=order)
            plt.title('Prix moyen par clarté')
            st.pyplot(fig)

    # ─── MACHINE LEARNING ────────────────────────────────────────
    elif choice == 'Machine Learning':
        st.subheader('Prédiction du prix d\'un diamant')

        model = load_model()

        # Saisie des caractéristiques dans la sidebar
        st.sidebar.subheader('Caractéristiques du diamant')

        carat   = st.sidebar.slider('Carat', 0.2, 5.0, 1.0, step=0.01)
        cut     = st.sidebar.selectbox('Cut', ['Fair', 'Good', 'Very Good', 'Premium', 'Ideal'])
        color   = st.sidebar.selectbox('Color', ['J', 'I', 'H', 'G', 'F', 'E', 'D'])
        clarity = st.sidebar.selectbox('Clarity', ['I1', 'SI2', 'SI1', 'VS2', 'VS1', 'VVS2', 'VVS1', 'IF'])
        depth   = st.sidebar.slider('Depth (%)', 43.0, 79.0, 61.5, step=0.1)
        table   = st.sidebar.slider('Table (%)', 43.0, 95.0, 57.0, step=0.5)
        x       = st.sidebar.slider('x (mm)', 0.0, 10.9, 4.5, step=0.01)
        y       = st.sidebar.slider('y (mm)', 0.0, 10.9, 4.5, step=0.01)
        z       = st.sidebar.slider('z (mm)', 0.0, 6.98, 2.8, step=0.01)

        # Encodage ordinal — même encodage que dans le notebook
        cut_map     = {'Fair': 0, 'Good': 1, 'Very Good': 2, 'Premium': 3, 'Ideal': 4}
        color_map   = {'J': 0, 'I': 1, 'H': 2, 'G': 3, 'F': 4, 'E': 5, 'D': 6}
        clarity_map = {'I1': 0, 'SI2': 1, 'SI1': 2, 'VS2': 3, 'VS1': 4, 'VVS2': 5, 'VVS1': 6, 'IF': 7}

        # Construction du dataframe de prédiction
        input_data = pd.DataFrame([{
            'carat':   carat,
            'cut':     cut_map[cut],
            'color':   color_map[color],
            'clarity': clarity_map[clarity],
            'depth':   depth,
            'table':   table,
            'x':       x,
            'y':       y,
            'z':       z
        }])

        st.write('Caractéristiques saisies :')
        st.write(input_data)

        if st.button('Prédire le prix'):
            prediction = model.predict(input_data)[0]
            st.success(f'Prix estimé : **{prediction:,.0f} USD**')


if __name__ == '__main__':
    main()
