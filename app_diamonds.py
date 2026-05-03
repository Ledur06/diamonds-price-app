import streamlit as st
import pandas as pd
import numpy as np
import pickle
import plotly.express as px
import plotly.graph_objects as go

# ---------------------------------------------------
# CONFIGURATION PAGE (TOUJOURS EN PREMIER)
# ---------------------------------------------------
st.set_page_config(
    page_title="Diamonds Price App",
    page_icon="💎",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------------------------------------------------
# STYLE CSS PRO
# ---------------------------------------------------
st.markdown("""
<style>
.main {
    background-color: #f8f9fa;
}
.block-container {
    padding-top: 2rem;
}
.metric-card {
    background: white;
    padding: 18px;
    border-radius: 14px;
    box-shadow: 0 4px 12px rgba(0,0,0,0.08);
}
.title-main {
    text-align:center;
    font-size:42px;
    font-weight:700;
    color:#6c3fc5;
}
.subtitle-main {
    text-align:center;
    font-size:18px;
    color:#555;
    margin-bottom:25px;
}
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------
# CHARGEMENT DATA
# ---------------------------------------------------
@st.cache_data
def load_data():
    df = pd.read_csv("diamonds.csv", index_col=0)
    df = df.drop_duplicates()
    df = df[(df["x"] > 0) & (df["y"] > 0) & (df["z"] > 0)]
    return df

@st.cache_resource
def load_model():
    with open("model_diamonds.pkl", "rb") as f:
        model = pickle.load(f)
    return model

df = load_data()
model = load_model()

# ---------------------------------------------------
# TITRE
# ---------------------------------------------------
st.markdown('<div class="title-main">💎 Diamonds Price App</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle-main">Prediction intelligente du prix d’un diamant avec Machine Learning</div>', unsafe_allow_html=True)

# ---------------------------------------------------
# MENU SIDEBAR
# ---------------------------------------------------
menu = st.sidebar.radio(
    "Navigation",
    ["Accueil", "Dashboard", "Visualisation", "Prédiction"]
)

# ---------------------------------------------------
# PAGE ACCUEIL
# ---------------------------------------------------
if menu == "Accueil":

    col1, col2, col3 = st.columns(3)

    col1.metric("Diamants", f"{len(df):,}")
    col2.metric("Prix moyen", f"{df['price'].mean():,.0f} $")
    col3.metric("Carat moyen", f"{df['carat'].mean():.2f}")

    st.markdown("---")

    st.subheader("À propos du projet")
    st.write("""
    Cette application utilise un modèle **XGBoost Regressor** pour estimer
    le prix d’un diamant selon ses caractéristiques physiques et qualitatives.
    
    Variables utilisées :
    - Carat
    - Cut
    - Color
    - Clarity
    - Depth
    - Table
    - x, y, z
    """)

    st.dataframe(df.head(10), use_container_width=True)

# ---------------------------------------------------
# DASHBOARD
# ---------------------------------------------------
elif menu == "Dashboard":

    st.subheader("Analyse Générale")

    col1, col2 = st.columns(2)

    with col1:
        fig = px.histogram(df, x="price", nbins=50, title="Distribution des prix")
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        fig = px.scatter(
            df.sample(3000),
            x="carat",
            y="price",
            opacity=0.5,
            title="Carat vs Prix"
        )
        st.plotly_chart(fig, use_container_width=True)

# ---------------------------------------------------
# VISUALISATION
# ---------------------------------------------------
elif menu == "Visualisation":

    st.subheader("Analyse Catégorielle")

    col1, col2 = st.columns(2)

    with col1:
        fig = px.box(
            df,
            x="cut",
            y="price",
            color="cut",
            title="Prix selon Cut"
        )
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        fig = px.box(
            df,
            x="clarity",
            y="price",
            color="clarity",
            title="Prix selon Clarity"
        )
        st.plotly_chart(fig, use_container_width=True)

# ---------------------------------------------------
# PREDICTION
# ---------------------------------------------------
elif menu == "Prédiction":

    st.subheader("Estimation du prix")

    with st.form("prediction_form"):

        col1, col2, col3 = st.columns(3)

        with col1:
            carat = st.slider("Carat", 0.2, 5.0, 1.0, 0.01)
            cut = st.selectbox("Cut", ['Fair', 'Good', 'Very Good', 'Premium', 'Ideal'])
            color = st.selectbox("Color", ['J', 'I', 'H', 'G', 'F', 'E', 'D'])

        with col2:
            clarity = st.selectbox("Clarity", ['I1', 'SI2', 'SI1', 'VS2', 'VS1', 'VVS2', 'VVS1', 'IF'])
            depth = st.slider("Depth", 43.0, 79.0, 61.5, 0.1)
            table = st.slider("Table", 43.0, 95.0, 57.0, 0.5)

        with col3:
            x = st.slider("x (mm)", 0.0, 10.9, 4.5, 0.01)
            y = st.slider("y (mm)", 0.0, 10.9, 4.5, 0.01)
            z = st.slider("z (mm)", 0.0, 6.98, 2.8, 0.01)

        submit = st.form_submit_button("Prédire")

    if submit:

        cut_map = {'Fair':0,'Good':1,'Very Good':2,'Premium':3,'Ideal':4}
        color_map = {'J':0,'I':1,'H':2,'G':3,'F':4,'E':5,'D':6}
        clarity_map = {'I1':0,'SI2':1,'SI1':2,'VS2':3,'VS1':4,'VVS2':5,'VVS1':6,'IF':7}

        X = pd.DataFrame([{
            "carat": carat,
            "cut": cut_map[cut],
            "color": color_map[color],
            "clarity": clarity_map[clarity],
            "depth": depth,
            "table": table,
            "x": x,
            "y": y,
            "z": z
        }])

        pred = model.predict(X)[0]

        st.success(f"Prix estimé : {pred:,.0f} USD")

        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=pred,
            title={'text': "Valeur estimée"},
            gauge={'axis': {'range': [0, max(20000, pred*1.2)]}}
        ))

        st.plotly_chart(fig, use_container_width=True)
