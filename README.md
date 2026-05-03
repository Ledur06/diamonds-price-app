# 💠 Diamonds Price App

> Application de prédiction du prix des diamants par Machine Learning  
> **Henri Ledoux Same** · Projet académique — Licence 3 Management et Technique Quantitative

---

## 📌 Présentation

Cette application web prédit le **prix d'un diamant en USD** à partir de ses caractéristiques physiques et qualitatives.  
Elle est construite avec **Streamlit** et repose sur un modèle **XGBoost Regressor** entraîné sur le dataset `diamonds.csv` (53 940 diamants).

🔗 **Application en ligne :** [diamonds-price-app.streamlit.app](https://diamonds-price-app-7iofuaynib87hqf2hy2vo2.streamlit.app)

---

## 🗂️ Structure du projet

```
diamonds-price-app/
│
├── app_diamonds.py        # Application Streamlit principale
├── model_diamonds.pkl     # Modèle XGBoost entraîné (sérialisé)
├── diamonds.csv           # Dataset source (53 940 diamants)
├── requirements.txt       # Dépendances Python
└── README.md              # Ce fichier
```

---

## 📊 Dataset

Le dataset `diamonds.csv` provient de la librairie **ggplot2** (R) et contient 53 940 diamants décrits par 10 variables.

| Variable  | Type         | Description                              |
|-----------|--------------|------------------------------------------|
| `carat`   | Numérique    | Poids du diamant (carat)                 |
| `cut`     | Catégorielle | Qualité de taille (Fair → Ideal)         |
| `color`   | Catégorielle | Couleur (D = meilleur → J)               |
| `clarity` | Catégorielle | Clarté (I1 = moins bonne → IF)           |
| `depth`   | Numérique    | Profondeur totale (%)                    |
| `table`   | Numérique    | Largeur du dessus (%)                    |
| `x`       | Numérique    | Longueur (mm)                            |
| `y`       | Numérique    | Largeur (mm)                             |
| `z`       | Numérique    | Profondeur (mm)                          |
| `price`   | Numérique    | **Prix en USD (variable cible)**         |

---

## 🤖 Modèle

- **Algorithme :** XGBoost Regressor
- **Encodage des variables catégorielles :** ordinal (cut, color, clarity)
- **Séparation :** 80% train / 20% test
- **Métriques d'évaluation :** RMSE, MAE, R²

### Encodages ordinaux appliqués

| Variable  | Encodage                                              |
|-----------|-------------------------------------------------------|
| `cut`     | Fair=0, Good=1, Very Good=2, Premium=3, Ideal=4      |
| `color`   | J=0, I=1, H=2, G=3, F=4, E=5, D=6                   |
| `clarity` | I1=0, SI2=1, SI1=2, VS2=3, VS1=4, VVS2=5, VVS1=6, IF=7 |

---

## 📱 Pages de l'application

### 🏠 Home
- Présentation du projet
- Métriques clés du dataset (nombre de diamants, prix moyen, médian, maximum)
- Description de toutes les variables

### 📊 Analysis
- Aperçu des données et résumé statistique
- Détection des valeurs manquantes
- Matrice de corrélation
- Statistiques par variable catégorielle (cut, color, clarity)

### 📈 Data Visualisation
- Distribution du prix
- Relation Carat vs Price
- Prix moyen par qualité de taille (Cut)
- Prix moyen par clarté (Clarity)
- Boxplot de la distribution des prix par Cut

### 🤖 Machine Learning
- Formulaire de saisie organisé en 3 sections (qualité, proportions, dimensions)
- Prédiction du prix en USD
- Percentile du diamant dans le dataset
- Fourchette de prix réels pour des diamants similaires

---

## ⚙️ Installation locale

### 1. Cloner le dépôt

```bash
git clone https://github.com/ton-pseudo/diamonds-price-app.git
cd diamonds-price-app
```

### 2. Installer les dépendances

```bash
pip install -r requirements.txt
```

### 3. Lancer l'application

```bash
streamlit run app_diamonds.py
```

L'application s'ouvre automatiquement sur `http://localhost:8501`

---

## 📦 Dépendances

```
streamlit
pandas
numpy
matplotlib
seaborn
xgboost
scikit-learn
```

---

## 🚀 Déploiement

L'application est déployée sur **Streamlit Community Cloud**.

1. Dépôt GitHub public avec les 4 fichiers nécessaires
2. Connexion sur [share.streamlit.io](https://share.streamlit.io)
3. Sélection du dépôt et du fichier `app_diamonds.py`
4. Déploiement automatique

---

## 👤 Auteur

**Henri Ledoux Same**  
Étudiant en Licence 3 Management et Technique Quantitative
  
Projet académique — Machine Learning avec XGBoost

---

## 📄 Licence

Projet académique — usage éducatif uniquement.
