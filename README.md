# 🦠 COVID-19 Analytics — Dashboard Streamlit & IA

## 👥 Concepteurs (Équipe Projet)
- **Cheikh A. B. GNINGUE**
- **Jean Paul I. MALAN**
- **Grace KOFFI**
- **Loua F. DIOMANDE**

## 🎯 Objectif du projet
Construire une application **Streamlit** interactive permettant :
- d’explorer un jeu de données COVID-19 (profil patient, comorbidités, prise en charge),
- de visualiser des indicateurs (répartition survivants/décédés, tendances, distributions),
- d’estimer un risque via un modèle de **Machine Learning**.

## 🧩 Fonctionnalités
- **Navigation multi-pages** : Accueil, Exploration Intuitive, Diagnostic IA, Méthodologie.
- **Graphiques Plotly** interactifs.
- **Chargement optimisé** : cache Streamlit pour accélérer l’exécution.

## 📊 Données & variable cible (DEATH)
Le fichier de données attendu est : `data/covid19_data.csv`.

### Définition des décès
Le dataset utilise la convention suivante :
- `DATE_DIED = '9999-99-99'` ⟶ patient **non décédé**
- toute autre valeur de `DATE_DIED` ⟶ patient **décédé** (date réelle)

Dans l’application, la cible binaire est calculée ainsi :

	df['DEATH'] = (df['DATE_DIED'] != '9999-99-99').astype(int)

### Courbe temporelle (Décès) : éviter les biais
Le nettoyage strict (`dropna()`) utilisé pour certaines analyses (notamment comorbidités) peut supprimer beaucoup de lignes si des variables sont manquantes.

Pour éviter une “chute artificielle” de la courbe, la **tendance temporelle des décès** est calculée à partir d’un sous-jeu minimal (`AGE`, `SEX`, `PATIENT_TYPE`, `DATE_DIED`) via `load_time_data()`.

## 🗂️ Structure du projet

	.
	├── app.py
	├── requirements.txt
	├── model_covid_rf.joblib
	├── data/
	│   └── covid19_data.csv
	├── GUIDE_INSTALLATION.md
	└── EXPLICATION_CODE.md

## ⚙️ Installation (local)
Prérequis : **Python 3.8+**.

### 1) Créer un environnement virtuel
macOS / Linux :

	python3 -m venv .venv
	source .venv/bin/activate

Windows (PowerShell) :

	python -m venv .venv
	.\.venv\Scripts\Activate.ps1

### 2) Installer les dépendances

	pip install -r requirements.txt

## ▶️ Lancer l’application

	streamlit run app.py

Puis ouvrir l’URL locale indiquée (souvent `http://localhost:8501`).

## ☁️ Déploiement (Streamlit Cloud)
1. Pousser le projet sur GitHub.
2. Sur Streamlit Cloud : **New app** ⟶ sélectionner le repo ⟶ `app.py` ⟶ Deploy.

Notes :
- Vérifier que `requirements.txt` est bien à la racine.
- Le modèle attendu par défaut est `model_covid_rf.joblib`.

## 📚 Documentation
- `GUIDE_INSTALLATION.md` : guide pas à pas d’installation et lancement.
- `EXPLICATION_CODE.md` : explication détaillée du code (`app.py`).
