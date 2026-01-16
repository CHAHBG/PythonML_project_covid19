# 🚀 Guide d'Installation : Projet COVID-19 Analytics

Ce guide explique comment installer et lancer l'application Streamlit sur un nouvel ordinateur.

## 1. Prérequis
Assurez-vous d'avoir installé :
*   **Python** (Version 3.8 ou plus récente). [Télécharger ici](https://www.python.org/downloads/)
*   **Git** (Optionnel, pour cloner le projet).

## 2. Structure des Dossiers
Pour que l'application fonctionne, votre dossier doit respecter cette structure exacte :

```
/MonDossierProjet
│
├── app.py                  # Le code principal de l'application
├── requirements.txt        # La liste des bibliothèques à installer
├── mon_modele_covid.pkl    # Le modèle IA entraîné (Random Forest)
│
└── data
    └── covid19_data.csv    # Le fichier de données
```

> **Note :** Si le dossier `data` n'existe pas, créez-le et placez-y votre fichier CSV.

## 3. Installation

Ouvrez votre terminal (Invite de commande sur Windows ou Terminal sur Mac/Linux) et suivez ces étapes :

### Étape 1 : Se placer dans le dossier du projet
```bash
cd chemin/vers/MonDossierProjet
```
*(Remplacez `chemin/vers/MonDossierProjet` par le vrai chemin de votre dossier)*

### Étape 2 : Créer un environnement virtuel (Recommandé)
Cela évite les conflits avec d'autres projets.

*   **Sur Windows :**
    ```bash
    python -m venv venv
    .\venv\Scripts\activate
    ```

*   **Sur Mac/Linux :**
    ```bash
    python3 -m venv venv
    source venv/bin/activate
    ```

### Étape 3 : Installer les bibliothèques
Une fois l'environnement activé (vous devriez voir `(venv)` au début de la ligne de commande), lancez :

```bash
pip install -r requirements.txt
```

Cela installera automatiquement : `streamlit`, `pandas`, `numpy`, `plotly`, `scikit-learn`, `matplotlib`, `seaborn`.

## 4. Lancement de l'Application

Une fois l'installation terminée, lancez l'application avec la commande suivante :

```bash
streamlit run app.py
```

Votre navigateur internet devrait s'ouvrir automatiquement à l'adresse `http://localhost:8501`.

## 5. Dépannage
*   **Erreur `FileNotFoundError` :** Vérifiez que `covid19_data.csv` est bien dans le sous-dossier `data`.
*   **Erreur de modèle :** Assurez-vous que `mon_modele_covid.pkl` est bien présent à côté de `app.py` et que vous avez installé `scikit-learn`.

## 6. Mettre l'application sur Internet (Cloud) ☁️

Tu veux que tout le monde puisse utiliser ton application sans rien installer ? Utilisons **Streamlit Cloud** (c'est gratuit et facile).

### Étape 1 : Mettre le code sur GitHub
1.  Crée un compte sur [GitHub.com](https://github.com/).
2.  Crée un nouveau "Repository" (Projet) nommé `covid-app`.
3.  Upload (télécharge) tes fichiers dedans :
    *   `app.py`
    *   `requirements.txt`
    *   `mon_modele_covid.pkl`
    *   Le dossier `data` (avec le fichier csv dedans).

### Étape 2 : Connecter Streamlit Cloud
1.  Va sur [share.streamlit.io](https://share.streamlit.io/).
2.  Connecte-toi avec ton compte GitHub.
3.  Clique sur **"New app"**.
4.  Choisis ton projet `covid-app` dans la liste.
5.  Clique sur **"Deploy!"**.

🚀 **C'est fini !** Streamlit va installer tout seul tes bibliothèques et te donnera un lien (URL) que tu pourras envoyer à tes amis ou ton professeur.

