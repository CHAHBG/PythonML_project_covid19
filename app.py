import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pickle
import joblib
import random

# --- CONFIGURATION DE LA PAGE ---
st.set_page_config(
    page_title="COVID-19 Risk Analyst",
    page_icon="🦠",
    layout="wide"
)

# --- FONCTIONS UTILES (Cache pour la vitesse) ---
@st.cache_resource
def load_data():
    # On charge un échantillon pour la fluidité (20 000 lignes)
    df = pd.read_csv('data/covid19_data.csv', nrows=20000)

    # PETIT NETTOYAGE RAPIDE POUR L'AFFICHAGE
    # On crée la colonne DEATH proprement
    df['DEATH'] = np.where(df['DATE_DIED'] == '9999-99-99', 0, 1)

    # On remplace les codes 1/2 par Oui/Non pour les graphiques (plus joli)
    cols_oui_non = ['PNEUMONIA', 'DIABETES', 'ASTHMA', 'OBESITY', 'CARDIOVASCULAR', 'INTUBED']
    for col in cols_oui_non:
        df[col] = df[col].replace({1: 'Oui', 2: 'Non', 97: 'Inconnu', 99: 'Inconnu'})

    return df

@st.cache_resource
def load_model():
    # Priorité au modèle léger (recommandé pour Streamlit Cloud)
    try:
        return joblib.load('model_covid_rf.joblib')
    except Exception:
        pass
    # Fallback local (ancien format)
    return pickle.load(open('mon_modele_covid.pkl', 'rb'))

def map_oui_non(text):
    return 1 if text == "Oui" else 2


def get_resources():
    """Charge (une seule fois par session) les données + le modèle avec une UI de progression."""
    if "_resources_ready" in st.session_state:
        return load_data(), load_model()

    tips = [
        "Les valeurs 97/99 dans ce dataset représentent souvent des données manquantes.",
        "Le rappel (recall) est crucial si on veut détecter un maximum de cas à risque.",
        "Un Random Forest combine plusieurs arbres pour réduire le surapprentissage.",
        "L'âge est l'un des facteurs les plus corrélés au risque de décès dans l'analyse exploratoire.",
    ]

    holder = st.empty()
    with holder.container():
        st.markdown("## ⏳ Chargement en cours")
        st.info(f"💡 Le saviez-vous ? {random.choice(tips)}")

        status = st.status("Initialisation…", expanded=True)
        progress = st.progress(0)

        status.write("1/3 Lecture et préparation des données…")
        progress.progress(10)
        with st.spinner("Lecture du fichier CSV…"):
            df = load_data()
        progress.progress(65)

        status.write("2/3 Chargement du modèle de prédiction…")
        with st.spinner("Chargement du modèle…"):
            model = load_model()
        progress.progress(90)

        status.write("3/3 Finalisation…")
        progress.progress(100)
        status.update(label="Prêt ✅", state="complete", expanded=False)

    holder.empty()
    st.session_state["_resources_ready"] = True
    return df, model

# --- NAVIGATION (SIDEBAR) ---
st.sidebar.image("https://cdn-icons-png.flaticon.com/512/2785/2785819.png", width=100)
st.sidebar.title("Navigation")
page = st.sidebar.radio("Aller vers :", ["🏠 Accueil", "📊 Exploration Visuelle", "🔮 Prédiction IA"])

st.sidebar.markdown("---")
st.sidebar.info(
    "Projet Data Science\n"
    "\n"
    "Concepteurs :\n"
    "- Cheikh Ahmadou Bamba Gningue\n"
    "- Koffi Grâce Amandine\n"
    "- Jean Paul Ildevert Malan\n"
    "- Diomade Loua"
)

# ==========================================
# PAGE 1 : ACCUEIL
# ==========================================
if page == "🏠 Accueil":
    st.title("🦠 Analyse des Risques COVID-19")
    st.markdown("### Bienvenue sur l'interface de prédiction médicale.")

    st.markdown(
        "**Concepteurs :**  "+
        "Cheikh Ahmadou Bamba Gningue • Koffi Grâce Amandine • Jean Paul Ildevert Malan • Diomade Loua"
    )

    st.success("👈 Commencez par explorer les données via le menu à gauche, ou passez directement à la prédiction.")

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
        **Ce projet permet de :**
        * 📊 **Visualiser** les facteurs aggravants du virus.
        * 🤖 **Prédire** le risque de décès grâce à une IA (Random Forest).
        * 🏥 **Aider** à la prise de décision médicale.
        """)
    with col2:
        # Tu peux mettre une image d'illustration ici
        st.write(" ")

# ==========================================
# PAGE 2 : EXPLORATION VISUELLE (EDA)
# ==========================================
elif page == "📊 Exploration Visuelle":
    st.title("🔎 Exploration des Données")

    df, _model = get_resources()

    # 1. LES CHIFFRES CLÉS (KPIs)
    st.subheader("📌 Vue d'ensemble")
    col1, col2, col3 = st.columns(3)

    nb_patients = len(df)
    nb_deces = df['DEATH'].sum()
    taux_mortalite = (nb_deces / nb_patients) * 100

    col1.metric("Patients Analysés", f"{nb_patients:,}")
    col2.metric("Nombre de Décès", f"{nb_deces:,}")
    col3.metric("Taux de Mortalité (Échantillon)", f"{taux_mortalite:.1f}%", delta_color="inverse")

    st.markdown("---")

    # 2. LES GRAPHIQUES
    st.subheader("📈 Visualisation des Facteurs de Risque")

    tab1, tab2, tab3 = st.tabs(["💀 Mortalité Globale", "🏥 Maladies & Risques", "🎂 Impact de l'Âge"])

    # --- ONGLET 1 : CAMEMBERT ---
    with tab1:
        st.write("Répartition des issues (Décès vs Guérison) dans notre jeu de données.")

        fig, ax = plt.subplots()
        df['DEATH_LABEL'] = df['DEATH'].replace({0: 'Survivant', 1: 'Décédé'})
        counts = df['DEATH_LABEL'].value_counts()

        ax.pie(counts, labels=counts.index, autopct='%1.1f%%', colors=['#66b3ff', '#ff9999'], startangle=90)
        ax.axis('equal')  # Pour que le camembert soit rond
        st.pyplot(fig)
        st.caption("Ce graphique montre la proportion de décès dans l'échantillon observé.")

    # --- ONGLET 2 : INTERACTIF (BARPLOT) ---
    with tab2:
        st.write("Quel est l'impact des comorbidités sur le décès ?")

        # Le sélecteur interactif
        option = st.selectbox("Choisissez une maladie à analyser :", 
                              ['PNEUMONIA', 'DIABETES', 'ASTHMA', 'OBESITY', 'CARDIOVASCULAR', 'INTUBED'])

        st.write(f"Comparaison des décès pour : **{option}**")

        fig2, ax2 = plt.subplots(figsize=(8, 5))
        # On compare le taux de décès selon Oui ou Non
        sns.barplot(x=option, y='DEATH', data=df, palette="viridis", ax=ax2, order=['Non', 'Oui'])
        ax2.set_ylabel("Probabilité de Décès")
        ax2.set_title(f"Risque de décès selon : {option}")

        st.pyplot(fig2)
        st.info(f"💡 Analyse : Si la barre 'Oui' est plus haute, c'est que **{option}** augmente le risque.")

    # --- ONGLET 3 : DISTRIBUTION (HISTPLOT) ---
    with tab3:
        st.write("Distribution de l'âge des patients décédés vs survivants.")

        fig3, ax3 = plt.subplots(figsize=(10, 6))
        sns.histplot(data=df, x='AGE', hue='DEATH_LABEL', kde=True, element="step", palette={'Survivant': 'blue', 'Décédé': 'red'}, ax=ax3)
        ax3.set_title("L'âge est-il un facteur déterminant ?")
        st.pyplot(fig3)
        st.warning("⚠️ On observe clairement que la courbe rouge (Décès) est décalée vers les âges avancés.")

# ==========================================
# PAGE 3 : PRÉDICTION IA
# ==========================================
elif page == "🔮 Prédiction IA":
    st.title("🤖 Diagnostic Intelligent")
    st.markdown("Remplissez le dossier médical du patient. L'IA calculera ses chances de survie.")

    _df, model = get_resources()

    with st.form("form_prediction"):
        col1, col2 = st.columns(2)

        with col1:
            age = st.slider("Âge", 0, 110, 45)
            sexe = st.radio("Sexe", ["Femme", "Homme"])
            hospital = st.radio("Hospitalisé ?", ["Non", "Oui"])
            intubation = st.selectbox("Intubation nécessaire ?", ["Non", "Oui"])
            pneumonie = st.selectbox("Pneumonie ?", ["Non", "Oui"])

        with col2:
            st.write("**Comorbidités**")
            diabete = st.checkbox("Diabète")
            bpco = st.checkbox("BPCO (Poumons)")
            asthme = st.checkbox("Asthme")
            immu = st.checkbox("Immunosupprimé")
            hyper = st.checkbox("Hypertension")
            cardio = st.checkbox("Maladie Cardiovasculaire")
            obesite = st.checkbox("Obésité")
            rein = st.checkbox("Insuffisance Rénale")
            tabac = st.checkbox("Fumeur")

        submit = st.form_submit_button("🩺 Lancer le Diagnostic")

    if submit:
        # Mapping des variables (A ajuster selon ton X_train exact !)
        # Ici j'utilise une logique standard
        # 1=Oui, 2=Non (Standard Dataset COVID)
        def to_code(bool_val): return 1 if bool_val else 2

        features = [
            1 if hospital == "Non" else 2, # USMER (Hypothèse)
            12, # MEDICAL UNIT
            1 if sexe == "Femme" else 2,
            1 if hospital == "Non" else 2, # PATIENT_TYPE (1=Home, 2=Hopital)
            to_code(intubation == "Oui"), # INTUBED
            to_code(pneumonie == "Oui"), # PNEUMONIA
            age,
            2, # PREGNANT
            to_code(diabete),
            to_code(bpco),
            to_code(asthme),
            to_code(immu),
            to_code(hyper),
            to_code(cardio), # OTHER DISEASE
            to_code(cardio),
            to_code(obesite),
            to_code(rein),
            to_code(tabac),
            7, # CLASSIF
            2 # ICU
        ]

        # Le modèle a été entraîné avec des noms de colonnes (DataFrame)
        feature_names = [
            'USMER', 'MEDICAL_UNIT', 'SEX', 'PATIENT_TYPE', 'INTUBED',
            'PNEUMONIA', 'AGE', 'PREGNANT', 'DIABETES', 'COPD', 'ASTHMA',
            'INMSUPR', 'HIPERTENSION', 'OTHER_DISEASE', 'CARDIOVASCULAR',
            'OBESITY', 'RENAL_CHRONIC', 'TOBACCO', 'CLASIFFICATION_FINAL', 'ICU'
        ]
        features_df = pd.DataFrame([features], columns=feature_names)

        try:
            prediction = model.predict(features_df)
            proba = model.predict_proba(features_df)
            risque = proba[0][1] * 100

            st.divider()
            if risque > 50:
                st.error(f"🔴 RISQUE ÉLEVÉ : {risque:.1f}% de probabilité de décès.")
                st.progress(int(risque))
            else:
                st.success(f"🟢 RISQUE FAIBLE : {risque:.1f}% de probabilité de décès.")
                st.progress(int(risque))

        except Exception as e:
            st.error("Erreur de format des données. Vérifiez le nombre de colonnes.")
