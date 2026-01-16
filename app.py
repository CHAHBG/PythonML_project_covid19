import streamlit as st
import pandas as pd
import numpy as np
import pickle
import joblib
import time
import random
import plotly.express as px
from pathlib import Path

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="COVID-19 Analytics",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- MODERN DARK THEME CSS ---
st.markdown("""
<style>
    /* Global App Background */
    .stApp {
        background-color: #0E1117;
        color: #FAFAFA;
    }
    
    /* Card Styling */
    .metric-card {
        background-color: #262730;
        border: 1px solid #41444C;
        padding: 20px;
        border-radius: 12px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.3);
        margin-bottom: 20px;
        text-align: center;
        transition: transform 0.2s;
    }
    .metric-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 6px 12px rgba(0,0,0,0.5);
    }
    
    .metric-value {
        font-size: 2.2rem;
        font-weight: 700;
        color: #FFFFFF;
        margin-bottom: 5px;
    }
    .metric-label {
        font-size: 1rem;
        color: #A3A8B8;
        font-weight: 500;
    }
    
    /* Team Card Styling */
    .team-card {
        background-color: #1E1E1E;
        padding: 15px;
        border-radius: 10px;
        border-left: 5px solid #EF553B;
        margin-bottom: 10px;
    }
    
    /* Headers */
    h1, h2, h3, h4 {
        color: #FFFFFF !important;
        font-family: 'Inter', sans-serif;
    }
    
    /* Custom Sidebar */
    [data-testid="stSidebar"] {
        background-color: #171923;
    }
</style>
""", unsafe_allow_html=True)

# --- DATA LOADING ---
@st.cache_data(ttl="2h")
def load_raw_preview(nrows: int = 20_000) -> pd.DataFrame:
    """Aperçu des données brutes pour l'affichage uniquement (léger)."""
    try:
        df_raw = pd.read_csv('data/covid19_data.csv', nrows=nrows)
    except Exception:
        return pd.DataFrame()

    map_dict = {1: 'Oui', 2: 'Non', 97: 'Inconnu', 99: 'Inconnu'}
    cols_to_map = [
        'PNEUMONIA', 'DIABETES', 'COPD', 'ASTHMA', 'INMSUPR',
        'HIPERTENSION', 'OTHER_DISEASE', 'CARDIOVASCULAR', 'OBESITY',
        'RENAL_CHRONIC', 'TOBACCO', 'INTUBED', 'ICU'
    ]
    for col in cols_to_map:
        if col in df_raw.columns:
            df_raw[f'{col}_LABEL'] = df_raw[col].map(map_dict).fillna('Inconnu')

    if 'SEX' in df_raw.columns:
        df_raw['SEX_LABEL'] = df_raw['SEX'].map({1: 'Femme', 2: 'Homme'})
    if 'PATIENT_TYPE' in df_raw.columns:
        df_raw['PATIENT_TYPE_LABEL'] = df_raw['PATIENT_TYPE'].map({1: 'Domicile', 2: 'Hospitalisation'})
    if 'DATE_DIED' in df_raw.columns:
        df_raw['DEATH'] = np.where(df_raw['DATE_DIED'] == '9999-99-99', 0, 1)
        df_raw['DEATH_LABEL'] = df_raw['DEATH'].map({0: 'Survivant', 1: 'Décédé'})
    return df_raw


@st.cache_data(ttl="2h")
def load_clean_data() -> pd.DataFrame:
    """Charge TOUT le CSV et produit le dataset nettoyé, en minimisant RAM/temps."""
    usecols = [
        'USMER', 'MEDICAL_UNIT', 'SEX', 'PATIENT_TYPE', 'INTUBED', 'PNEUMONIA',
        'AGE', 'PREGNANT', 'DIABETES', 'COPD', 'ASTHMA', 'INMSUPR',
        'HIPERTENSION', 'OTHER_DISEASE', 'CARDIOVASCULAR', 'OBESITY',
        'RENAL_CHRONIC', 'TOBACCO', 'CLASIFFICATION_FINAL', 'ICU', 'DATE_DIED'
    ]

    try:
        try:
            df_clean = pd.read_csv('data/covid19_data.csv', engine='pyarrow', usecols=usecols)
        except Exception:
            df_clean = pd.read_csv('data/covid19_data.csv', usecols=usecols, low_memory=False)
    except Exception:
        st.error("Erreur : Le fichier 'data/covid19_data.csv' est introuvable.")
        return pd.DataFrame()

    # Normalisation Oui/Non
    cols_to_normalize = [c for c in df_clean.columns if c not in ['AGE', 'DATE_DIED', 'MEDICAL_UNIT', 'CLASIFFICATION_FINAL']]
    df_clean[cols_to_normalize] = df_clean[cols_to_normalize].replace({2.0: 0, 1.0: 1})

    # Remplacer 97/98/99 par NaN
    cols_to_correct = [c for c in df_clean.columns if c not in ['AGE', 'DATE_DIED']]
    df_clean[cols_to_correct] = df_clean[cols_to_correct].replace({97: np.nan, 98: np.nan, 99: np.nan})

    # Règles métier
    df_clean.loc[df_clean['SEX'] == 0, 'PREGNANT'] = 0
    for col in ['ICU', 'INTUBED']:
        df_clean.loc[df_clean['PATIENT_TYPE'] == 1, col] = df_clean.loc[df_clean['PATIENT_TYPE'] == 1, col].fillna(0)

    # Drop NaN (nettoyage strict)
    df_clean = df_clean.dropna()

    # Target
    df_clean['DEATH'] = (df_clean['DATE_DIED'] != '9999-99-99').astype(int)

    # Labels nécessaires à l'app (évite de dupliquer trop de colonnes)
    df_clean['SEX_LABEL'] = df_clean['SEX'].map({1: 'Femme', 0: 'Homme'})
    df_clean['PATIENT_TYPE_LABEL'] = df_clean['PATIENT_TYPE'].map({1: 'Domicile', 0: 'Hospitalisation'})
    df_clean['DEATH_LABEL'] = df_clean['DEATH'].map({0: 'Survivant', 1: 'Décédé'})
    df_clean['ICU_LABEL'] = df_clean['ICU'].map({1: 'Oui', 0: 'Non'})

    # Colonnes label pour les graphiques (Comorbidités, etc.)
    binary_map = {1: 'Oui', 0: 'Non'}
    cols_to_label = [
        'PNEUMONIA', 'DIABETES', 'COPD', 'ASTHMA', 'INMSUPR',
        'HIPERTENSION', 'OTHER_DISEASE', 'CARDIOVASCULAR', 'OBESITY',
        'RENAL_CHRONIC', 'TOBACCO', 'INTUBED', 'ICU'
    ]
    for col in cols_to_label:
        if col in df_clean.columns:
            # Catégorie = beaucoup plus léger qu'une colonne string classique
            df_clean[f'{col}_LABEL'] = pd.Categorical(df_clean[col].map(binary_map), categories=['Non', 'Oui'])

    # Mois (rapide) + filtre de validité YYYY-MM pour éviter des axes "mélangés"
    died = df_clean['DATE_DIED'].where(df_clean['DATE_DIED'] != '9999-99-99', pd.NA)
    mois = died.astype('string').str.slice(0, 7)
    df_clean['MOIS'] = mois.where(mois.str.match(r'^\d{4}-\d{2}$', na=False))

    return df_clean


@st.cache_data(ttl="2h")
def load_time_data() -> pd.DataFrame:
    """Dataset minimal pour la tendance temporelle (évite le biais du dropna strict)."""
    usecols = ['AGE', 'SEX', 'PATIENT_TYPE', 'DATE_DIED']
    try:
        try:
            df_time = pd.read_csv('data/covid19_data.csv', engine='pyarrow', usecols=usecols)
        except Exception:
            df_time = pd.read_csv('data/covid19_data.csv', usecols=usecols, low_memory=False)
    except Exception:
        return pd.DataFrame()

    df_time['DEATH'] = (df_time['DATE_DIED'] != '9999-99-99').astype(int)
    df_time['SEX_LABEL'] = df_time['SEX'].map({1: 'Femme', 2: 'Homme', 0: 'Homme'})
    df_time['PATIENT_TYPE_LABEL'] = df_time['PATIENT_TYPE'].map({1: 'Domicile', 2: 'Hospitalisation', 0: 'Hospitalisation'})
    return df_time

# --- MODEL MANAGEMENT ---
@st.cache_resource
def get_model(_df_clean):
    # En production (Streamlit Cloud) on évite absolument le ré-entraînement au runtime.
    joblib_path = Path('model_covid_rf.joblib')
    if joblib_path.exists():
        return joblib.load(joblib_path)

    # Fallback local uniquement (on ne versionne pas le pickle sur Cloud)
    pickle_path = Path('mon_modele_covid.pkl')
    if pickle_path.exists():
        # Garde-fou: si le pickle est énorme, il a tendance à faire crasher le Cloud
        if pickle_path.stat().st_size <= 150 * 1024 * 1024:
            return pickle.load(open(pickle_path, 'rb'))

    st.error("❌ Modèle introuvable. Assurez-vous que 'model_covid_rf.joblib' est présent dans le dépôt.")
    return None

# --- INIT LOADING & SPLASH SCREEN ---
# Astuces éducatives pendant le chargement
medical_facts = [
    "💡 Le saviez-vous ? Le lavage des mains réduit de 50% la transmission des infections respiratoires.",
    "🧬 Analyse : Les modèles Random Forest combinent des centaines d'arbres de décision pour plus de précision.",
    "⚠️ Facteur : L'âge est le facteur de risque le plus significatif dans notre base de données.",
    "📊 Données : Nous analysons plus de 500 000 dossiers patients anonymisés.",
    "🫁 Info : La pneumonie est une complication majeure surveillée par notre algorithme.",
    "🤖 IA : Ce modèle apprend des motifs complexes invisibles à l'œil nu."
]

loader_placeholder = st.empty()

# Si les données ne sont pas chargées, on affiche l'animation
if 'data_loaded' not in st.session_state:
    fact = random.choice(medical_facts)
    loader_placeholder.markdown(f"""
    <style>
        .loader-container {{
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: center;
            height: 400px;
            background-color: #0E1117;
            border-radius: 15px;
            border: 1px solid #262730;
            padding: 20px;
            text-align: center;
        }}
        .virus-loader {{
            width: 80px;
            height: 80px;
            background-color: #EF553B;
            border-radius: 50%;
            position: relative;
            animation: pulse-virus 1.5s infinite ease-in-out;
            box-shadow: 0 0 20px #EF553B;
            margin-bottom: 30px;
        }}
        .virus-loader::after {{
            content: '';
            position: absolute;
            top: -10px; left: -10px; right: -10px; bottom: -10px;
            border: 4px solid #00CC96;
            border-radius: 50%;
            border-top-color: transparent;
            animation: spin 1s linear infinite;
        }}
        @keyframes pulse-virus {{
            0% {{ transform: scale(0.95); opacity: 0.8; }}
            50% {{ transform: scale(1.05); opacity: 1; }}
            100% {{ transform: scale(0.95); opacity: 0.8; }}
        }}
        @keyframes spin {{
            0% {{ transform: rotate(0deg); }}
            100% {{ transform: rotate(360deg); }}
        }}
        .loading-text {{
            color: #FAFAFA;
            font-size: 24px;
            font-weight: bold;
            font-family: 'Inter', sans-serif;
            margin-bottom: 10px;
        }}
        .fact-text {{
            color: #00CC96;
            font-style: italic;
            font-size: 16px;
            max-width: 600px;
        }}
    </style>
    <div class="loader-container">
        <div class="virus-loader"></div>
        <div class="loading-text">Initialisation du Système d'IA...</div>
        <div class="fact-text">{fact}</div>
    </div>
    """, unsafe_allow_html=True)
    
    # Chargement réel
    df_raw = load_raw_preview()
    df = load_clean_data()
    df_time_all = load_time_data()
    model = get_model(df)
    
    # Validation du chargement
    st.session_state['data_loaded'] = True
    loader_placeholder.empty()

else:
    # Si déjà chargé, on récupère directement (le cache gère la rapidité)
    df_raw = load_raw_preview()
    df = load_clean_data()
    df_time_all = load_time_data()
    model = get_model(df)

# --- SIDEBAR & NAVIGATION ---
st.sidebar.title("Navigation")
page = st.sidebar.radio("Aller vers :", 
    ["Accueil", "Exploration Intuitive", "Diagnostic IA", "Méthodologie & Insights"]
)

st.sidebar.markdown("---")
st.sidebar.markdown("### 👥 Équipe Projet")
st.sidebar.markdown("""
* **Cheikh A. B. GNINGUE**
* **Jean Paul I. MALAN**
* **Grace KOFFI**
* **Loua F. DIOMANDE**
""")

st.sidebar.markdown("---")
# --- GLOBAL FILTERS ---
df_filtered = df.copy()

if page == "Exploration Intuitive" and not df.empty:
    st.sidebar.header("🔍 Filtres Globaux")
    age_range = st.sidebar.slider("Tranche d'âge", int(df['AGE'].min()), int(df['AGE'].max()), (0, 100))
    gender_opts = df['SEX_LABEL'].dropna().unique()
    gender_sel = st.sidebar.multiselect("Genre", gender_opts, default=gender_opts)
    pt_opts = df['PATIENT_TYPE_LABEL'].dropna().unique()
    pt_sel = st.sidebar.multiselect("Type de Prise en Charge", pt_opts, default=pt_opts)
    
    # Filtre sécurisé
    mask = (df['AGE'] >= age_range[0]) & (df['AGE'] <= age_range[1])
    if gender_sel:
        mask &= df['SEX_LABEL'].isin(gender_sel)
    if pt_sel:
        mask &= df['PATIENT_TYPE_LABEL'].isin(pt_sel)
        
    df_filtered = df[mask]
    st.sidebar.markdown(f"**{len(df_filtered):,} patients sélectionnés**")

# --- PAGE 1: ACCUEIL ---
if page == "Accueil":
    st.title("🦠 Plateforme Analytics COVID-19")
    st.markdown("### 🏥 Tableau de Bord Stratégique")
    
    # 1. TEAM SECTION
    st.markdown("---")
    st.subheader("👥 L'Équipe de Réalisation")
    
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.info("**Cheikh Ahmadou Bamba GNINGUE**")
    with c2:
        st.info("**Jean Paul Ildevert MALAN**")
    with c3:
        st.info("**Grace KOFFI**")
    with c4:
        st.info("**Loua Franck DIOMANDE**")
    
    # 2. METRICS
    if not df.empty:
        st.markdown("---")
        c1, c2, c3, c4 = st.columns(4)
        total = len(df)
        deaths = df['DEATH'].sum()
        rate = (deaths/total)*100 if total > 0 else 0
        avg_age = df['AGE'].mean()
        
        with c1:
            st.markdown(f'<div class="metric-card"><div class="metric-label">Patients Analysés</div><div class="metric-value">{total:,}</div></div>', unsafe_allow_html=True)
        with c2:
            st.markdown(f'<div class="metric-card"><div class="metric-label">Décès Confirmés</div><div class="metric-value">{deaths:,}</div></div>', unsafe_allow_html=True)
        with c3:
            st.markdown(f'<div class="metric-card"><div class="metric-label">Taux de Mortalité</div><div class="metric-value">{rate:.2f}%</div></div>', unsafe_allow_html=True)
        with c4:
            st.markdown(f'<div class="metric-card"><div class="metric-label">Âge Moyen</div><div class="metric-value">{avg_age:.1f} ans</div></div>', unsafe_allow_html=True)
            
    st.markdown("---")
    
    st.subheader("📂 Aperçu des Données")
    st.write("Le jeu de données a été rigoureusement nettoyé pour garantir la précision des analyses.")
    
    tab_raw, tab_clean = st.tabs(["📄 Données Brutes", "✨ Données Normalisées & Nettoyées"])
    
    with tab_raw:
        st.info("Données brutes avec valeurs manquantes.")
        st.dataframe(df_raw.head(50), width='stretch')
        
    with tab_clean:
        st.success("Données nettoyées et normalisées (0/1).")
        st.dataframe(df.head(50), width='stretch')

# --- PAGE 2: EXPLORATION INTUITIVE ---
elif page == "Exploration Intuitive":
    st.title("📊 Exploration Dynamique")
    
    if df_filtered.empty:
        st.warning("Aucune donnée disponible avec les filtres actuels.")
    else:
        col_L, col_R = st.columns(2)
        
        with col_L:
            st.markdown("#### 🍩 Répartition des Issues")
            counts = df_filtered['DEATH_LABEL'].value_counts().reset_index()
            counts.columns = ['Statut', 'Nombre']
            
            fig1 = px.pie(counts, values='Nombre', names='Statut', hole=0.5, 
                          color='Statut', color_discrete_map={'Survivant':'#00CC96', 'Décédé':'#EF553B'})
            fig1.update_traces(textinfo='percent+label')
            fig1.update_layout(showlegend=False, margin=dict(t=0, b=0, l=0, r=0), paper_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(fig1, width='stretch')
        
        with col_R:
            st.markdown("#### 📈 Tendance Temporelle (Décès)")
            # IMPORTANT: on calcule la tendance sur un dataset minimal (AGE/SEX/PATIENT_TYPE/DATE_DIED)
            # pour éviter le biais du dropna strict sur les comorbidités.
            df_time = df_time_all.copy() if 'df_time_all' in globals() else pd.DataFrame()
            if not df_time.empty:
                # Appliquer les mêmes filtres que l'exploration
                mask_t = (df_time['AGE'] >= age_range[0]) & (df_time['AGE'] <= age_range[1])
                if gender_sel:
                    mask_t &= df_time['SEX_LABEL'].isin(gender_sel)
                if pt_sel:
                    mask_t &= df_time['PATIENT_TYPE_LABEL'].isin(pt_sel)

                df_time = df_time.loc[mask_t & (df_time['DEATH'] == 1)]

            if not df_time.empty:
                died_dt = pd.to_datetime(df_time['DATE_DIED'], errors='coerce')
                months = died_dt.dt.to_period('M').astype(str)
                time_counts = months.value_counts().reset_index()
                time_counts.columns = ['Mois', 'Décès']
                time_counts['Mois_dt'] = pd.to_datetime(time_counts['Mois'], format='%Y-%m', errors='coerce')
                time_counts = time_counts.dropna(subset=['Mois_dt']).sort_values('Mois_dt')

                fig2 = px.line(time_counts, x='Mois', y='Décès', markers=True,
                               labels={'Mois': 'Mois', 'Décès': 'Nombre de Décès'})
                fig2.update_traces(line_color='#EF553B', line_width=3)
                fig2.update_layout(xaxis_title=None, yaxis_title=None, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                                  xaxis=dict(showgrid=False), yaxis=dict(showgrid=True, gridcolor='#444'))
                st.plotly_chart(fig2, width='stretch')
            else:
                st.info("Aucun décès disponible pour la tendance avec les filtres actuels.")

        st.markdown("---")
        
        tab1, tab2, tab3 = st.tabs(["Facteurs de Risque", "Comorbidités", "Corrélation"])
        
        with tab1:
            c_bio1, c_bio2 = st.columns(2)
            with c_bio1:
                st.markdown("**Distribution Âge**")
                fig3 = px.histogram(df_filtered, x="AGE", color="DEATH_LABEL", nbins=50, 
                                    color_discrete_map={'Survivant':'#00CC96', 'Décédé':'#EF553B'},
                                    barmode="overlay", opacity=0.7)
                fig3.update_layout(xaxis_title="Âge", yaxis_title="Nombre", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                                   font=dict(color="white"), legend_title="Statut")
                st.plotly_chart(fig3, width='stretch')
                
            with c_bio2:
                st.markdown("**Soins Intensifs (ICU)**")
                df_icu = df_filtered[df_filtered['ICU_LABEL'].isin(['Oui', 'Non'])]
                if not df_icu.empty:
                    fig4 = px.violin(df_icu, y="AGE", x="ICU_LABEL", color="ICU_LABEL", box=True, points=False,
                                     color_discrete_map={'Oui':'#EF553B', 'Non':'#3498db'})
                    fig4.update_layout(xaxis_title="Admission en Réa", yaxis_title="Âge", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                                       font=dict(color="white"), showlegend=False)
                    st.plotly_chart(fig4, width='stretch')

        with tab2:
            st.markdown("**Impact des Pathologies**")
            disease = st.selectbox("Choisir pathologies", ['DIABETES', 'ASTHMA', 'OBESITY', 'CARDIOVASCULAR', 'HIPERTENSION'])
            label_col = f'{disease}_LABEL'
            
            # GroupBy est rapide, on peut utiliser tout le dataset filtré pour garder la précision
            if label_col in df_filtered.columns:
                group = df_filtered.groupby(label_col)['DEATH'].mean().reset_index()
                group['Taux Mortalité (%)'] = group['DEATH'] * 100
                
                fig5 = px.bar(group, x=label_col, y='Taux Mortalité (%)', color='Taux Mortalité (%)',
                              color_continuous_scale='Reds')
                fig5.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", 
                                   font=dict(color="white"))
                st.plotly_chart(fig5, width='stretch')

        with tab3:
            if st.button("Voir Matrice"):
                num_df = df_filtered.select_dtypes(include=[np.number])
                corr = num_df.corr()
                
                fig6 = px.imshow(corr, text_auto=".2f", color_continuous_scale='RdBu_r', aspect="auto")
                fig6.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                                   font=dict(color="white"))
                st.plotly_chart(fig6, width='stretch')

# --- PAGE 3: DIAGNOSTIC IA ---
elif page == "Diagnostic IA":
    st.title("🤖 Diagnostic IA")
    st.write("Estimation du risque vital via Random Forest.")
    
    with st.container():
        st.markdown('<div class="metric-card" style="text-align:left;">', unsafe_allow_html=True)
        c1, c2 = st.columns(2)
        with c1:
            st.subheader("👤 Profil")
            age = st.slider("Âge", 0, 100, 45)
            sexe = st.radio("Sexe", ["Femme", "Homme"], horizontal=True)
            hospital = st.radio("Prise en Charge", ["Domicile", "Hôpital"], horizontal=True)
            
            st.markdown("---")
            st.subheader("🚬 Habitudes & Autres")
            col_hab1, col_hab2 = st.columns(2)
            with col_hab1:
                tabac = st.checkbox("Fumeur")
            with col_hab2:
                enceinte = False
                if sexe == "Femme":
                    enceinte = st.checkbox("Enceinte")
                
        with c2:
            st.subheader("🏥 Clinique (Comorbidités)")
            col_a, col_b = st.columns(2)
            with col_a:
                intub = st.checkbox("Intubation")
                pneu = st.checkbox("Pneumonie")
                diab = st.checkbox("Diabète")
                copd = st.checkbox("BPCO (Poumons)")
                asthme = st.checkbox("Asthme")
            with col_b:
                immu = st.checkbox("Immunosupprimé")
                hyper = st.checkbox("Hypertension")
                cardio = st.checkbox("Cardiovasculaire")
                rein = st.checkbox("Insuffisance Rénale")
                obes = st.checkbox("Obésité")
                autre_maladie = st.checkbox("Autre Maladie")

        st.markdown('</div>', unsafe_allow_html=True)

    if st.button("Lancer Calcul de Risque", type="primary"):
        with st.spinner("Analyse du profil patient en cours..."):
            time.sleep(1) # Simulation de calcul pour UX
            
            def gv(x): return 1 if x else 0
            
            f_sex = 1 if sexe == "Femme" else 0
            f_pt = 0 if hospital == "Hôpital" else 1
            
            feat = [
                2, # USMER
                1, # MEDICAL_UNIT
                f_sex, # SEX
                f_pt, # PATIENT_TYPE
                gv(intub), # INTUBED
                gv(pneu), # PNEUMONIA
                age, # AGE
                gv(enceinte), # PREGNANT
                gv(diab), # DIABETES
                gv(copd), # COPD
                gv(asthme), # ASTHMA
                gv(immu), # INMSUPR
                gv(hyper), # HIPERTENSION
                gv(autre_maladie), # OTHER_DISEASE
                gv(cardio), # CARDIOVASCULAR
                gv(obes), # OBESITY
                gv(rein), # RENAL_CHRONIC
                gv(tabac), # TOBACCO
                3, # CLASIFFICATION_FINAL
                0  # ICU
            ]
            
            feature_names = ['USMER', 'MEDICAL_UNIT', 'SEX', 'PATIENT_TYPE', 'INTUBED', 
                            'PNEUMONIA', 'AGE', 'PREGNANT', 'DIABETES', 'COPD', 'ASTHMA', 
                            'INMSUPR', 'HIPERTENSION', 'OTHER_DISEASE', 'CARDIOVASCULAR', 
                            'OBESITY', 'RENAL_CHRONIC', 'TOBACCO', 'CLASIFFICATION_FINAL', 'ICU']
            
            feat_df = pd.DataFrame([feat], columns=feature_names)
            feat_df = feat_df.astype(float)
            
            if model:
                try:
                    # Alignement des colonnes si possible
                    if hasattr(model, 'feature_names_in_'):
                        # On filtre/ordonne selon le modèle
                        cols_model = model.feature_names_in_
                        # On remplit les manquantes par 0 si besoin (mais ici on a tout)
                        feat_df = feat_df.reindex(columns=cols_model, fill_value=0)
                    
                    prob = model.predict_proba(feat_df)[0][1] * 100
                    
                    st.markdown("---")
                    cR1, cR2 = st.columns([1,2])
                    with cR1:
                        color = "#EF553B" if prob > 50 else "#00CC96"
                        st.markdown(f'<div style="background-color:{color};border-radius:50%;width:150px;height:150px;display:flex;align-items:center;justify-content:center;font-size:36px;font-weight:bold;margin:auto;box-shadow: 0 0 20px {color};">{prob:.0f}%</div>', unsafe_allow_html=True)
                    with cR2:
                        st.markdown(f"### Probabilité de Décès : {prob:.2f}%")
                        st.progress(int(prob))
                        if prob > 50: 
                            st.error("⚠️ HAUT RISQUE : Surveillance intensive recommandée.")
                        else: 
                            st.success("✅ Risque Modéré : Protocole standard.")
                except Exception as e:
                    st.error(f"Erreur technique lors de la prédiction : {e}")
                    st.warning("Détail : Vérifiez la compatibilité des colonnes du modèle.")
            else:
                st.error("Le modèle n'a pas pu être chargé.")

# --- PAGE 4: METHODOLOGIE ---
elif page == "Méthodologie & Insights":
    st.title("📚 Méthodologie & Modélisation")
    
    st.markdown("### 1. Protocole d'Entraînement")
    c_meth1, c_meth2 = st.columns(2)
    with c_meth1:
        st.markdown("""
        <div class="metric-card">
            <h4>✂️ Division des Données</h4>
            <div style="font-size: 24px; font-weight: bold; margin: 10px 0;">80% - 20%</div>
            <div style="color: #A3A8B8;">Train - Test Split</div>
        </div>
        """, unsafe_allow_html=True)
        
    with c_meth2:
        st.markdown("""
        <div class="metric-card">
            <h4>🎯 Cible (Target)</h4>
            <div style="font-size: 24px; font-weight: bold; margin: 10px 0;">DEATH</div>
            <div style="color: #A3A8B8;">Classification Binaire</div>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("### 2. Performance Comparée")
    st.write("Le modèle Random Forest a été retenu pour sa robustesse et sa précision.")
    
    perf_data = {
        'Modèle': ['Logistic Regression', 'Random Forest', 'Naive Bayes', 'SVM'],
        'Accuracy': ['94%', '95%', '94%', '93%'],
        'F1-Score (0)': ['97%', '97%', '97%', '96%']
    }
    df_perf = pd.DataFrame(perf_data)
    # Highlight max
    st.dataframe(df_perf.style.highlight_max(axis=0, color='#262730'), width='stretch')
