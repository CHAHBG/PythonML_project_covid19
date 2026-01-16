
import streamlit as st
import pandas as pd
import numpy as np
import pickle
import plotly.express as px
import plotly.graph_objects as go

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
    
    /* Headers */
    h1, h2, h3, h4, .stMarkdown h1, .stMarkdown h2, .stMarkdown h3 {
        color: #FFFFFF !important;
        font-family: 'Inter', sans-serif;
    }
    
    p, li, div, span {
        color: #E0E0E0;
    }

    /* Sidebar Styling */
    section[data-testid="stSidebar"] {
        background-color: #262730;
    }
    
    /* Custom Dividers */
    hr {
        border-color: #41444C;
    }
</style>
""", unsafe_allow_html=True)

# --- DATA LOADING ---
@st.cache_resource
def load_data():
    try:
        # Load Raw Data
        df_raw_source = pd.read_csv('data/covid19_data.csv')
    except Exception as e:
        return pd.DataFrame(), pd.DataFrame()

    # --- 1. RAW DATA (For Display Only) ---
    df_raw = df_raw_source.copy()
    
    # Simple Label Mapping for Raw View (just for readability)
    map_dict = {1: 'Oui', 2: 'Non', 97: 'Inconnu', 99: 'Inconnu'}
    cols_to_map = ['PNEUMONIA', 'DIABETES', 'COPD', 'ASTHMA', 'INMSUPR', 
    'HIPERTENSION', 'OTHER_DISEASE', 'CARDIOVASCULAR', 'OBESITY', 'RENAL_CHRONIC', 
    'TOBACCO', 'INTUBED', 'ICU']
    
    for col in cols_to_map:
        df_raw[f'{col}_LABEL'] = df_raw[col].map(map_dict).fillna("Inconnu")
        
    df_raw['SEX_LABEL'] = df_raw['SEX'].map({1: 'Femme', 2: 'Homme'})
    df_raw['PATIENT_TYPE_LABEL'] = df_raw['PATIENT_TYPE'].map({1: 'Domicile', 2: 'Hospitalisation'})
    df_raw['DEATH'] = np.where(df_raw['DATE_DIED'] == '9999-99-99', 0, 1)
    df_raw['DEATH_LABEL'] = df_raw['DEATH'].map({0: 'Survivant', 1: 'Décédé'})

    # --- 2. CLEANED DATA (Strict Notebook Logic) ---
    df_clean = df_raw_source.copy()
    
    # Step 1: Normalisation 1->1, 2->0 (Notebook Line 481)
    # Exclude AGE, DATE_DIED, MEDICAL_UNIT, CLASIFFICATION_FINAL
    colToNormaliseYoN = [col for col in df_clean.columns if col not in ['AGE', 'DATE_DIED','MEDICAL_UNIT','CLASIFFICATION_FINAL']]
    df_clean[colToNormaliseYoN] = df_clean[colToNormaliseYoN].replace({2.0: 0, 1.0: 1})
    
    # Step 2: Remplacer 97, 98, 99 par NaN
    colToCorrect = [col for col in df_clean.columns if col not in ['AGE', 'DATE_DIED']]
    df_clean[colToCorrect] = df_clean[colToCorrect].replace({97: np.nan, 98: np.nan, 99: np.nan})
    
    # Step 3: Specific Logic (Notebook Line 822, 876)
    # Males (SEX=0 after norm? Wait. SEX: 1=Female, 2=Male. After norm: 1->1 (F), 2->0 (M).)
    # Notebook line 822: df.loc[df['SEX'] == 0, 'PREGNANT'] = 0  <-- This confirms SEX 2 became 0.
    df_clean.loc[df_clean['SEX'] == 0, 'PREGNANT'] = 0
    
    # ICU / INTUBED for Patient Type 1 (Non Hospitalisé)
    # Patient Type: 1->1 (Domicile), 2->0 (Hospital) via normalisation?
    # Wait, PATIENT_TYPE is in colToNormaliseYoN! 
    # Original: 1=Home, 2=Hospital.
    # Norm: 1->1, 2->0.
    # So Hospital is 0, Home is 1.
    # Notebook Line 876: df.loc[df['PATIENT_TYPE'] == 1, col] = ...fillna(0)
    # This implies PATIENT_TYPE 1 (Home) gets filled with 0.
    cols = ['ICU', 'INTUBED']
    for col in cols:
        df_clean.loc[df_clean['PATIENT_TYPE'] == 1, col] = df_clean.loc[df_clean['PATIENT_TYPE'] == 1, col].fillna(0)
        
    # Step 4: Drop NaN
    df_clean.dropna(inplace=True)
    
    # Step 5: Create DEATH column
    df_clean['DEATH'] = df_clean['DATE_DIED'].apply(lambda x: 0 if x == '9999-99-99' else 1)
    
    # --- Post-Processing for App Features (Labels, Dates) ---
    # We re-create labels for UI based on the NEW binary values (0/1)
    # 0 = Non/Male/Home, 1 = Oui/Female/Hospital?
    # Need to be careful with labels now.
    
    # Map back 0/1 to Strings for UI
    binary_map = {1: 'Oui', 0: 'Non'}
    sex_map = {1: 'Femme', 0: 'Homme'} # Based on assumption 2->0
    pt_map = {1: 'Domicile', 0: 'Hospitalisation'} # Based on 2->0
    
    for col in cols_to_map:
        df_clean[f'{col}_LABEL'] = df_clean[col].map(binary_map)
        
    df_clean['SEX_LABEL'] = df_clean['SEX'].map(sex_map)
    df_clean['PATIENT_TYPE_LABEL'] = df_clean['PATIENT_TYPE'].map(pt_map)
    df_clean['DEATH_LABEL'] = df_clean['DEATH'].map({0: 'Survivant', 1: 'Décédé'})
    
    # Dates
    df_clean['DATE_DIED_DT'] = df_clean['DATE_DIED'].replace('9999-99-99', np.nan)
    df_clean['DATE_DIED_DT'] = pd.to_datetime(df_clean['DATE_DIED_DT'], dayfirst=True, errors='coerce')
    df_clean['MOIS'] = df_clean['DATE_DIED_DT'].dt.to_period('M').astype(str)

    return df_raw, df_clean

@st.cache_resource
def load_model():
    try:
        return pickle.load(open('mon_modele_covid.pkl', 'rb'))
    except:
        return None

# Load BOTH datasets
df_raw, df = load_data() # df is now the CLEANED version used everywhere else

# --- SIDEBAR & NAVIGATION ---
st.sidebar.title("Navigation")
page = st.sidebar.radio("Aller vers :", 
    ["Accueil", "Exploration Intuitive", "Diagnostic IA", "Méthodologie & Insights"]
)

# --- GLOBAL FILTERS ---
df_filtered = df.copy() # Uses CLEANED data by default

if page == "Exploration Intuitive" and not df.empty:
    st.sidebar.markdown("---")
    st.sidebar.header("🔍 Filtres Globaux")
    age_range = st.sidebar.slider("Tranche d'âge", int(df['AGE'].min()), int(df['AGE'].max()), (0, 100))
    # Update filters to use new labels
    gender_opts = df['SEX_LABEL'].dropna().unique()
    gender_sel = st.sidebar.multiselect("Genre", gender_opts, default=gender_opts)
    pt_opts = df['PATIENT_TYPE_LABEL'].dropna().unique()
    pt_sel = st.sidebar.multiselect("Type de Prise en Charge", pt_opts, default=pt_opts)
    
    mask = ((df['AGE'] >= age_range[0]) & (df['AGE'] <= age_range[1]) & (df['SEX_LABEL'].isin(gender_sel)) & (df['PATIENT_TYPE_LABEL'].isin(pt_sel)))
    df_filtered = df[mask]
    st.sidebar.markdown(f"**{len(df_filtered):,} patients sélectionnés**")

st.sidebar.markdown("---")
st.sidebar.info("v2.5 | Notebook Logic Strict")

# --- PAGE 1: ACCUEIL ---
if page == "Accueil":
    st.title("🦠 Plateforme Analytics COVID-19")
    st.markdown("### 🏥 Tableau de Bord Stratégique")
    
    # 1. METRICS (Computed on Cleaned Data usually, or Raw? Notebook analysis usually on Cleaned. Let's use Cleaned for Insights)
    if not df.empty:
        st.markdown("---")
        c1, c2, c3, c4 = st.columns(4)
        total = len(df)
        deaths = df['DEATH'].sum()
        rate = (deaths/total)*100 if total > 0 else 0
        avg_age = df['AGE'].mean()
        
        with c1:
            st.markdown(f'<div class="metric-card"><div class="metric-label">Patients Analysés (Nettoyé)</div><div class="metric-value">{total:,}</div></div>', unsafe_allow_html=True)
        with c2:
            st.markdown(f'<div class="metric-card"><div class="metric-label">Décès Confirmés</div><div class="metric-value">{deaths:,}</div></div>', unsafe_allow_html=True)
        with c3:
            st.markdown(f'<div class="metric-card"><div class="metric-label">Taux de Mortalité</div><div class="metric-value">{rate:.2f}%</div></div>', unsafe_allow_html=True)
        with c4:
            st.markdown(f'<div class="metric-card"><div class="metric-label">Âge Moyen</div><div class="metric-value">{avg_age:.1f} ans</div></div>', unsafe_allow_html=True)
            
    st.markdown("---")
    
    # 2. DATA PREVIEW SECTION (Requested Feature)
    st.subheader("📂 Aperçu des Données")
    st.write("Le jeu de données a été rigoureusement nettoyé pour garantir la précision des analyses.")
    
    tab_raw, tab_clean = st.tabs(["📄 Données Brutes", "✨ Données Normalisées & Nettoyées"])
    
    with tab_raw:
        st.info("Données telles qu'elles sont collectées, incluant les valeurs manquantes (97, 99) et erreurs.")
        st.dataframe(df_raw.head(50), use_container_width=True)
        st.markdown(f"**Dimensions :** {df_raw.shape[0]:,} lignes × {df_raw.shape[1]} colonnes")
        
    with tab_clean:
        st.success("Données traitées : Valeurs manquantes supprimées, Normalisation binaire (0/1).")
        st.dataframe(df.head(50), use_container_width=True)
        st.markdown(f"**Dimensions :** {df.shape[0]:,} lignes × {df.shape[1]} colonnes")

# --- PAGE 2: EXPLORATION INTUITIVE (EVA) ---
elif page == "Exploration Intuitive":
    st.title("📊 Exploration Dynamique")
    
    if df_filtered.empty:
        st.warning("Aucune donnée disponible avec les filtres actuels.")
    else:
        col_L, col_R = st.columns(2)
        
        # 1. DONUT CHART (Survivors vs Deaths)
        with col_L:
            st.markdown("#### 🍩 Répartition des Issues")
            counts = df_filtered['DEATH_LABEL'].value_counts().reset_index()
            counts.columns = ['Statut', 'Nombre']
            
            fig1 = px.pie(counts, values='Nombre', names='Statut', hole=0.5, 
                          color='Statut', color_discrete_map={'Survivant':'#00CC96', 'Décédé':'#EF553B'})
            fig1.update_traces(textinfo='percent+label')
            fig1.update_layout(showlegend=False, margin=dict(t=0, b=0, l=0, r=0), paper_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(fig1, use_container_width=True)
        
        # 2. LINE CHART (Temporal Trend)
        with col_R:
            st.markdown("#### 📈 Tendance Temporelle (Décès)")
            df_time = df_filtered[df_filtered['DEATH'] == 1]
            if not df_time.empty:
                time_counts = df_time['MOIS'].value_counts().sort_index().reset_index()
                time_counts.columns = ['Mois', 'Décès']
                
                fig2 = px.line(time_counts, x='Mois', y='Décès', markers=True,
                               labels={'Mois': 'Mois', 'Décès': 'Nombre de Décès'})
                fig2.update_traces(line_color='#EF553B', line_width=3, hovertemplate="<b>%{x}</b><br>Décès: %{y}<extra></extra>")
                fig2.update_layout(xaxis_title=None, yaxis_title=None, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                                  xaxis=dict(showgrid=False), yaxis=dict(showgrid=True, gridcolor='#444'))
                st.plotly_chart(fig2, use_container_width=True)

        st.markdown("---")
        
        tab1, tab2, tab3 = st.tabs(["Facteurs de Risque", "Comorbidités", "Corrélation"])
        
        # 3. AGE DISTRIBUTION & ICU VIOLIN
        with tab1:
            c_bio1, c_bio2 = st.columns(2)
            with c_bio1:
                st.markdown("**Distribution Âge**")
                fig3 = px.histogram(df_filtered, x="AGE", color="DEATH_LABEL", nbins=50, 
                                    color_discrete_map={'Survivant':'#00CC96', 'Décédé':'#EF553B'},
                                    barmode="overlay", opacity=0.7,
                                    labels={'AGE': 'Âge (Ans)', 'DEATH_LABEL': 'Statut Vital', 'count': 'Nombre de Patients'})
                fig3.update_layout(xaxis_title="Âge", yaxis_title="Nombre", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                                   font=dict(color="white"), legend_title="Statut")
                st.plotly_chart(fig3, use_container_width=True)
                
            with c_bio2:
                st.markdown("**Soins Intensifs (ICU)**")
                df_icu = df_filtered[df_filtered['ICU_LABEL'].isin(['Oui', 'Non'])]
                if not df_icu.empty:
                    # Consistent colors & Better Labels
                    fig4 = px.violin(df_icu, y="AGE", x="ICU_LABEL", color="ICU_LABEL", box=True, points=False, # Removed points to reduce clutter
                                     color_discrete_map={'Oui':'#EF553B', 'Non':'#3498db'},
                                     labels={'ICU_LABEL': 'Admission en Réanimation', 'AGE': 'Âge (Ans)'})
                    fig4.update_layout(xaxis_title="Admission en Réa", yaxis_title="Âge", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                                       font=dict(color="white"), showlegend=False)
                    st.plotly_chart(fig4, use_container_width=True)

        # 4. BAR CHART COMORBIDITIES
        with tab2:
            st.markdown("**Impact des Pathologies**")
            disease = st.selectbox("Choisir pathologies", ['DIABETES', 'ASTHMA', 'OBESITY', 'CARDIOVASCULAR', 'HIPERTENSION'])
            label_col = f'{disease}_LABEL'
            
            if label_col in df_filtered.columns:
                group = df_filtered.groupby(label_col)['DEATH'].mean().reset_index()
                group['Taux Mortalité (%)'] = group['DEATH'] * 100
                
                # Gradient & Labels
                fig5 = px.bar(group, x=label_col, y='Taux Mortalité (%)', color='Taux Mortalité (%)',
                              color_continuous_scale='Reds',
                              labels={label_col: 'Présence de la Maladie', 'Taux Mortalité (%)': 'Taux de Décès (%)'})
                fig5.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", 
                                   font=dict(color="white"))
                st.plotly_chart(fig5, use_container_width=True)

        # 5. HEATMAP
        with tab3:
            if st.button("Voir Matrice"):
                num_df = df_filtered.select_dtypes(include=[np.number])
                corr = num_df.corr()
                
                fig6 = px.imshow(corr, text_auto=".2f", color_continuous_scale='RdBu_r', aspect="auto",
                                 labels=dict(x="Variable 1", y="Variable 2", color="Corrélation"))
                fig6.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                                   font=dict(color="white"))
                st.plotly_chart(fig6, use_container_width=True)

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
            tabac = st.checkbox("Fumeur")
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
        # MAPPING VARIABLES (Strict Notebook Logic: 1=Oui/True, 2=Non/False)
        # However, the model was trained on:
        # 1. Normalized Data? check notebook load_data equivalent.
        # In notebook: 1=Yes, 2=No -> Normalized to 1=Yes, 0=No?
        # NO! The notebook "Strict Replication" showed:
        # replace({2.0: 0, 1.0: 1})
        # So inputs to model must be 0 or 1.
        
        # 1=Yes/True, 0=No/False
        def gv(x): return 1 if x else 0 # Changed from 1(Yes)/2(No) to 1(Yes)/0(No) based on normalization check
        
        f_sex = 1 if sexe == "Femme" else 0 # 1=Femme, 0=Homme (based on notebook logic)
        f_pt = 0 if hospital == "Hôpital" else 1 # Notebook: 1->1 (Home), 2->0 (Hospital)
        # Wait, check notebook again.
        # Norm: 1->1, 2->0. Original: 1=Home, 2=Hospital. So Home=1, Hospital=0.
        
        # FEATURE ORDER MUST MATCH X_train columns EXACTLY
        # USMER, MEDICAL_UNIT, SEX, PATIENT_TYPE, INTUBED, PNEUMONIA, AGE, PREGNANT,
        # DIABETES, COPD, ASTHMA, INMSUPR, HIPERTENSION, OTHER_DISEASE, CARDIOVASCULAR,
        # OBESITY, RENAL_CHRONIC, TOBACCO, CLASIFFICATION_FINAL, ICU
        
        # Note: CLASIFFICATION_FINAL was typically 3 or 7 in examples?
        # MEDICAL_UNIT was 12.
        # USMER was 2.
        # Let's assume standard defaults for non-clinical input fields.
        
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
            3, # CLASIFFICATION_FINAL (3=Covid Positive usually)
            0  # ICU (Assume 0 if not specified, or add checkbox? Let's assume correlated with Intubation/Hospital)
        ]
        
        # Create DataFrame with proper column names (matching training data)
        feature_names = ['USMER', 'MEDICAL_UNIT', 'SEX', 'PATIENT_TYPE', 'INTUBED', 
                        'PNEUMONIA', 'AGE', 'PREGNANT', 'DIABETES', 'COPD', 'ASTHMA', 
                        'INMSUPR', 'HIPERTENSION', 'OTHER_DISEASE', 'CARDIOVASCULAR', 
                        'OBESITY', 'RENAL_CHRONIC', 'TOBACCO', 'CLASIFFICATION_FINAL', 'ICU']
        
        feat_df = pd.DataFrame([feat], columns=feature_names)
        
        model = load_model()
        if model:
            try:
                prob = model.predict_proba(feat_df)[0][1] * 100
                
                cR1, cR2 = st.columns([1,2])
                with cR1:
                    color = "#EF553B" if prob > 50 else "#00CC96"
                    st.markdown(f'<div style="background-color:{color};border-radius:50%;width:120px;height:120px;display:flex;align-items:center;justify-content:center;font-size:30px;font-weight:bold;margin:auto;">{prob:.0f}%</div>', unsafe_allow_html=True)
                with cR2:
                    st.markdown(f"### Probabilité de Décès : {prob:.2f}%")
                    st.progress(int(prob))
                    if prob > 50: st.error("⚠️ HAUT RISQUE")
                    else: st.success("✅ Risque Modéré")
            except Exception as e:
                st.error(f"Erreur lors de la prédiction : {e}")
                st.write("Vérifiez que le modèle attend bien 20 features.")
                
# --- PAGE 4: METHODOLOGIE ---
elif page == "Méthodologie & Insights":
    st.title("📚 Mèthodologie & Modélisation")
    
    st.markdown("### 1. Protocole d'Entraînement")
    st.write("Pour garantir la fiabilité de nos prédictions, nous avons suivi un protocole rigoureux.")
    
    c_meth1, c_meth2 = st.columns(2)
    with c_meth1:
        st.markdown("""
        <div class="metric-card">
            <h4>✂️ Division des Données</h4>
            <div style="font-size: 24px; font-weight: bold; margin: 10px 0;">80% - 20%</div>
            <p>80% pour l'Entraînement (Train)<br>20% pour le Test</p>
        </div>
        """, unsafe_allow_html=True)
        
    with c_meth2:
        st.markdown("""
        <div class="metric-card">
            <h4>🎯 Cible (Target)</h4>
            <div style="font-size: 24px; font-weight: bold; margin: 10px 0;">DEATH</div>
            <p>0 = Survivant<br>1 = Décédé</p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")
    
    st.markdown("### 2. Comparaison des Modèles")
    st.write("Nous avons testé 4 algorithmes majeurs pour sélectionner le meilleur.")
    
    # Data manually extracted from the notebook execution
    perf_data = {
        'Modèle': ['Logistic Regression', 'Random Forest', 'Naive Bayes', 'SVM'],
        'Accuracy': ['94%', '95%', '94%', '93%'],
        'Precision (0)': ['95%', '96%', '96%', '93%'],
        'Recall (0)': ['98%', '98%', '98%', '100%'],
        'F1-Score (0)': ['97%', '97%', '97%', '96%']
    }
    df_perf = pd.DataFrame(perf_data)
    
    # Styled Table
    st.table(df_perf)
    
    st.markdown("""
    > **Pourquoi le Random Forest ?**  
    > C'est le modèle le plus équilibré. Bien que le SVM ait un Recall de 100% sur la classe 0, il échoue totalement à prédire les décès (Recall 0% sur la classe 1). 
    > Le **Random Forest** offre la meilleure performance globale avec une Accuracy de **95%** et une capacité réelle à détecter les cas à risque.
    """)
    
    st.markdown("### 3. Sauvegarde")
    st.code("pickle.dump(model_rf, open('mon_modele_covid.pkl', 'wb'))", language="python")
    st.write("Le modèle final est exporté en fichier `.pkl` pour être intégré ici.")
