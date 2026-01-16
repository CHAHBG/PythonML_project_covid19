# 📘 Guide Complet du Code : Comprendre & Réutiliser `app.py`

Ce guide décortique le fichier `app.py` bloc par bloc. Pour chaque partie, vous trouverez :
1.  **Le Code** : L'extrait important.
2.  **L'Utilité** : À quoi ça sert ?
3.  **Réutilisation** : Comment vous en servir pour un autre projet (ex: Immo, Finance, RH).

---

## 1. La Configuration de la Page
C'est la première chose que Streamlit lit.

```python
st.set_page_config(
    page_title="COVID-19 Analytics",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded"
)
```
*   **Utilité :** Définit le titre de l'onglet du navigateur, l'icône, et dit à l'application de prendre toute la largeur de l'écran (`layout="wide"`).
*   **Comment le réutiliser :** Copiez ce bloc au tout début de n'importe quelle app Streamlit. Changez juste le `page_title` (ex: "Analyse Bourse") et l'icône.

---

## 2. Le Chargement des Données avec Cache
C'est crucial pour la vitesse.

```python
@st.cache_resource
def load_data():
    df = pd.read_csv('data/covid19_data.csv')
    # ... nettoyage ...
    return df
```
*   **Utilité :** `@st.cache_resource` est une "boîte magique". La première fois, Streamlit lit le fichier (c'est lent). La deuxième fois, il se souvient du résultat (c'est instantané). Sans ça, l'app serait lente à chaque clic.
*   **Comment le réutiliser :** Utilisez toujours `@st.cache_resource` ou `@st.cache_data` avant vos fonctions qui chargent des fichiers lourds (Excel, CSV, Base de données).

---

## 3. Le Nettoyage de Données (Data Cleaning)
Dans `load_data`, on a ce genre de logique :

```python
map_dict = {1: 'Oui', 2: 'Non', 97: 'Inconnu'}
df['DIABETES_LABEL'] = df['DIABETES'].map(map_dict)
```
*   **Utilité :** Les ordinateurs aiment les chiffres (1, 2), les humains aiment les mots ("Oui", "Non"). On crée des nouvelles colonnes (`_LABEL`) juste pour l'affichage, tout en gardant les originaux pour les calculs.
*   **Comment le réutiliser :** Dans tous vos projets, séparez les données de calcul (chiffres) des données d'affichage (textes). Créez des dictionnaires `map_dict` pour traduire vos codes.

---

## 4. La Barre Latérale (Sidebar) & Navigation
Pour créer un menu simple.

```python
st.sidebar.title("Navigation")
page = st.sidebar.radio("Aller vers :", ["Accueil", "Exploration", "Diagnostic"])

if page == "Accueil":
    # ... code de la page accueil ...
```
*   **Utilité :** Permet de créer une application "multi-pages" dans un seul fichier. On utilise une simple condition `if` pour savoir quoi afficher.
*   **Comment le réutiliser :** C'est la structure standard. Pour ajouter une page "Contact", ajoutez simplement "Contact" dans la liste et créez un bloc `elif page == "Contact":`.

---

## 5. Les Graphiques Interactifs (Plotly)
On a remplacé les images statiques par des graphiques où on peut zoomer.

```python
# Exemple de Camembert (Pie Chart)
fig = px.pie(counts, values='Nombre', names='Statut', 
             color_discrete_map={'Survivant':'#00CC96', 'Décédé':'#EF553B'})
st.plotly_chart(fig, use_container_width=True)
```
*   **Utilité :** `px.pie` crée le visuel. `color_discrete_map` force les couleurs (Vert pour survivant, Rouge pour décès) pour que ce soit constant.
*   **Comment le réutiliser :**
    *   `px.bar(...)` pour des histogrammes.
    *   `px.line(...)` pour des évolutions dans le temps.
    *   Toujours utiliser `use_container_width=True` pour que le graphique s'adapte aux mobiles.

---

## 6. L'Intégration du Modèle IA (Machine Learning)
C'est le cœur intelligent.

```python
# 1. Chargement
model = pickle.load(open('mon_modele_covid.pkl', 'rb'))

# 2. Préparation des données saisies par l'utilisateur
# L'utilisateur coche "Diabète" (Vrai/Faux) -> On traduit en 1 ou 0
def gv(valeur_case): return 1 if valeur_case else 0
feat = [2, 1, gv(diabete), gv(fumeur), ...] 

# 3. Prédiction
prob = model.predict_proba([feat])[0][1] # Probabilité de la classe 1 (Décès)
```
*   **Utilité :** Connecte l'interface visuelle (les boutons) au cerveau mathématique (le fichier `.pkl`).
*   **Comment le réutiliser :**
    1.  Entraînez votre modèle dans un Notebook (Jupyter).
    2.  Sauvegardez-le avec `pickle.dump()`.
    3.  Chargez-le dans Streamlit avec `pickle.load()`.
    4.  **Important :** L'ordre des variables dans `feat` doit être *exactement* le même que lors de l'entraînement.

---

## 7. Le "Dark Mode" Perso (CSS Injection)
Pour avoir ce look pro.

```python
st.markdown("""
<style>
    .stApp { background-color: #0E1117; }
    .metric-card { background-color: #262730; ... }
</style>
""", unsafe_allow_html=True)
```
*   **Utilité :** Streamlit permet d'injecter du code CSS (le langage de style du web) pour modifier l'apparence au-delà des options de base.
*   **Comment le réutiliser :** Copiez ce bloc si vous voulez un thème sombre "Dashboard". Pour un thème clair, changez les codes couleurs un par un (ex: `#FFFFFF` pour le fond).
