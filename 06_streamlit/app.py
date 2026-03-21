# ============================================================================
# Import des packages nécéssaires & Paramétrage de l'environnement de travail
# ============================================================================
from config import ROOT
import streamlit as st
from fonctions_streamlit.onglet_1 import onglet_1
from fonctions_streamlit.onglet_2 import onglet_2
from fonctions_streamlit.onglet_3 import onglet_3
from fonctions_streamlit.onglet_4 import onglet_4

# ===============
# Streamlit Main
# ===============
def main():
    # Titre de l'application
    st.set_page_config(page_title="Détection de fraude carte bancaire", layout="wide")
    
    st.markdown(
        """
        <style>
        button[data-baseweb="tab"] > div[data-testid="stMarkdownContainer"] p {
            font-size: 1.3rem;
            font-weight: 600;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    st.title("Prédiction de la fraude bancaire - Modèle XGBoost explicable")

    # Titres des onglets
    tab1, tab2, tab3, tab4 = st.tabs([
        "Métriques ML",
        "Métriques métier & coûts",
        "Transactions & explications",
        "Simulateur de transaction",
    ])

    with tab1:
        onglet_1() # Onglet 1
    with tab2:
        onglet_2() # Onglet 2
    with tab3:
        onglet_3() # Onglet 3
    with tab4:
        onglet_4() # Onglet 4


if __name__ == "__main__":
    main()
