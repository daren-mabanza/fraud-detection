# ================================
# Import des packages nécéssaires
# ================================

from config import ROOT
import streamlit as st
import pandas as pd
import joblib
import random
from fonctions_streamlit.utilitaires import score_transaction, sample_from_tranche

# ==========================================
# Paramétrage de l'environnement de travail
# ==========================================

JOBLIB_DATA = ROOT / "01_data" / "03_joblib"

# =========
# Onglet 4
# =========


def onglet_4():
    st.header("Votre transaction serait-elle considérée comme frauduleuse ?")
    
    st.markdown(
    """
        <div style="font-size:1.3rem;">
        Renseignez les caractéristiques d’une transaction (type d’achat, montant, heure, profil client) puis cliquez sur <strong>"Évaluer la transaction"</strong> pour obtenir une <strong>estimation du risque de fraude</strong> et la décision associée.
        </div>    
    """,
    unsafe_allow_html=True
)

    # États qu'on propose explicitement dans l'UI
    etats_principaux = ["CA", "NY", "TX", "FL", "NJ"]
    
    # Liste complète des états hors ceux qui seront explicitement proprosés dans l'UI
    modalites_etat_client = joblib.load(JOBLIB_DATA / "values_etat_client.joblib")
    autres_etats = [etat for etat in modalites_etat_client if etat not in etats_principaux]

    # Mise en page : deux colonnes
    col_client, col_tx = st.columns(2, gap="medium")

    # -------------------------
    # Widgets utilisateur
    # -------------------------
    with col_tx:
        st.subheader("Détails de la transaction")

        type_magasin = st.selectbox(
            "Type de magasin",
            [
                "grocery_pos",
                "gas_transport",
                "shopping_pos",
                "misc_net",
                "shopping_net",
                "entertainment",
                "food_dining",
                "health_fitness",
                "personal_care",
                "travel",
            ],
        )

        montant_tranche = st.selectbox(
            "Montant de la transaction (tranche)",
            [
                "0 – 50 €",
                "50 – 100 €",
                "100 – 200 €",
                "200 – 500 €",
                "500 – 1 000 €",
                "1 000 – 5 000 €",
            ],
        )

        heure = st.selectbox(
            "Heure de la transaction",
            list(range(0, 24)),
            format_func=lambda h: f"{h:02d}h",
        )

    with col_client:
        st.subheader("Profil client")

        etat_ui = st.selectbox(
            "État du client",
            etats_principaux + ["Autre"],
        )

        age_tranche = st.selectbox(
            "Âge du client (tranche)",
            [
                "18 – 25 ans",
                "25 – 35 ans",
                "35 – 50 ans",
                "50 – 65 ans",
                "65 – 90 ans",
            ],
        )

    st.markdown("---")

    # -------------------------
    # Bouton d'évaluation
    # -------------------------
    if st.button("Évaluer la transaction", type="primary"):

        # Dictionnaires de bornes pour les tranches
        montant_bounds = {
            "0 – 50 €": (0, 50),
            "50 – 100 €": (50, 100),
            "100 – 200 €": (100, 200),
            "200 – 500 €": (200, 500),
            "500 – 1 000 €": (500, 1000),
            "1 000 – 5 000 €": (1000, 5000),
        }

        age_bounds = {
            "18 – 25 ans": (18, 25),
            "25 – 35 ans": (25, 35),
            "35 – 50 ans": (35, 50),
            "50 – 65 ans": (50, 65),
            "65 – 90 ans": (65, 90),
        }

        # Tirage aléatoire dans chaque tranche
        montant = sample_from_tranche(montant_tranche, montant_bounds)
        age = int(round(sample_from_tranche(age_tranche, age_bounds)))

        # Gestion de "autre" : on choisit un état réel non déjà dans etats_principaux
        if etat_ui == "Autre":
            etat_effectif = random.choice(autres_etats)
        else:
            etat_effectif = etat_ui

        # Dictionnaire final envoyé au modèle (types forcés)
        input_dict = {
            "type_magasin": str(type_magasin),
            "montant_transaction": float(montant),
            "etat_client": str(etat_effectif),
            "heure_transaction": str(int(heure)),
            "age_client": int(age),
        }
        
        # 1. Scoring
        proba, shap_values, X_proc = score_transaction(input_dict)

        cost_optimizer = joblib.load(JOBLIB_DATA / "cost_optimizer.joblib")
        seuil = cost_optimizer.get_best_threshold() 
        
        decision = "ALERTE FRAUDE" if proba >= seuil else "ACCEPTÉE"

        # Bandeau couleur en haut
        if decision == "ALERTE FRAUDE":
            st.markdown(
                f"""
                <div style="padding:0.75rem; border-radius:0.5rem;
                            background-color:#7f1d1d; color:#fef2f2;
                            border:1px solid #fecaca; margin-bottom:0.75rem;">
                    <strong>Décision automatique :</strong> transaction signalée comme <strong>suspecte</strong>
                    (seuil {seuil:.3f}, proba fraude = {proba:.3f}).
                </div>
                """,
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                f"""
                <div style="padding:0.75rem; border-radius:0.5rem;
                            background-color:#064e3b; color:#ecfdf5;
                            border:1px solid #6ee7b7; margin-bottom:0.75rem;">
                    <strong>Décision automatique :</strong> transaction considérée comme
                    <strong>légitime</strong> (seuil {seuil:.3f}, proba fraude = {proba:.3f}).
                </div>
                """,
                unsafe_allow_html=True,
            )

        # Metrics
        col_score, col_decision = st.columns(2)
        with col_score:
            st.metric(
                label="Probabilité estimée de fraude",
                value=f"{proba:.3f}",
            )
        with col_decision:
            st.metric(
                label="Décision (seuil 0,191)",
                value=decision,
            )

        # Récap des valeurs utilisées
        st.markdown("#### Valeurs utilisées pour la simulation")
        recap = {
            "type_magasin": [type_magasin],
            "etat_client": [etat_effectif],
            "montant_transaction": [round(montant, 2)],
            "age_client": [age],
            "heure_transaction": [heure]
        }
        st.table(pd.DataFrame(recap))
        
        st.markdown(
    """
    <div style="font-size:1rem; line-height:1.5; margin-bottom:20px;">
        <strong>⚠️ Avertissement</strong><br><br>

        Le modèle a été entraîné sur des <strong>transactions simulées</strong>.<br>
        La probabilité affichée dans cet onglet indique donc comment la transaction créée se positionne par rapport aux 
        <strong>schémas appris dans ces données générées</strong>, et non par rapport à l’ensemble des situations réelles possibles.<br>
        Une transaction pouvant paraître très étrange dans la réalité ne sera donc pas forcément classée comme très risquée ici, 
        si ce type de cas n’est pas correctement représenté dans les données simulées d’entraînement.<br><br>

        Ce score doit être lu comme une <strong>illustration du fonctionnement du modèle dans un cadre simulé</strong>.
    </div>
    """,
    unsafe_allow_html=True
)