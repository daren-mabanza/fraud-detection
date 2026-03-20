# ================================
# Import des packages nécéssaires
# ================================

import streamlit as st
from pathlib import Path
import numpy as np
import joblib
import matplotlib.pyplot as plt
import shap
from fonctions_streamlit.utilitaires import score_transaction, make_tx_key, load_sample_data
from fonctions_streamlit.ia_utilitaires import call_llm_explanation
from fonctions_streamlit.viz_onglet_3 import plot_local_shap_vertical

# ==========================================
# Paramétrage de l'environnement de travail
# ==========================================

ROOT = Path.cwd().parents[0]
JOBLIB_DATA = ROOT / "01_data" / "03_joblib"

# =========
# Onglet 3
# =========

def onglet_3():
    shap_values_global = joblib.load(JOBLIB_DATA / "shap_values_model.joblib")

    st.header("Transactions et explications locales")

    st.markdown(
        """
        <div style="font-size:1.75rem;">
    Cet onglet permet d’explorer des <strong>transactions individuelles</strong> et de comprendre,
    pour chacune d’elles, <strong>pourquoi</strong> le modèle attribue un certain <strong>score de fraude</strong>.
    On combine ici une <strong>explication locale via SHAP</strong> et une <strong>reformulation métier</strong>
    générée par un <strong>LLM</strong>.
        </div>
        """,
        unsafe_allow_html=True,
    )

    # 0. Importance moyenne SHAP (global) + texte d'interprétation
    col_imp_left, col_imp_right = st.columns(2, gap="medium")

    with col_imp_left:
        st.subheader("Importance moyenne des variables (SHAP – global)")
        plt.clf()
        plt.figure(figsize=(6, 4))
        shap.plots.bar(shap_values_global, max_display=8, show=False)
        fig_bar = plt.gcf()
        st.pyplot(fig_bar, bbox_inches="tight", dpi=120)

    with col_imp_right:
        st.subheader("Lecture du graphique global")
        
        st.markdown(
    """
    <div style="font-size:1.3rem;">
    Sur l'ensemble des transactions, <strong>le montant</strong> ressort comme le facteur
    le plus influent sur le score de fraude, loin devant les autres
    variables. Les <strong>créneaux horaires</strong> (par exemple 22h–23h), certains
    <strong>types de magasins</strong> (transport, grande distribution) et l'<strong>if_score</strong>
    apportent également une contribution notable au risque moyen.
    <br><br>
    Concrètement, cela signifie que le modèle s'appuie d'abord sur des
    <strong>signaux simples et intuitifs</strong> (montant, heure, type de commerce),
    complétés par un indicateur d'<strong>anormalité de la transaction</strong>
    (if_score), pour différencier les opérations usuelles de celles qui
    ressemblent statistiquement à des fraudes.
    </div>
    """,
    unsafe_allow_html=True,
)

    # Chargement d'un petit échantillon
    
    sample_data = load_sample_data().copy()
    sample_data = sample_data.reset_index(drop=True)
    sample_data.insert(0, "id_ligne", sample_data.index + 1)  # 1 à 10

    st.subheader("Sélectionnez une transaction pour voir l’explication détaillée :")
    st.dataframe(sample_data, hide_index=True, use_container_width=True, height = 350)

    # 2. Choix d'un id_ligne (1 à 10)
    ids = sample_data["id_ligne"].tolist()
    selected_id = st.selectbox(
        "Transaction à expliquer (identifiant de la ligne affichée) :",
        options=ids,
        index=0,
    )

    # On retrouve la ligne correspondante
    row = (
        sample_data.loc[sample_data["id_ligne"] == selected_id]
        .drop(columns=["id_ligne", "target"])
        .iloc[0]
    )
    
    row_dict = row.to_dict()
    tx_key = make_tx_key(row_dict)

        # 3. Scoring + SHAP local
    proba, shap_values, X_proc = score_transaction(row_dict)
    st.markdown(f"**Probabilité estimée de fraude :** {proba:.2%}")

    st.markdown("---")

    # ===== Explication locale (SHAP – waterfall) en pleine largeur =====
    st.subheader("Explication locale (SHAP – waterfall)")

    shap_arr = np.array(shap_values)
    if shap_arr.ndim == 1:
        sv_row = shap_arr
    else:
        sv_row = shap_arr[0, :]

    feature_names = X_proc.columns.tolist()

    fig = plot_local_shap_vertical(shap_values, X_proc.columns, top_k=10)
    st.plotly_chart(fig, use_container_width=True)

    st.markdown(
        """
        <div style="font-size:1.25rem; line-height:1.4; color:#888888; margin-top:0.5rem;">
        ⚠️ <strong>Lecture des contributions locales</strong><br>
        Les barres ci-dessus montrent l'impact de chaque variable sur le
        <em>score interne de risque</em> du modèle <strong>XGBoost</strong>, avant calibration
        des probabilités. XGBoost est très performant pour discriminer les
        transactions frauduleuses, mais ses probabilités brutes sont
        souvent mal calibrées.<br>
        Une étape de <em>calibration statistique</em> est donc appliquée
        ensuite pour obtenir la <strong>probabilité finale de fraude</strong> affichée pour
        la décision. Les contributions <strong>SHAP</strong> doivent être lues comme un outil
        pour comprendre quelles variables augmentent ou diminuent le risque
        relatif, en gardant à l'esprit qu'il s'agit du <strong>score avant calibration</strong> :
        de légères différences avec la probabilité finale sont normales.
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("---")

    # ===== Explication reformulée pour le métier (pleine largeur) =====
    st.subheader("Explication reformulée pour le métier")

    if st.button("Générer une explication métier (LLM)"):
        with st.spinner("Génération de l'explication en langage naturel..."):
            shap_1d = sv_row
            explanation = call_llm_explanation(
                tx_key=tx_key,
                proba=proba,
                shap_values=shap_1d,
                input_dict=row_dict,
                feature_names=feature_names,
                k_top=5,
            )

        st.markdown(
            f"""
            <div style="font-size:1.45rem; line-height:1.5;">
                {explanation}
            </div>
            """,
            unsafe_allow_html=True,
        )
