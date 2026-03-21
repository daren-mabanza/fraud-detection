# ================================
# Import des packages nécéssaires
# ================================

from config import ROOT
import streamlit as st
import joblib
from fonctions_streamlit.viz_onglet_1 import afficher_metriques_binaires, afficher_matrice_confusion, afficher_courbe_pr_train_test, afficher_courbe_calibration_train_test


# ==========================================
# Paramétrage de l'environnement de travail
# ==========================================

JOBLIB_DATA = ROOT / "01_data" / "03_joblib"

# =========
# Onglet 1
# =========

def onglet_1():
    
    st.header("Métriques du modèle")

    st.markdown(
    """
    <div style="font-size:1.55rem;">
    Cet onglet rassemble les principaux éléments de pilotage du modèle de détection
    de fraude : <strong>métriques globales</strong>, <strong>matrice de confusion</strong>,
    <strong>courbes Précision–Rappel</strong> et <strong>courbes de calibration</strong>.
    L’objectif est de vérifier que le modèle discrimine correctement les transactions
    frauduleuses, que le seuil de décision retenu reste cohérent avec les <strong>coûts métier</strong>,
    et que les probabilités prédites constituent une base fiable pour la <strong>prise de décision opérationnelle</strong>.
    </div>
    """,
    unsafe_allow_html=True
)


    y_train = joblib.load(JOBLIB_DATA / "y_train.joblib")
    train_proba = joblib.load(JOBLIB_DATA / "train_proba.joblib")
    y_test = joblib.load(JOBLIB_DATA / "y_test.joblib")
    test_proba = joblib.load(JOBLIB_DATA / "test_proba.joblib")
    
    cost_optimizer = joblib.load(JOBLIB_DATA / "cost_optimizer.joblib")
    seuil_decision = cost_optimizer.get_best_threshold()

    st.subheader("")

    col_left, col_right = st.columns([1.4, 1], gap="medium")

    # ------------------ Colonne gauche : métriques globales ------------------
    with col_left:
        afficher_metriques_binaires(
            y_true=y_test,
            y_proba=test_proba,
            seuil_decision=seuil_decision,
            title="Métriques globales",
        )

        st.markdown(
            """
            <div style="font-size:1.1rem;">
    Sur la période de test (oct.–déc. 2020), le modèle conserve un très
    fort pouvoir discriminant (<strong>ROC‑AUC ≈ 99%</strong>) tout en maintenant un
    <strong>Brier score</strong> très faible, signe d’une bonne calibration des probabilités.
    Avec un <strong>recall</strong> autour de 79 % et une <strong>average precision</strong> d’environ 72%,
    le compromis retenu est cohérent avec un contexte de fraude rare mais
    coûteuse, où l’on cherche à capter un maximum de cas tout en évitant
    un volume d’alertes excessif.
            </div>
            """,
            unsafe_allow_html=True,
        )

    # ------------------ Colonne droite : matrice de confusion ----------------
    with col_right:
        st.subheader("")
        afficher_matrice_confusion(
            y_true=y_test,
            y_proba=test_proba,
            seuil_decision=seuil_decision,
            title="Matrice de confusion",
        )

        st.markdown(
            """
            <div style="font-size:1.1rem;">
    Sur l’échantillon test, le modèle détecte la majorité des fraudes
    (<strong>735 TP pour 201 FN</strong>), tout en limitant fortement les blocages à tort
    au regard du volume total de transactions (<strong>740 FP pour plus de 279 000 légitimes</strong>).
    Le <strong>ratio</strong> d’environ 1 faux positif pour 1 vrai positif est acceptable dans le
    cadre de la structure de coûts retenue, où une <strong>fraude non détectée</strong> est jugée
    cinq fois plus coûteuse qu’une alerte injustifiée.
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("---")

    # ------------------ Bas : PR et calibration -----------------------------
    bottom_left, bottom_right = st.columns([1, 1], gap="medium")

    with bottom_left:
        st.subheader("Précision–Rappel (train vs test)")
        fig_pr = afficher_courbe_pr_train_test(
            y_train=y_train,
            train_proba=train_proba,
            y_test=y_test,
            test_proba=test_proba,
            figsize=(6, 5),
        )
        st.pyplot(fig_pr, width="content")

        st.markdown(
            """
            <div style="font-size:1.1rem;">
    La courbe <strong>Précision–Rappel</strong> montre qu’il existe un large éventail
    de seuils pour lesquels le modèle conserve un <strong>rappel élevé</strong>, au
    prix d’une <strong>précision</strong> plus modérée. Le seuil de <strong>22,6%</strong> retenu sur
    l’échantillon de validation privilégie volontairement la captation
    de fraudes (<strong>recall ≈ 79 %</strong>) au détriment d’une précision parfaite,
    conformément à la hiérarchie des coûts où les <strong>fraudes non détectées</strong>
    sont beaucoup plus pénalisantes que les <strong>fausses alertes</strong>.
            </div>
            """,
            unsafe_allow_html=True,
        )

    with bottom_right:
        st.subheader("Calibration (train vs test)")
        fig_cal = afficher_courbe_calibration_train_test(
            y_train=y_train,
            train_proba=train_proba,
            y_test=y_test,
            test_proba=test_proba,
            n_bins=10,
            figsize=(6, 5),
        )
        st.pyplot(fig_cal, width="content")

        st.markdown(
            """
            <div style="font-size:1.1rem;">
    Les courbes de <strong>calibration</strong> et les <strong>Brier scores</strong> très faibles
    indiquent que les probabilités prédites restent globalement
    cohérentes entre <strong>train, validation et test</strong>. La légère tendance à
    <strong>surestimer le risque</strong> en test s’explique par la baisse du taux de
    fraude sur la période récente (drift), plus que par un <strong>surapprentissage</strong>
    du modèle. Cela permet d’utiliser les <strong>scores</strong> comme une véritable
    <strong>échelle de risque</strong>, exploitable pour fixer le seuil et prioriser
    les <strong>investigations métier</strong>.
            </div>
            """,
            unsafe_allow_html=True,
        )

