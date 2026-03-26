# ================================
# Import des packages nécéssaires
# ================================

from config import ROOT
import streamlit as st
import joblib
from fonctions_streamlit.viz_onglet_1 import afficher_metriques_binaires, afficher_matrice_confusion, afficher_courbe_pr_train_test, afficher_courbe_calibration_train_test, afficher_courbe_roc_auc_train_test


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
    <strong>courbes Précision-Rappel</strong> et <strong>courbes de calibration</strong>.
    L’objectif est premierement de vérifier que le modèle discrimine correctement les transactions
    frauduleuses et que les probabilités prédites constituent une base fiable pour la <strong>prise de décision opérationnelle</strong>.
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
    Sur la période de test (octobre - décembre 2020), le modèle conserve un très
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
    (<strong>736 TP pour 200 FN</strong>), tout en limitant fortement les blocages à tort
    au regard du volume total de transactions (<strong>746 FP pour plus de 279 000 légitimes</strong>).
    Le <strong>ratio</strong> d’environ 1 faux positif pour 1 vrai positif est acceptable dans le
    cadre de la structure de coûts retenue, où une <strong>fraude non détectée</strong> est jugée
    cinq fois plus coûteuse qu’une alerte injustifiée.
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("---")

# ------------------ Bas : ROC, PR et calibration -----------------------------
    bottom_left, bottom_center, bottom_right = st.columns(3, gap="medium")

    with bottom_left:
        st.subheader("ROC-AUC (train vs test)")
        fig_roc = afficher_courbe_roc_auc_train_test(
            y_train=y_train,
            train_proba=train_proba,
            y_test=y_test,
            test_proba=test_proba,
            figsize=(6, 5),
        )
        st.pyplot(fig_roc, width="content")

        st.markdown(
            """
            <div style="font-size:0.9rem;">
            La courbe <strong>ROC</strong> montre que le modèle conserve une excellente capacité
            à <strong>distinguer</strong> les transactions frauduleuses des transactions légitimes,
            quel que soit le seuil retenu. La proximité entre les courbes train et test,
            ainsi que des <strong>ROC-AUC très élevés</strong> dans les deux cas, indiquent que le
            modèle maintient un <strong>fort pouvoir discriminant</strong> et une performance
            globalement stable entre les différentes périodes.
            </div>
            """,
            unsafe_allow_html=True,
        )

    with bottom_center:
        st.subheader("Précision-Rappel (train vs test)")
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
            <div style="font-size:0.9rem;">
            La courbe <strong>Précision-Rappel</strong> met en évidence le compromis entre
            <strong>détection des fraudes</strong> et <strong>volume d’alertes</strong>. Le seuil de <strong>19,1%</strong>
            retenu privilégie un <strong>rappel élevé</strong> (<strong>≈ 79 %</strong>), au prix d’une
            <strong>précision</strong> plus modérée, en cohérence avec la hiérarchie des coûts où les
            fraudes non détectées sont plus pénalisantes que les fausses alertes.
            On observe par ailleurs une <strong>baisse du taux de fraude</strong> entre les périodes
            (≈ 0,56 % en train, ≈ 0,54 % en validation, ≈ 0,33 % en test), correspondant à un
            <strong>drift de prévalence</strong>. Cette évolution du contexte impacte directement
            le compromis précision-rappel pour un seuil fixé sur une période antérieure.
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
    <div style="font-size:0.9rem; line-height:1.5;">
    Les courbes de <strong>calibration</strong> et les <strong>Brier scores</strong> très faibles
    montrent que les probabilités prédites restent globalement
    <strong>cohérentes</strong> entre <strong>train, validation et test</strong>.
    On observe néanmoins une <strong>légère surestimation du risque</strong> en test,
    qui s’explique par la <strong>baisse du taux de fraude</strong> sur la période récente
    (drift de prévalence), plutôt que par un changement du comportement du modèle.
    Dans ce contexte, les <strong>scores</strong> peuvent être interprétés comme une
    véritable <strong>échelle de risque</strong>, permettant de <strong>fixer le seuil de décision</strong>
    et de <strong>prioriser efficacement les investigations métier</strong>.
    </div>
    """,
    unsafe_allow_html=True,
)
        
    st.subheader("Conclusion")
        
    st.markdown("""     
        <div style="font-size:1.1rem; line-height:1.6;">
        En synthèse, les trois analyses apportent une lecture cohérente du comportement du modèle. La <strong>courbe ROC</strong> met en évidence un <strong>fort pouvoir discriminant</strong>, stable entre train et test. La <strong>courbe Précision-Rappel</strong> illustre un compromis adapté au contexte métier, avec un <strong>rappel élevé</strong> privilégié au regard du coût des fraudes non détectées.  
        Les <strong>courbes de calibration</strong> confirment que les probabilités produites restent <strong>cohérentes et exploitables</strong> comme une véritable échelle de risque. Les écarts observés entre les périodes s’expliquent principalement par une <strong>baisse du taux de fraude</strong> (drift de prévalence), passant d’environ <strong>0,05 % à 0,03 %</strong>, ce qui impacte mécaniquement la précision et conduit à une <strong>légère surestimation du risque</strong> sur les données les plus récentes.
        La <strong>stabilité des performances en AUC</strong> ainsi que des <strong>PSI faibles (voir Github)</strong> sur les probabilités et les variables indiquent que la structure des données reste globalement inchangée.  
        Le modèle conserve donc sa capacité à <strong>discriminer efficacement</strong> les transactions, sans signe de dégradation de sa robustesse.
        Dans ce contexte, le modèle apparaît <strong>stable, cohérent et exploitable</strong> dans un cadre opérationnel bancaire.  
        </div>
        """
    ,unsafe_allow_html=True)

