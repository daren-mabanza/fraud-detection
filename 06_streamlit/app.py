# Paramétrage de l'environnement de travail et import des packages
import sys
from pathlib import Path
    # ------------------------------------------
ROOT = Path.cwd().parents[0]
    # ------------------------------------------
RAW_DATA = ROOT / "01_data" / "01_raw"
PROCESSED_DATA = ROOT / "01_data" / "02_processed"
MODEL_DATA = ROOT / "04_model"
    # ------------------------------------------
sys.path.append(str(ROOT / "03_fonctions"))
    # ------------------------------------------
import pandas as pd
import streamlit as st
import shap
import joblib
import json
from perplexity import Perplexity
from fonctions_perso.machine_learning import BinaryMetricsSimple, graphique_courbe_pr, graphique_courbe_calibration, graphique_courbe_roc, ThresholdCostOptimizer
from fonctions_streamlit.onglet1 import afficher_metriques_binaires, afficher_matrice_confusion, afficher_courbe_pr_train_test, afficher_courbe_calibration_train_test
from fonctions_streamlit.onglet2 import plot_cost_threshold_curve

import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go
from sklearn import set_config

# Cache global (LLM)
EXPLANATION_CACHE = {}

EXPLANATION_CACHE_PATH = PROCESSED_DATA / "llm_explanations_cache.json"

# Chargement du cache persistant
try:
    with open(EXPLANATION_CACHE_PATH, "r", encoding="utf-8") as f:
        EXPLANATION_CACHE = json.load(f)
except FileNotFoundError:
    EXPLANATION_CACHE = {}


def save_explanation_cache():
    with open(EXPLANATION_CACHE_PATH, "w", encoding="utf-8") as f:
        json.dump(EXPLANATION_CACHE, f, ensure_ascii=False, indent=2)


# Paramétrage de l'accès au LLM 
perplexity_json = "perplexity_api_key.json"
with open(ROOT / perplexity_json, "r") as f:
    secrets = json.load(f)

TOKEN = secrets["PERPLEXITY_API_KEY"]

client = Perplexity(api_key=TOKEN)

# Fonctions utilitaires de l'application
@st.cache_resource
def load_models():

    modele_propre = joblib.load(MODEL_DATA / "fraud_detection_model_xgboost.joblib")
    modele_shap = joblib.load(MODEL_DATA / "fraud_detection_xgb_pas_calibre.joblib")
 
    return modele_propre, modele_shap

    # ------------------------------------------
@st.cache_data
def load_sample_data():

    data_1_sample = joblib.load(PROCESSED_DATA / "1_fraud_data_sample.joblib")
    data_0_sample = joblib.load(PROCESSED_DATA / "0_fraud_data_sample.joblib")

    sample_data = pd.concat([data_1_sample,data_0_sample], axis = 0).sample(10)

    return sample_data

    # ------------------------------------------
def score_transaction(input_dict):
    """
    input_dict : dict des variables brutes (même format que X_train/X_test du notebook)
    renvoie : proba, shap_values, X_proc
    """
    modele_propre, modele_shap = load_models()   # modele_shap = pipeline complet utilisé pendant le training
    explainer = joblib.load(PROCESSED_DATA / "explainer.joblib")

    # 1. DataFrame brut
    X_raw = pd.DataFrame([input_dict])

    # 2. Passage dans le pipeline EXACTEMENT comme pendant le fit SHAP
    # Ici: on applique tout le pipeline "modele_shap" et on enlève juste l'estimateur final
    set_config(transform_output="pandas")
    preprocess = modele_shap[:-1]
    X_proc = preprocess.transform(X_raw)

    # 3. Proba avec le modèle calibré (si modele_propre attend X_raw)
    proba = modele_propre.predict_proba(X_raw)[:, 1][0]

    # 4. SHAP sur le même espace que x_test_proc
    shap_values = explainer.shap_values(X_proc)

    return proba, shap_values, X_proc


    # ------------------------------------------
def plot_local_shap_plotly(
    shap_values,
    feature_names,
    base_value,
    top_k: int = 15,
    title: str = "Contribution locale des variables (SHAP-like)"
):
    """
    Construit un graphique type waterfall en Plotly à partir des valeurs SHAP.

    shap_values : array 1D (n_features,)
    feature_names : liste des noms de variables (longueur n_features)
    base_value : valeur de base du modèle (expected_value)
    top_k : nombre maximum de variables affichées (en valeur absolue)
    """

    shap_arr = np.array(shap_values).astype(float)
    feature_names = list(feature_names)

    # 1) Top k features en importance absolue
    idx_sorted = np.argsort(-np.abs(shap_arr))
    idx_top = idx_sorted[:top_k]

    vals = shap_arr[idx_top]
    names = [feature_names[i] for i in idx_top]

    # 2) Calcul des positions cumulées (waterfall)
    # f(x) = base_value + somme(shap)
    contribs = vals
    cumulative = base_value + np.cumsum(contribs)

    # point de départ
    x = list(range(len(contribs)))
    base_line = [base_value] + list(cumulative[:-1])

    # 3) Couleur selon signe de la contribution
    colors = ["#e74c3c" if v > 0 else "#3498db" for v in contribs]

    fig = go.Figure()

    fig.add_trace(
        go.Bar(
            x=names,
            y=contribs,
            base=base_line,
            marker_color=colors,
            name="Contribution SHAP",
            hovertemplate=(
                "<b>%{x}</b><br>"
                "Contribution : %{y:.3f}<br>"
                "Score cumulé : %{base:.3f} → %{customdata:.3f}<extra></extra>"
            ),
            customdata=cumulative,
        )
    )

    # Ligne horizontale au niveau de base_value
    fig.add_hline(
        y=base_value,
        line_dash="dot",
        line_color="#7f8c8d",
        annotation_text=f"Valeur de base = {base_value:.3f}",
        annotation_position="top left",
        annotation_font=dict(size=10),
    )

    fig.update_layout(
        title=title,
        xaxis_title="Variables (top importance absolue)",
        yaxis_title="Contribution au score modèle",
        barmode="relative",
        template="plotly_dark",
        margin=dict(l=60, r=40, t=80, b=80),
        height=450,
    )

    return fig

    # ------------------------------------------

def build_prompt_for_transaction(proba, input_dict, top_features):
    decision = "bloquée (suspecte)" if proba >= 0.226 else "acceptée"
    top_str = "\n".join(
        [f"- {name} (contribution = {value:+.3f})" for name, value in top_features]
    )

    prompt = f"""
Tu es un analyste fraude dans une banque française.
Explique en français, de façon claire et compréhensible pour un client,
pourquoi la transaction suivante est {decision} par l’algorithme interne de détection de fraude.

Contexte :
- Probabilité estimée de fraude : {proba:.3f}
- Décision automatique : {decision}

Caractéristiques de la transaction :
- Montant : {input_dict.get("montant_transaction")}
- Type de magasin : {input_dict.get("type_magasin")}
- Heure de la transaction : {input_dict.get("heure_transaction")}h
- État du client : {input_dict.get("etat_client")}
- Âge du client : {input_dict.get("age_client")}
- Distance domicile–magasin : {input_dict.get("distance_domicile_magasin")}

Principaux éléments ayant influencé la décision :
{top_str}

Consignes :
- Ton professionnel mais pédagogique.
- Ne parle pas de SHAP.
- Ne parle pas de XGBoost.
- Si tu cites la variable "if_score" ou un score interne similaire,
  présente-la comme un indicateur du caractère inhabituel ou anormal de la transaction,
  pas comme un 'score interne' opaque.
- Mentionne les éléments qui augmentent le risque et ceux qui le réduisent.
- Rédige 1 à 2 paragraphes maximum.
"""
    return prompt


@st.cache_data(show_spinner=False)
def call_llm_explanation(tx_id, proba, shap_values, input_dict, feature_names, k_top=5):
    # 1) Si déjà en cache → on renvoie directement
    tx_id = str(tx_id)
    if tx_id in EXPLANATION_CACHE:
        return EXPLANATION_CACHE[tx_id]

    # 2) Calcul normal
    shap_values = np.array(shap_values).flatten()

    idx_sorted = np.argsort(-np.abs(shap_values))
    top_idx = idx_sorted[:k_top]
    top_features = [(feature_names[i], float(shap_values[i])) for i in top_idx]

    prompt = build_prompt_for_transaction(proba, input_dict, top_features)

    try:
        print("API appelée")
        completion = client.chat.completions.create(
            model="sonar-pro",
            messages=[
        {
            "role": "system",
            "content": (
                "Tu es un assistant data qui explique un modèle de fraude. "
                "Ne fais AUCUNE référence à des sources externes, "
                "n'ajoute PAS de citations ni de numéros entre crochets comme [1], [2], [3]. "
                "Ne mentionne pas de recherche web."
                ),
            },
            {
                "role": "user",
                "content": prompt,
         },
        ],
        temperature=0.3,
)

        content = completion.choices[0].message.content
        explanation = content if content else "Aucune explication n'a été générée."

    except Exception as e:
        explanation = f"Erreur lors de la génération de l'explication : {e}"

    # 3) On met à jour le cache en mémoire + disque
    EXPLANATION_CACHE[tx_id] = explanation
    save_explanation_cache()

    return explanation



# ------------------------------------------
# Onglet 1
# ------------------------------------------
def tab_ml_metrics():
    st.header("Métriques du modèle")

    st.markdown(
        """
        <div style="font-size:0.95rem; line-height:1.5;">
        Cet onglet rassemble les principaux éléments de pilotage du modèle de détection
        de fraude&nbsp;: métriques globales, matrice de confusion, courbes Précision–Rappel
        et courbes de calibration. L’objectif est de vérifier que le modèle discrimine
        correctement les transactions frauduleuses, que le seuil de décision retenu reste
        cohérent avec les coûts métier, et que les probabilités prédites constituent une
        base fiable pour la prise de décision opérationnelle.
        </div>
        """,
        unsafe_allow_html=True,
        )


    y_train = joblib.load(PROCESSED_DATA / "y_train.joblib")
    train_proba = joblib.load(PROCESSED_DATA / "train_proba.joblib")
    y_test = joblib.load(PROCESSED_DATA / "y_test_for_shap.joblib")
    test_proba = joblib.load(PROCESSED_DATA / "test_proba.joblib")
    seuil_decision = 0.226

    st.subheader("")

    col_left, col_right = st.columns([1.4, 1], gap="medium")

    with col_left:
        afficher_metriques_binaires(
            y_true=y_test,
            y_proba=test_proba,
            seuil_decision=seuil_decision,
            title="Métriques globales",
        )

        st.markdown(
        """
        <p style="font-size:0.9rem; color:#cccccc; margin-top:0.5rem;">
        Sur la période de test (oct.–déc. 2020), le modèle conserve un très
        fort pouvoir discriminant (ROC‑AUC ≈ 0,99) tout en maintenant un
        Brier score très faible, signe d’une bonne calibration des probabilités.
        Avec un recall autour de 79 % et une average precision d’environ 0,72,
        le compromis retenu est cohérent avec un contexte de fraude rare mais
        coûteuse, où l’on cherche à capter un maximum de cas tout en évitant
        un volume d’alertes excessif.
        </p>
        """,
        unsafe_allow_html=True,
        )

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
        <p style="font-size:0.9rem; color:#cccccc; margin-top:0.5rem;">
        Sur l’échantillon test, le modèle détecte la majorité des fraudes
        (737 TP pour 199 FN), tout en limitant fortement les blocages à tort
        au regard du volume total de transactions (740 FP pour plus de
        279 000 légitimes). Le ratio d’environ 1 faux positif pour 1 vrai
        positif est acceptable dans le cadre de la structure de coûts retenue,
        où une fraude non détectée est jugée cinq fois plus coûteuse qu’une
        alerte injustifiée.
        </p>
        """,
        unsafe_allow_html=True,
        )


    st.markdown("---")

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
            <p style="font-size:0.9rem; color:#cccccc; margin-top:0.5rem;">
            La courbe Précision–Rappel montre qu’il existe un large éventail
            de seuils pour lesquels le modèle conserve un rappel élevé, au
            prix d’une précision plus modérée. Le seuil de 0,226 retenu sur
            l’échantillon de validation privilégie volontairement la captation
            de fraudes (recall ≈ 79 %) au détriment d’une précision parfaite,
            conformément à la hiérarchie des coûts où les fraudes non détectées
            sont beaucoup plus pénalisantes que les fausses alertes.
            </p>
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
            <p style="font-size:0.9rem; color:#cccccc; margin-top:0.5rem;">
            Les courbes de calibration et les Brier scores très faibles
            indiquent que les probabilités prédites restent globalement
            cohérentes entre train, validation et test. La légère tendance à
            surestimer le risque en test s’explique par la baisse du taux de
            fraude sur la période récente, plus que par un surapprentissage
            du modèle. Cela permet d’utiliser les scores comme une véritable
            échelle de risque, exploitable pour fixer le seuil et prioriser
            les investigations métier.
            </p>
            """,
            unsafe_allow_html=True,
        )


# ------------------------------------------
# Onglet 2
# ------------------------------------------
def tab_business_metrics():
    st.header("Métriques métier & coût du seuil")
    
    st.markdown(
        """
        <div style="font-size:0.95rem; line-height:1.5;">
        Cet onglet permet d’analyser le modèle sous un angle <b>métier</b> plutôt que purement
        statistique. La courbe de coût total montre comment le choix du seuil de décision
        impacte directement les euros dépensés&nbsp;: d’un côté les faux positifs
        (transactions légitimes bloquées), de l’autre les fraudes non détectées.
        Les indicateurs métier présentés sous le graphique (taux de capture de fraude,
        taux de perte, taux d’alerte, ratio faux positifs&nbsp;/ vrais positifs, etc.)
        aident à juger si ce compromis coût&nbsp;/ protection reste acceptable pour la banque.
        </div>
        """,
        unsafe_allow_html=True,
    )


    # Chargement des données de validation
    y_validation = joblib.load(PROCESSED_DATA / "y_validation.joblib")
    proba_validation = joblib.load(PROCESSED_DATA / "proba_validation.joblib")
    cost_optimizer = joblib.load(PROCESSED_DATA / "cost_optimizer.joblib")

    # Coûts métier fixés (25 € FP, 125 € FN)
    cost_fp = 25
    cost_fn = 125

    seuils = cost_optimizer.seuils_
    costs = cost_optimizer.costs_
    best_t = cost_optimizer.get_best_threshold()
    best_cost = cost_optimizer.get_best_cost()

    st.subheader("")

    # Graphique Plotly interactif
    fig = plot_cost_threshold_curve(
        seuils=seuils,
        costs=costs,
        best_t=best_t,
        best_cost=best_cost,
    )
    st.plotly_chart(fig, use_container_width=True)

    # Court texte d’interprétation
    st.markdown(
        f"""

        Le point rouge correspond au seuil de décision retenu sur l'échantillon de validation
        (≈ {best_t:.3f}), pour lequel le coût total estimé est minimal (≈ {best_cost:,.0f} €).
        La courbe montre comment un seuil plus bas augmente rapidement les faux positifs
        (coût client et opérationnel), tandis qu’un seuil plus élevé laisse passer davantage
        de fraudes non détectées, ce qui renchérit le coût global pour la banque.
        """
    )
    
        # Bloc métriques métier détaillées sous le graphique
    col_left, col_right = st.columns(2)

    with col_left:
        st.markdown(
            """
            **Indicateurs métier**

            | Indicateur                     | Valeur                        |
            |--------------------------------|-------------------------------|
            | Fraud Capture Rate             | 78,7 %                        |
            | Fraud Loss Rate                | 21,3 %                        |
            | Précision                      | 49,9 % (≈ 1 alerte sur 2)     |
            | Alert Rate                     | 0,52 % des transactions       |
            | False Positive Rate            | 0,26 %                        |
            | Ratio FP / TP                  | 1 (1 faux positif pour 1 TP)  |
            """
        )

    with col_right:
        st.markdown(
            """
            **Lecture des résultats**

            Le modèle détecte près de 8 fraudes sur 10 tout en ne mettant en alerte
            qu’environ 0,5 % des transactions, ce qui reste compatible avec une
            exploitation opérationnelle. La précision à 49,9 % signifie qu’une alerte
            sur deux correspond effectivement à une fraude, un niveau acceptable
            compte tenu de la rareté du phénomène et de la hiérarchie des coûts.
            Le ratio FP/TP ≈ 1 traduit un équilibre raisonnable entre pertes évitées
            et irritations clients, dans un contexte où une fraude non détectée
            reste nettement plus coûteuse qu’un blocage injustifié.
            """
        )



# ------------------------------------------
# Onglet 3
# ------------------------------------------
def tab_transactions_and_explanations():
    shap_values_global = joblib.load(MODEL_DATA / "shap_values_model.joblib")

    st.header("Transactions et explications locales")

    st.markdown(
        """
        <div style="font-size:0.95rem; line-height:1.5;">
        Cet onglet permet d’explorer des transactions individuelles et de comprendre,
        pour chacune d’elles, <b>pourquoi</b> le modèle attribue un certain score de fraude.
        On combine ici une explication locale via SHAP et une reformulation métier
        générée par un LLM.
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
            Sur l’ensemble des transactions, **le montant** ressort comme le facteur
            le plus influent sur le score de fraude, loin devant les autres
            variables. Les créneaux horaires (par exemple 22h–23h), certains
            types de magasins (transport, grande distribution) et l’**if_score**
            apportent également une contribution notable au risque moyen.
            
            Concrètement, cela signifie que le modèle s’appuie d’abord sur des
            signaux simples et intuitifs (montant, heure, type de commerce),
            complétés par un indicateur d’**anormalité de la transaction**
            (if_score), pour différencier les opérations usuelles de celles qui
            ressemblent statistiquement à des fraudes.
            """
        )

    # 1. Chargement d'un petit échantillon
    sample_data = load_sample_data().copy()
    sample_data = sample_data.reset_index(drop=True)
    sample_data.insert(0, "id_ligne", sample_data.index + 1)  # 1 à 10

    st.markdown("**Sélectionnez une transaction pour voir l’explication détaillée :**")
    st.dataframe(sample_data, hide_index=True, use_container_width=True)

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

    # 3. Scoring + SHAP local
    proba, shap_values, X_proc = score_transaction(row.to_dict())
    st.markdown(f"**Probabilité estimée de fraude :** {proba:.3f}")

    col_left, col_right = st.columns(2, gap="medium")

    # ----- Colonne de gauche : waterfall SHAP -----
    with col_left:
        st.subheader("Explication locale (SHAP – waterfall)")

        shap_arr = np.array(shap_values)
        if shap_arr.ndim == 1:
            sv_row = shap_arr
        else:
            sv_row = shap_arr[0, :]

        feature_names = X_proc.columns.tolist()
        explainer = joblib.load(PROCESSED_DATA / "explainer.joblib")

        shap_expl = shap.Explanation(
            values=sv_row,
            base_values=explainer.expected_value,
            data=X_proc.iloc[0, :].values,
            feature_names=feature_names,
        )

        plt.clf()
        plt.figure(figsize=(10, 8))
        shap.plots.waterfall(shap_expl, max_display=8, show=False)
        fig = plt.gcf()
        st.pyplot(fig, bbox_inches="tight", dpi=130)

    # ----- Colonne de droite : explication LLM -----
    with col_right:
        st.subheader("Explication reformulée pour le métier")

        if st.button("Générer une explication métier (LLM)"):
            with st.spinner("Génération de l'explication en langage naturel..."):
                shap_1d = sv_row
                explanation = call_llm_explanation(
                    tx_id=int(selected_id),
                    proba=proba,
                    shap_values=shap_1d,
                    input_dict=row.to_dict(),
                    feature_names=feature_names,
                    k_top=5,
                )
            st.markdown(explanation)


# ------------------------------------------
# Onglet 4
# ------------------------------------------
import numpy as np
import pandas as pd
import random
import joblib


def sample_from_tranche(label, bounds_dict):
    low, high = bounds_dict[label]
    return float(np.random.uniform(low, high))


def tab_simulator():
    st.header("Votre transaction serait-elle considérée comme frauduleuse ?")

    # Liste complète des états vus au training (nettoyés)
    values_etats_client = joblib.load(PROCESSED_DATA / "values_etat_client.joblib")
    values_etats_client = (
        pd.Series(values_etats_client)
        .dropna()
        .astype(str)
        .tolist()
    )

    # États qu'on propose explicitement dans l'UI
    etats_principaux = ["CA", "NY", "TX", "FL", "NJ"]

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

        distance_tranche = st.selectbox(
            "Distance domicile–magasin (km, tranche)",
            [
                "0 – 1 km",
                "1 – 5 km",
                "5 – 20 km",
                "20 – 50 km",
                "50 – 200 km",
            ],
        )

    with col_client:
        st.subheader("Profil client")

        etat_ui = st.selectbox(
            "État du client",
            etats_principaux + ["autre"],
        )

        population_tranche = st.selectbox(
            "Population de la ville de résidence (tranche)",
            [
                "< 10 000",
                "10 000 – 50 000",
                "50 000 – 200 000",
                "200 000 – 1 000 000",
                "> 1 000 000",
            ],
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

        distance_bounds = {
            "0 – 1 km": (0, 1),
            "1 – 5 km": (1, 5),
            "5 – 20 km": (5, 20),
            "20 – 50 km": (20, 50),
            "50 – 200 km": (50, 200),
        }

        population_bounds = {
            "< 10 000": (0, 10000),
            "10 000 – 50 000": (10000, 50000),
            "50 000 – 200 000": (50000, 200000),
            "200 000 – 1 000 000": (200000, 1000000),
            "> 1 000 000": (1000000, 3000000),
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
        distance = sample_from_tranche(distance_tranche, distance_bounds)
        population_ville_client = sample_from_tranche(
            population_tranche, population_bounds
        )
        age = int(round(sample_from_tranche(age_tranche, age_bounds)))

        # Gestion de "autre" : on choisit un état réel non déjà dans etats_principaux
        if etat_ui == "autre":
            candidats = [e for e in values_etats_client if e not in etats_principaux]
            if not candidats:  # sécurité
                candidats = values_etats_client
            etat_effectif = random.choice(candidats)
        else:
            etat_effectif = etat_ui

        # Dictionnaire final envoyé au modèle (types forcés)
        input_dict = {
            "type_magasin": str(type_magasin),
            "montant_transaction": float(montant),
            "etat_client": str(etat_effectif),
            "population_ville_client": float(population_ville_client),
            "heure_transaction": int(heure),
            "distance_domicile_magasin": float(distance),
            "age_client": int(age),
        }

        # Debug temporaire si besoin
        # X_raw = pd.DataFrame([input_dict])
        # st.write("X_raw dtypes :", X_raw.dtypes)
        # st.write("X_raw :", X_raw)

        # 1. Scoring + SHAP
        proba, shap_values, X_proc = score_transaction(input_dict)

        shap_arr = np.array(shap_values)
        if shap_arr.ndim == 1:
            sv_row = shap_arr
        else:
            sv_row = shap_arr[0, :]

        feature_names = X_proc.columns.tolist()
        explainer = joblib.load(PROCESSED_DATA / "explainer.joblib")
        base_value = float(explainer.expected_value)

        seuil = 0.226
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
                label="Décision (seuil 0,226)",
                value=decision,
            )

        # Récap des valeurs utilisées
        st.markdown("#### Valeurs utilisées pour la simulation")
        recap = {
            "type_magasin": [type_magasin],
            "etat_client": [etat_effectif],
            "montant_transaction": [round(montant, 2)],
            "distance_domicile_magasin": [round(distance, 2)],
            "population_ville_client": [int(population_ville_client)],
            "age_client": [age],
            "heure_transaction": [heure],
        }
        st.table(pd.DataFrame(recap))

        st.markdown("---")

        # 2. Explication locale SHAP uniquement
        st.subheader("Explication locale (SHAP – waterfall)")
        fig_shap = plot_local_shap_plotly(
            shap_values=sv_row,
            feature_names=feature_names,
            base_value=base_value,
            top_k=8,
            title="Contribution des principales variables au score",
        )
        st.plotly_chart(fig_shap, use_container_width=True)

 
# ------------------------------------------
# Streamlit Main
# ------------------------------------------
def main():
    st.set_page_config(page_title="Détection de fraude carte bancaire", layout="wide")

    st.title("Fraude carte bancaire – Modèle XGBoost explicable")

    tab1, tab2, tab3, tab4 = st.tabs([
        "Métriques ML",
        "Métriques métier & coûts",
        "Transactions & explications",
        "Simulateur de transaction",
    ])

    with tab1:
        tab_ml_metrics()
    with tab2:
        tab_business_metrics()
    with tab3:
        tab_transactions_and_explanations()
    with tab4:
        tab_simulator()


if __name__ == "__main__":
    main()
# ------------------------------------------
