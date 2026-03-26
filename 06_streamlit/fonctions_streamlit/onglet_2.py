# ================================
# Import des packages nécéssaires
# ================================

from config import ROOT
import streamlit as st
import joblib
from fonctions_streamlit.viz_onglet_2 import plot_cost_threshold_curve

# ==========================================
# Paramétrage de l'environnement de travail
# ==========================================

JOBLIB_DATA = ROOT / "01_data" / "03_joblib"

# =========
# Onglet 2
# =========

def onglet_2():
    st.header("Métriques métier & coût du seuil")
    
    st.markdown(
        """
        <div style="font-size:1.55rem;">
    Cet onglet permet d’analyser le modèle sous un angle <strong>métier</strong> plutôt que purement
    statistique. La <strong>courbe de coût total</strong> montre comment le choix du seuil de décision
    impacte directement les euros dépensés : d’un côté les <strong>faux positifs</strong>
    (transactions légitimes bloquées), de l’autre les <strong>fraudes non détectées</strong>.
    Les <strong>indicateurs métier</strong> présentés sous le graphique (taux de capture de fraude,
    taux de perte, taux d’alerte, ratio faux positifs / vrais positifs, etc.)
    aident à juger si ce compromis <strong>coût / protection</strong> reste acceptable pour la banque.
    
    Les coûts ont été fixés à <strong>25 € pour un faux positif</strong> et <strong>125 € pour un faux négatif</strong>.
    Ces ordres de grandeur s’inspirent des pratiques du secteur bancaire, où une fraude non détectée
    (coûts de remboursement, traitement et impact client) est nettement plus coûteuse qu’un blocage injustifié.
    Le ratio d’environ <strong>5x</strong> reflète cette hiérarchie métier et oriente le modèle vers une meilleure
    détection des fraudes tout en maintenant un volume d’alertes acceptable.
        </div>
        """,
        unsafe_allow_html=True
    )


    # Chargement des données de validation
    cost_optimizer = joblib.load(JOBLIB_DATA / "cost_optimizer.joblib")

    # Paramétrage des seuils et couts
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
    <div style="font-size:1.1rem;">
    Le <strong>point rouge</strong> correspond au <strong>seuil de décision</strong> retenu sur l'échantillon de validation
    (≈ {best_t:.3f}), pour lequel le <strong>coût total estimé</strong> est minimal (≈ {best_cost:,.0f} €).
    La courbe montre comment un <strong>seuil plus bas</strong> augmente rapidement les <strong>faux positifs</strong>
    (coût client et opérationnel), tandis qu’un <strong>seuil plus élevé</strong> laisse passer davantage
    de <strong>fraudes non détectées</strong>, ce qui renchérit le <strong>coût global</strong> pour la banque.
    </div>
    """,
    unsafe_allow_html=True,
    )


    st.markdown("---")
    
    
    # Bloc métriques métier détaillées sous le graphique
    st.subheader("Indicateurs métiers")

    st.markdown(
    """
    <style>
        .business-metrics-table {
            border-collapse: collapse;
            margin-top: 0.75rem;
            background-color: #f9fafb;
            color: #111827;
            min-width: 420px;
            margin-left: auto;
            margin-right: auto;
        }
        .business-metrics-table th,
        .business-metrics-table td {
            border: 1px solid #e5e7eb;
            padding: 10px 16px;
            text-align: left;
            font-size: 1.0rem;
        }
        .business-metrics-header {
            background-color: #e5e7eb;
            font-weight: 600;
        }
        .business-metrics-name {
            font-weight: 500;
        }
    </style>

    <table class="business-metrics-table">
        <tr class="business-metrics-header">
            <th>Indicateur</th>
            <th>Valeur</th>
        </tr>
        <tr>
            <td class="business-metrics-name">Fraud Capture Rate</td>
            <td>78,6 %</td>
        </tr>
        <tr>
            <td class="business-metrics-name">Fraud Loss Rate</td>
            <td>22,4 %</td>
        </tr>
        <tr>
            <td class="business-metrics-name">Précision</td>
            <td>50 % (≈ 1 alerte sur 2)</td>
        </tr>
        <tr>
            <td class="business-metrics-name">Alert Rate</td>
            <td>0,51 % des transactions</td>
        </tr>
        <tr>
            <td class="business-metrics-name">False Positive Rate</td>
            <td>0,24 %</td>
        </tr>
        <tr>
            <td class="business-metrics-name">Ratio FP / TP</td>
            <td>1 (1 faux positif pour 1 TP)</td>
        </tr>
    </table>
    """,
    unsafe_allow_html=True,
    )


    st.markdown(
    """
    <div style="font-size:1.1rem;">
    Le modèle détecte près de <strong>8 fraudes sur 10</strong> tout en ne mettant en alerte
    qu'environ <strong>0,51 % des transactions</strong>, ce qui reste compatible avec une
    exploitation opérationnelle. La <strong>précision à 49,6 %</strong> signifie qu'une alerte
    sur deux correspond effectivement à une fraude, un niveau acceptable
    compte tenu de la rareté du phénomène et de la hiérarchie des coûts.
    Le <strong>ratio FP/TP ≈ 1</strong> traduit un équilibre raisonnable entre <strong>pertes évitées</strong>
    et <strong>irritations clients</strong>, dans un contexte où une <strong>fraude non détectée</strong>
    reste nettement plus coûteuse qu'un <strong>blocage injustifié</strong>.
    </div>
    """,
    unsafe_allow_html=True,
    )