# ================================
# Import des packages nécéssaires
# ================================
from config import ROOT
import streamlit as st
import numpy as np
from openai import OpenAI                                          

# ==========================================
# Paramétrage de l'environnement de travail
# ==========================================

JOBLIB_DATA = ROOT / "01_data" / "03_joblib"
EXPLANATION_CACHE = {}
EXPLANATION_CACHE_PATH = JOBLIB_DATA / "llm_explanations_cache.json"

# =======================
# Création des fonctions
# =======================

def get_llm_client():
    """
    Charge la clé API Perplexity depuis les secrets Streamlit et renvoie un client prêt à l'emploi.
    """
    api_key = st.secrets["PERPLEXITY_API_KEY"]
    return OpenAI(                                                 
        api_key=api_key,
        base_url="https://api.perplexity.ai"
    )

def build_prompt_for_transaction(proba, input_dict, top_features):
    """
    Fonction permettant de construire le prompt qui servira au LLM pour interpréter les transactions.
    """
    decision = "bloquée (suspecte)" if proba >= 0.226 else "acceptée"

    feature_mapping = {
        'type_magasin_grocery_pos': 'supermarché/grande distribution',
        'type_magasin_gas_transport': 'station-service/transport',
        'type_magasin_shopping_pos': 'magasin de shopping',
        'type_magasin_neuro_transac': 'supérette/neuro',
        'type_magasin_misc_pos': 'magasin divers',
        'mont_transaction': 'montant du paiement',
        'heure_transaction': "heure de la transaction",
        'if_score': 'anomalie transaction',
        'etat_client': 'état géographique client',
        'age_client': 'âge du client'
    }

    top_str = []
    for name, shap_value in top_features[:5]:
        readable_name = feature_mapping.get(name.split('_scaler_')[-1], name)
        real_value = input_dict.get(name.split('_scaler_')[-1], "inhabituel")

        if shap_value > 0:
            assoc = "souvent associé à des comportements frauduleux"
        else:
            assoc = "généralement considéré comme normal"

        top_str.append(f"- {readable_name} = {real_value} ({assoc}, impact {shap_value:+.3f})")

    top_str = "\n".join(top_str)

    prompt = f"""
Un client te contacte au sujet de son paiement de {input_dict.get('montant_transaction', 'N/A')} € 
dans un {input_dict.get('type_magasin', 'N/A')} à {input_dict.get('heure_transaction', 'N/A')}h, 
qui a été {decision}.

Voici les facteurs qui ont influencé cette décision :
{top_str}

Explique-lui en 3 à 4 phrases ce qui a concrètement fait monter le risque et ce qui l'a fait baisser, 
en te basant UNIQUEMENT sur ces facteurs. 

Règles absolues :
- Commence directement par les facteurs de risque, sans introduction ni formule de politesse.
- Cite chaque facteur avec sa valeur réelle (montant, heure, type de magasin...).
- Sépare clairement ce qui a augmenté le risque de ce qui l'a réduit.
- Termine par une phrase courte sur la décision finale et pourquoi.
- Texte brut uniquement : aucun crochet, aucune puce, aucun [1] ou [2], aucun gras.
- Langage simple et naturel. Jamais de jargon technique (SHAP, algorithme, modèle, score).
- Si tu mentionnes l'indicateur d'anomalie de la transaction (if_score), précise bien que cette transaction 
  est inhabituelle par rapport à l'ensemble des transactions de nos clients en général, 
  et NON par rapport à l'historique personnel de ce client.
"""

    return prompt


@st.cache_data(show_spinner=False, ttl=3600)
def call_llm_explanation(tx_key, proba, shap_values, input_dict, feature_names, k_top=5):
    """
    Fonction d'appel au LLM Sonar.
    """
    if tx_key in EXPLANATION_CACHE:
        return EXPLANATION_CACHE[tx_key]

    shap_values = np.array(shap_values).flatten()
    idx_sorted = np.argsort(-np.abs(shap_values))
    top_idx = idx_sorted[:k_top]
    top_features = [(feature_names[i], float(shap_values[i])) for i in top_idx]

    prompt = build_prompt_for_transaction(
        proba=proba,
        input_dict=input_dict,
        top_features=top_features,
    )

    try:
        completion = get_llm_client().chat.completions.create(
            model="sonar-pro",
            messages=[
                {
                    "role": "system",
                    "content": (
                        "Tu es un conseiller fraude bancaire. "
                        "Tu expliques des décisions de paiement en te basant uniquement sur les faits concrets de la transaction. "
                        "Texte brut uniquement, sans crochets, sans numéros de citation, sans mise en forme. "
                        "Tu vas droit au but : facteurs de risque d'abord, facteurs rassurants ensuite, conclusion."
                    ),
                },
                {"role": "user", "content": prompt},
            ],
            temperature=0.2,
            max_tokens=450,
        )
        content = completion.choices[0].message.content.strip()
        explanation = content if content else "Explication temporaire indisponible."
    except Exception as e:
        explanation = (                                            # MODIFIÉ : affiche l'erreur réelle + supprime save_explanation_cache()
            f"🚦 Erreur LLM : {e} | "
            f"Analyse : {'suspecte' if proba >= 0.226 else 'légitime'} "
            f"(proba {proba:.1%})."
        )

    EXPLANATION_CACHE[tx_key] = explanation
    return explanation                                             # MODIFIÉ : save_explanation_cache() supprimé (crash sur Streamlit Cloud)