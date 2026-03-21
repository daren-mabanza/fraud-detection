# ================================
# Import des packages nécéssaires
# ================================
from config import ROOT
import json
import streamlit as st
import numpy as np
from perplexity import Perplexity
from fonctions_streamlit.utilitaires import save_explanation_cache

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
    Charge la clé API Perplexity depuis le fichier JSON et renvoie un client Perplexity prêt à l'emploi.
    """
    api_key = st.secrets["PERPLEXITY_API_KEY"]
    
    return api_key


def build_prompt_for_transaction(proba, input_dict, top_features):
    """
    Fonction permettant de construire le prompt qui servira au LLM pour interpréter les transactions.
    """
    decision = "bloquée (suspecte)" if proba >= 0.226 else "acceptée"
    
    # Mapping des variables en lanagage naturel
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
    
    # Formatage des SHAP values pour que le LLM puisse les interpréter
    top_str = []
    for name, shap_value in top_features[:5]:
        # Traduction intelligente
        readable_name = feature_mapping.get(name.split('_scaler_')[-1], name)
        real_value = input_dict.get(name.split('_scaler_')[-1], "inhabituel")
        
        if shap_value > 0:
            assoc = "souvent associé à des comportements frauduleux"
        else:
            assoc = "généralement considéré comme normal"
            
        top_str.append(f"- {readable_name} = {real_value} ({assoc}, impact {shap_value:+.3f})")
    
    top_str = "\n".join(top_str)
    
    # Prompt d'instruction à suivre 
    prompt = f"""
Tu es un analyste fraude expérimenté dans une banque française.
Explique à un client, en langage simple et rassurant, pourquoi son paiement
a été {decision} par le système automatique.

TRANSACTION :
- Montant : {input_dict.get('montant_transaction', 'N/A')} €
- Type de magasin : {input_dict.get('type_magasin', 'N/A')}
- Heure : {input_dict.get('heure_transaction', 'N/A')} h

FACTEURS PRINCIPAUX (issus de l’analyse automatique) :
{top_str}

Consignes :
- Reste factuel et concentré uniquement sur cette transaction.
- Explique simplement quels éléments augmentent le risque et lesquels le réduisent.
- Utilise un vocabulaire courant : par exemple "magasin de shopping", "supermarché",
  "station-service", pas les noms techniques comme "shopping_pos" ou "gas_transport".
- Ne mentionne pas de termes techniques (modèle, algorithme, SHAP, XGBoost, calibration, API).
- Pas de pourcentages ni de statistiques globales, uniquement le cas concret du client.
- Formule ta réponse en 1 ou 2 courts paragraphes, ton professionnel mais accessible.
"""

    return prompt



@st.cache_data(show_spinner=False)
def call_llm_explanation(tx_key, proba, shap_values, input_dict, feature_names, k_top=5):
    """
    Fonction d'appel au LLM Sonar.
    """
    if tx_key in EXPLANATION_CACHE:
        return EXPLANATION_CACHE[tx_key]

    # Chargement des SHAP values et trie pour donner au LLM les variables les plus importantes
    shap_values = np.array(shap_values).flatten()
    idx_sorted = np.argsort(-np.abs(shap_values))
    top_idx = idx_sorted[:k_top]
    top_features = [(feature_names[i], float(shap_values[i])) for i in top_idx]

    # Appel à la fonction contenant le prompt
    prompt = build_prompt_for_transaction(
        proba=proba,
        input_dict=input_dict,
        top_features=top_features,
    )

    # Appel au LLM Sonar
    try:
        completion = get_llm_client().chat.completions.create(
            model="sonar-pro",
            messages=[
                {
                    "role": "system",
                    "content": (
                        "Tu es un analyste fraude bancaire. "
                        "Réponses courtes (175-200 mots), rassurantes, "
                        "professionnelles. Jamais de jargon technique "
                        "(SHAP, XGBoost, calibration, API). "
                        "Toujours lier caractéristiques transaction + décision + facteurs risque."
                    ),
                },
                {"role": "user", "content": prompt},
            ],
            temperature=0.2,
            max_tokens=250,
        )
        content = completion.choices[0].message.content.strip()
        explanation = content if content else "Explication temporaire indisponible."
    except Exception as e:
        explanation = (
            f"🚦 Le service LLM est temporairement saturé, veuillez réessayer plus tard."
            f"Analyse automatique : {'suspecte' if proba >= 0.226 else 'légitime'} "
            f"(proba {proba:.1%})."
        )

    EXPLANATION_CACHE[tx_key] = explanation
    save_explanation_cache()
    return explanation