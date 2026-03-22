# ================================
# Import des packages nécéssaires
# ================================
from config import ROOT
import streamlit as st
import numpy as np
from openai import OpenAI  
import joblib                                        

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
    
    cost_optimizer = joblib.load(JOBLIB_DATA / "cost_optimizer.joblib")
    seuil = cost_optimizer.get_best_threshold()
    
    decision = "bloquée" if proba >= seuil else "acceptée"
    proba_pct = f"{proba:.0%}"

    feature_mapping = {
        'type_magasin_grocery_pos': 'un achat en supermarché',
        'type_magasin_gas_transport': 'un achat en station-service ou transport',
        'type_magasin_shopping_pos': 'un achat dans un magasin de shopping',
        'type_magasin_neuro_transac': 'un achat en supérette',
        'type_magasin_misc_pos': 'un achat dans un commerce divers',
        'mont_transaction': 'le montant de la transaction',
        'heure_transaction': "l'heure de la transaction",
        'if_score': "l'indice d'anomalie statistique de la transaction",
        'etat_client': "l'état géographique du client",
        'age_client': "l'âge du client"
    }

    facteurs_risque = []
    facteurs_rassurants = []

    for name, shap_value in top_features[:5]:
        clean_name = name.split('_scaler_')[-1]
        label = feature_mapping.get(clean_name, clean_name)
        real_value = input_dict.get(clean_name, "non disponible")

        if clean_name == 'if_score':
            anomalie_niveau = (
                "très anormale" if real_value < -0.1
                else "légèrement atypique" if real_value < 0
                else "normale"
            )
            description = (
                f"{label} = {real_value} ({anomalie_niveau}). "
                f"Ce score mesure à quel point la transaction est rare parmi l'ensemble des transactions observées. "
                f"Un score négatif indique un comportement inhabituel à l'échelle globale."
            )
        elif clean_name == 'heure_transaction':
            description = f"{label} = {real_value}h"
        elif clean_name == 'mont_transaction':
            description = f"{label} = {real_value} €"
        elif clean_name == 'age_client':
            description = f"{label} = {real_value} ans"
        elif clean_name == 'etat_client':
            description = f"{label} = {real_value}"
        else:
            description = f"{label} = {real_value}"

        poids = abs(shap_value)

        if shap_value > 0:
            facteurs_risque.append(f"{description} (impact {poids:.3f})")
        else:
            facteurs_rassurants.append(f"{description} (impact {poids:.3f})")

    bloc_risque = "\n".join(facteurs_risque) if facteurs_risque else "aucun facteur majeur"
    bloc_rassurant = "\n".join(facteurs_rassurants) if facteurs_rassurants else "aucun facteur notable"

    contexte_decision = (
        f"La probabilité estimée est de {proba_pct}. "
        f"Le seuil de blocage est fixé à 19%. "
        f"{'La transaction dépasse ce seuil et a donc été bloquée.' if proba >= seuil else 'La transaction reste sous ce seuil et a donc été acceptée.'}"
    )

    prompt = f"""
CONTEXTE MÉTIER :
Un client contacte sa banque suite à une transaction.

Transaction :
- Montant : {input_dict.get('mont_transaction', 'N/A')} €
- Heure : {input_dict.get('heure_transaction', 'N/A')}h
- Type de commerce : {input_dict.get('type_magasin', 'N/A')}
- Âge du client : {input_dict.get('age_client', 'N/A')} ans
- Localisation : {input_dict.get('etat_client', 'N/A')}

DÉCISION :
Paiement {decision}.
{contexte_decision}

----------------------------------------
INTERPRÉTATION DES FACTEURS :
----------------------------------------

Les éléments suivants expliquent la décision.

IMPORTANT :
- Les facteurs sont classés par importance.
- Plus l’impact est élevé, plus le facteur a pesé dans la décision.
- Tu dois expliquer la décision UNIQUEMENT à partir de ces éléments.

FACTEURS QUI ONT AUGMENTÉ LE RISQUE :
{bloc_risque}

FACTEURS QUI ONT RÉDUIT LE RISQUE :
{bloc_rassurant}

----------------------------------------
INTERPRÉTATION DE L'INDICE D'ANOMALIE :
----------------------------------------

L'indice d'anomalie (if_score) mesure à quel point cette transaction est inhabituelle
par rapport à l'ensemble des transactions observées.

- Une valeur négative signifie que la transaction est rare à l'échelle globale.
- Une valeur proche de zéro signifie qu'elle est normale.

ATTENTION :
Ce score ne regarde PAS l'historique du client.
Ce n'est pas "inhabituel pour ce client", mais "inhabituel parmi tous les clients".

----------------------------------------
CADRE D’INTERPRÉTATION :
----------------------------------------

- Un facteur qui augmente le risque rapproche la transaction de comportements déjà associés à de la fraude.
- Un facteur qui réduit le risque rapproche la transaction de comportements habituels.
- Il s'agit d'une comparaison statistique, pas d'une vérité absolue.

----------------------------------------
TA MISSION :
----------------------------------------

Explique la décision au client en 3-4 phrases maximum.

- Commence par les éléments les plus déterminants.
- Si la transaction est bloquée → commence par les facteurs de risque.
- Si elle est acceptée → commence par les éléments rassurants.
- Cite toujours des éléments concrets (montant, heure, type de commerce…).
- Fais un lien clair entre ces éléments et la décision.

----------------------------------------
RÈGLES STRICTES :
----------------------------------------

- Texte brut uniquement.
- Aucun symbole, aucune liste, aucun tiret.
- Aucun jargon technique.
- Interdiction des mots : modèle, algorithme, score, seuil, SHAP.
- Aucune invention : uniquement les facteurs fournis.
- Style naturel, comme un conseiller bancaire expérimenté.
- Maximum 4 phrases.
- Toujours terminer par un point.
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
                        "Tu es un conseiller fraude dans une banque française. "
                        "Tu reçois une analyse automatique d'une transaction et tu l'expliques au client en langage naturel. "
                        "Tu ne génères QUE du texte brut : aucun crochet, aucun [1] ou [2], aucune référence, aucune citation, aucune puce. "
                        "Tu n'es PAS un moteur de recherche. Tu ne fournis PAS de sources. "
                        "Tu vas droit au but : tu commences immédiatement par les faits de la transaction."
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