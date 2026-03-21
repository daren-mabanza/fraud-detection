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
    decision = "bloquée" if proba >= 0.226 else "acceptée"
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
                f"{label} = {real_value} ({anomalie_niveau}) : "
                f"ce score mesure à quel point cette transaction ressemble à des transactions rares "
                f"dans l'ensemble de notre base de données, tous clients confondus. "
                f"Un score négatif signifie que la transaction est statistiquement inhabituelle "
                f"par rapport aux millions de transactions traitées globalement. "
                f"Ce score ne tient PAS compte de l'historique personnel de ce client : "
                f"même si ce client effectue régulièrement ce type d'achat, "
                f"un score négatif reste un signal d'alerte au niveau population."
            )
        elif clean_name == 'heure_transaction':
            description = f"{label} = {real_value}h (heure à laquelle le paiement a été effectué)"
        elif clean_name == 'mont_transaction':
            description = f"{label} = {real_value} € (montant exact payé)"
        elif clean_name == 'age_client':
            description = f"{label} = {real_value} ans"
        elif clean_name == 'etat_client':
            description = f"{label} = {real_value} (localisation géographique déclarée du client)"
        else:
            description = f"{label} = {real_value}"

        if shap_value > 0:
            facteurs_risque.append(f"  - {description} → a AUGMENTÉ le risque de fraude")
        else:
            facteurs_rassurants.append(f"  - {description} → a RÉDUIT le risque de fraude")

    bloc_risque = "\n".join(facteurs_risque) if facteurs_risque else "  - aucun facteur de risque majeur"
    bloc_rassurant = "\n".join(facteurs_rassurants) if facteurs_rassurants else "  - aucun facteur rassurant détecté"

    contexte_decision = (
        f"Le système a calculé un score de risque de fraude de {proba_pct}. "
        f"Le seuil de blocage est fixé à 23%. "
        f"{'Ce score dépasse le seuil, donc la transaction a été bloquée.' if proba >= 0.226 else 'Ce score est sous le seuil, donc la transaction a été acceptée.'}"
    )

    prompt = f"""
CONTEXTE :
Un client appelle sa banque au sujet d'un paiement de {input_dict.get('montant_transaction', 'N/A')} €
effectué à {input_dict.get('heure_transaction', 'N/A')}h dans un commerce de type "{input_dict.get('type_magasin', 'N/A')}".
Le client a {input_dict.get('age_client', 'N/A')} ans et est localisé dans l'état : {input_dict.get('etat_client', 'N/A')}.

DÉCISION DU SYSTÈME : paiement {decision}.
{contexte_decision}

FACTEURS QUI ONT AUGMENTÉ LE RISQUE :
{bloc_risque}

FACTEURS QUI ONT RÉDUIT LE RISQUE :
{bloc_rassurant}

NOTE SUR L'INDICE D'ANOMALIE :
Un indice négatif signifie que la transaction est rare dans notre base globale, tous clients confondus.
Ce n'est pas "ce client n'a pas l'habitude" mais "ce type de transaction est statistiquement peu fréquent
parmi l'ensemble de nos clients". Ne confonds jamais les deux dans ton explication.

TA MISSION :
Rédige 2 à 3 phrases en texte brut pour expliquer concrètement cette décision au client.
Commence par ce qui a le plus pesé dans la décision (facteurs de risque si bloqué, facteurs rassurants si accepté).
Sois précis : cite les valeurs réelles (montant exact, heure exacte, type de commerce).
Conclus par la décision finale et ce qu'elle signifie pour le client.

RÈGLES NON NÉGOCIABLES :
- Texte brut uniquement : zéro crochet, zéro [1], zéro [2], zéro puce, zéro tiret, zéro gras.
- Tu n'es pas un moteur de recherche, tu ne cites aucune source. Aucun numéro de référence.
- Zéro jargon : jamais de "SHAP", "XGBoost", "Isolation Forest", "score", "modèle", "algorithme", "seuil".
- Zéro introduction, zéro formule de politesse, zéro "Bonjour", zéro "Je suis ravi".
- Langage naturel, direct, comme un conseiller bancaire expérimenté au téléphone.
- Termine toujours ta dernière phrase avec un point.
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