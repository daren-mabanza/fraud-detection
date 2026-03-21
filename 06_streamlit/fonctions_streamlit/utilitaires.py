# ================================
# Import des packages nécéssaires
# ================================

from config import ROOT
import json
import hashlib
import joblib
import streamlit as st
import pandas as pd
from sklearn import set_config
import numpy as np

# ==========================================
# Paramétrage de l'environnement de travail
# ==========================================

JOBLIB_DATA = ROOT / "01_data" / "03_joblib"
MODEL_DATA = ROOT / "04_model"
EXPLANATION_CACHE = {}
EXPLANATION_CACHE_PATH = JOBLIB_DATA / "llm_explanations_cache.json"

# =======================
# Création des fonctions
# =======================

    # 
def make_tx_key(row_dict):
    """
    Foncion permettant d'assurer la correspondance entre l'index de l'observation et l'interprétation renvoyé par le LLM meme si l'ordre des lignes change.
    """
    payload = json.dumps(row_dict, sort_keys=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


@st.cache_resource
def load_models():
    """
    Fonction permettant de charger les deux modèles :
        --> modele_propre : Modèle XGBoost calibré
        --> modele_shap : Modèle XGBoost non calibré
    """
    modele_propre = joblib.load(MODEL_DATA / "fraud_detection_model_xgboost.joblib")
    modele_shap = joblib.load(MODEL_DATA / "fraud_detection_xgb_pas_calibre.joblib")
 
    return modele_propre, modele_shap


@st.cache_data
def load_sample_data():
    """
    Fonction permettant de charger l'échantillon de données qui sera utilisé dans l'onglet 3 (Prédictions et explications).
    """
    
    sample_data = joblib.load(JOBLIB_DATA / "test_sample_onglet_3.joblib")

    return sample_data


def sample_from_tranche(label, bounds_dict):
    """
    Fonction retournant une valeur aléatoire dans une tranche de valeurs.
    """
    low, high = bounds_dict[label]
    return float(np.random.uniform(low, high))


def score_transaction(input_dict):
    """
    Fonction qui permet de calculer la probabilité de fraude et de récupérer les SHAP values associées à une observation.
    
    input_dict : dict des variables brutes (même format que X_train/X_test du notebook)
    renvoie : proba, shap_values, X_proc
    """
    
    modele_propre, modele_shap = load_models()
    explainer = joblib.load(JOBLIB_DATA / "explainer.joblib")

    # DataFrame brut
    X_raw = pd.DataFrame([input_dict])

    # Passage dans le pipeline (ici: on applique tout le pipeline "modele_shap" et on enlève juste l'estimateur final)
    set_config(transform_output="pandas")
    preprocess = modele_shap[:-1]
    X_proc = preprocess.transform(X_raw)

    # Proba avec le modèle calibré
    proba = modele_propre.predict_proba(X_raw)[:, 1][0]

    # SHAP sur le même espace que x_test_proc
    shap_values = explainer.shap_values(X_proc)

    return proba, shap_values, X_proc


def save_explanation_cache():
    """
    Les explications du cache sont sauvegardés afin de ne pas utiliser tous les jetons disponibles du LLM
    """
    
    with open(EXPLANATION_CACHE_PATH, "w", encoding="utf-8") as f:
        json.dump(EXPLANATION_CACHE, f, ensure_ascii=False, indent=2)

    # Chargement du cache déja existant (si existant)
try:
    with open(EXPLANATION_CACHE_PATH, "r", encoding="utf-8") as f:
        EXPLANATION_CACHE = json.load(f) # Le cache n'est pas vide : je l'utilise
except FileNotFoundError:
    EXPLANATION_CACHE = {}               # Si erreur alors cache vide 