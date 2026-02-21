# =========================================================================
from pathlib import Path

import pandas as pd
import numpy as np
from fonctions_perso.data_manipulation import multi_astype

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.compose import make_column_selector

from sklearn.model_selection import train_test_split

from xgboost import XGBClassifier

from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import randint, uniform, loguniform
from sklearn.calibration import CalibratedClassifierCV

from fonctions_perso.machine_learning import IsolationForestCustom
from sklearn.preprocessing import StandardScaler, OneHotEncoder

from fonctions_perso.machine_learning import ThresholdCostOptimizer

from sklearn import set_config
set_config(transform_output="pandas")

import joblib
# ==========================================================================
ROOT = Path.cwd().parents[0]

RAW_DATA = ROOT / "01_data" / "01_raw"
PROCESSED_DATA = ROOT / "01_data" / "02_processed"
# ==========================================================================

def xgboost_model():

    # Import des données
    data_fraud = pd.read_parquet(PROCESSED_DATA / "data_fraud_eda_processed.parquet").reset_index(drop = True)
    df = data_fraud.copy()

    print("Import des données : OK")
    print("="*50)


    # Modification des types pour les variables à encoder (en cat)
    multi_astype(
    df,
    ["type_magasin","etat_client","heure_transaction"],
    "string"
    )

    print("Modification des types pour les variables à encoder : OK")
    print("="*50)


    # Création des échantillons '2019' et 'test'
    variables_a_retirer = ["target","nom_magasin","ville_client","profession_client","numero_transaction",
                           "annee_transaction","latitude_domicile_client","longitude_domicile_client",
                           "latitude_magasin","longitude_magasin","mois_transaction","jour_transaction"]


    x_2019 = df[df["annee_transaction"]==2019].drop(variables_a_retirer, axis=1)
    y_2019 = df[df["annee_transaction"]==2019]["target"]

    x_test = df[df["annee_transaction"]==2020].drop(variables_a_retirer, axis=1)
    y_test = df[df["annee_transaction"]==2020]["target"]

    print("Création des échantillons '2019' et 'test' : OK")
    print("="*50)


    # Création de l'échantillon 'train' et 'validation'
    x_train, x_validation, y_train, y_validation = train_test_split(
        x_2019, y_2019,
        test_size=0.20,          
        random_state=1,       
        stratify=y_2019                   
    )

    print("Création de l'échantillon 'train' et 'validation' : OK")
    print("="*50)


    # Création de la Pipeline
        # Preprocessing
            # -- Encodage des variables catégorielles 
    var_cat_pipe = Pipeline([
        ("one_hot_encoder", OneHotEncoder(drop=None, sparse_output=False, handle_unknown='ignore'))
    ])

            # -- Isolation Forest sur les données et création de la variable "if_score"
    all_var_pipe = Pipeline([
        ("isolation_forest", IsolationForestCustom(
            n_estimators=100,        
            max_samples="auto",      
            contamination="auto",    
            max_features=1.0,        
            bootstrap=False,         
            n_jobs=-1,               
            random_state=1,         
            verbose=0,               
            warm_start=False         
        ))
    ])

            # -- Standardisation des données 
    num_pipe = Pipeline([
        ("scaler", StandardScaler())
    ])


    print("Fonctions Preprocessing : OK")
    print("="*50)



        # XGBoost (Objectif : Stabilité)
            # -- Modèle de base
    xgb = XGBClassifier(
        scale_pos_weight=200, # 0.995/0.005
        random_state=1,
        n_jobs=-1
    )

            # -- Préparation des hyperparamètres à tuner
    hyperparametres = {
        "model__n_estimators": randint(50, 200),
        "model__max_depth": randint(3, 6),
    
            # --- Apprentissage lent (0.01 - 0.05)
        "model__learning_rate": loguniform(0.01, 0.05),
    
            # --- Bagging et Feature sampling forts
        "model__subsample": uniform(0.5, 0.3),        
        "model__colsample_bytree": uniform(0.5, 0.3), 
    
            # --- Régularisation forte sur les feuilles (min_child_weight)
        "model__min_child_weight": randint(5, 10),
    
            # --- Réduction de perte minimale pour splitter (Gamma)
        "model__gamma": loguniform(0.01, 0.5), 
    
            # --- Régularisation L1 et L2 (Forte)
        "model__reg_alpha": loguniform(0.1, 1.0),
        "model__reg_lambda": loguniform(1.0, 10.0)
    }


    print("Paramétrage du modèle : OK")
    print("="*50)


        # Assemblage 
    preprocessing_1 = ColumnTransformer([
        ("one_hot_encoding", var_cat_pipe, make_column_selector(dtype_exclude=np.number))
    ], remainder="passthrough")

    preprocessing_2 = ColumnTransformer([
        ("scaler", num_pipe, make_column_selector(dtype_include=np.number))
    ], remainder="passthrough")

    pipeline_complete = Pipeline([
        ("ohe", preprocessing_1),
        ("isolation_forest", all_var_pipe),
        ("scaler", preprocessing_2),
        ("model", xgb)
    ])


    print("Assemblage complet (Preprocessing + XGB) : OK")
    print("="*50)


        # Tuning des hyperparamètres
    recherche = RandomizedSearchCV(
        estimator = pipeline_complete,
        param_distributions = hyperparametres,
        n_iter = 6,
        scoring = "neg_brier_score",
        cv = 3,  
        return_train_score = True,
        refit = True,
        random_state = 1,
        n_jobs = -1,
        error_score = "raise",
    )

    _ = recherche.fit(x_train, y_train)
    modele = recherche.best_estimator_


    print("Tuning et recherche des hyperparamètres : OK")
    print("="*50)


        # Calibration des probabilités
    model_propre = CalibratedClassifierCV(
        estimator = modele,  
        method = 'isotonic',       
        cv = 5                    
    )

    _ = model_propre.fit(x_train, y_train)


    print("Calibration des probabilités : OK")
    print("="*50)


    # Recherche du seuil minimisant les couts sur l'échantillon 'validation'
    y_validation_proba = model_propre.predict_proba(x_validation)[:,1]

    cost_optimizer = ThresholdCostOptimizer(cost_fp=25, cost_fn=125, n_thresholds=200)

    cost_optimizer.fit(y_validation, y_validation_proba)

    seuil_optimal = cost_optimizer.get_best_threshold()


    print("Recherche du seuil minimisant les couts sur l'échantillon 'validation' : OK")
    print("="*50)



    # Entrainement du modèle sur l'échantillon 'train' et 'test'
        # --- Train
    train_proba = model_propre.predict_proba(x_train)[:,1]
    train_pred = (train_proba >= seuil_optimal).astype(int)

        # --- Test
    test_proba = model_propre.predict_proba(x_test)[:,1]
    test_pred = (test_proba >= seuil_optimal).astype(int)


    print("Entrainement du modèle sur l'échantillon 'train' et 'test' : OK")
    print("="*50)


    # Sauvegarde des modèles et de x_test pour SHAP
        # --- XGBoost calibré
    joblib.dump(model_propre, ROOT / "04_model" / "fraud_detection_model_xgboost.joblib") 

        # --- XGBoost non calibré
    joblib.dump(modele, ROOT / "04_model" / "fraud_detection_xgb_pas_calibre.joblib")

        # --- x_test
    x_test.to_parquet(PROCESSED_DATA / "x_test_for_shap.parquet")

    print("Sauvegarde des modèles + x_test : OK")
    print("="*50)
