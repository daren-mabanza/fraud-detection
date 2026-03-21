# =========================================================================
# Import des packages nécéssaires 
# =========================================================================
from pathlib import Path

import pandas as pd
import numpy as np
from fonctions_perso.data_manipulation import multi_astype
from fonctions_perso.data_manipulation import stratified_sample

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.compose import make_column_selector

from xgboost import XGBClassifier

from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit
from scipy.stats import randint, uniform, loguniform
from sklearn.calibration import CalibratedClassifierCV

from fonctions_perso.machine_learning import IsolationForestCustom
from sklearn.preprocessing import OneHotEncoder

from fonctions_perso.machine_learning import ThresholdCostOptimizer

import joblib
import shap

from sklearn import set_config
set_config(transform_output="pandas")



# ==========================================================================
# Paramétrage de l'environnement de travail
# ==========================================================================
ROOT = Path.cwd().parents[0]

RAW_DATA = ROOT / "01_data" / "01_raw"
PROCESSED_DATA = ROOT / "01_data" / "02_processed"
JOBLIB_DATA = ROOT / "01_data" / "03_joblib"
MODEL_DATA = ROOT / "04_model"

# ==========================================================================
# Pipeline complète 
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


    # Création des échantillons train | validation | test
    variables_a_retirer = ["target","nom_magasin","ville_client","profession_client","numero_transaction",
                       "annee_transaction","latitude_domicile_client","longitude_domicile_client",
                       "latitude_magasin","longitude_magasin","mois_transaction","jour_transaction",
                       "population_ville_client","distance_domicile_magasin","date_heure_transaction"]

    periode_validation = ["January","February","March","April","May","June","July","August","September"]

        # Train trié par ordre de transaction pour préparer la validation croisée temporelle (TimeSeriesSplit)
    train_2019 = df[df["annee_transaction"]==2019].sort_values(by="date_heure_transaction").reset_index(drop=True)

    x_train = train_2019.drop(variables_a_retirer, axis = 1)
    y_train = train_2019["target"]


        # Test & Validation
    data_fraud_2020_validation = (df[(df["annee_transaction"]==2020) & (df["mois_transaction"].isin(periode_validation))]
                                .sort_values(by="date_heure_transaction")
                                .reset_index(drop=True))

    data_fraud_2020_test = (df[(df["annee_transaction"]==2020) & (~df["mois_transaction"].isin(periode_validation))]
                            .sort_values(by="date_heure_transaction")
                            .reset_index(drop=True))


            # --- Validation
    x_validation = data_fraud_2020_validation.drop(variables_a_retirer, axis=1)
    y_validation = data_fraud_2020_validation["target"]

            # --- Test
    x_test = data_fraud_2020_test.drop(variables_a_retirer, axis=1)
    y_test = data_fraud_2020_test["target"]


    print("Création des échantillons train | validation | test : OK")
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


    print("Fonctions Preprocessing : OK")
    print("="*50)



        # XGBoost (Objectif : Stabilité)
            # -- Modèle de base
    xgb = XGBClassifier(
        scale_pos_weight=200, # 0.995/0.005 (Taux de tranaction frauduleuses en train)
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

    pipeline_complete = Pipeline([
        ("ohe", preprocessing_1),
        ("isolation_forest", all_var_pipe),
        ("model", xgb)
    ])


    print("Assemblage complet (Preprocessing + XGB) : OK")
    print("="*50)

        # Tuning des hyperparamètres
    tscv = TimeSeriesSplit(n_splits=4)    
        
    recherche = RandomizedSearchCV(
        estimator = pipeline_complete,
        param_distributions = hyperparametres,
        n_iter = 6,
        scoring = "neg_brier_score",
        cv = tscv,  
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
    modele_propre = CalibratedClassifierCV(
        estimator = modele,  
        method = 'isotonic',       
        cv = 5                    
    )

    _ = modele_propre.fit(x_train, y_train)


    print("Calibration des probabilités : OK")
    print("="*50)


    # Recherche du seuil minimisant les couts sur l'échantillon 'validation'
    y_validation_proba = modele_propre.predict_proba(x_validation)[:,1]
    df_y_validation_proba = pd.DataFrame(y_validation_proba)
    df_y_validation_proba = df_y_validation_proba.rename(columns={0:"proba"})


    cost_optimizer = ThresholdCostOptimizer(cost_fp=25, cost_fn=125, n_thresholds=200)
    
    cost_optimizer_artifacts = {
        "seuils": cost_optimizer.seuils_,
        "costs": cost_optimizer.costs_,
        "best_threshold": cost_optimizer.get_best_threshold(),
        "best_cost": cost_optimizer.get_best_cost()
    }

    cost_optimizer.fit(y_validation, y_validation_proba)


    print("Recherche du seuil minimisant les couts sur l'échantillon 'validation' : OK")
    print("="*50)



    # Entrainement du modèle sur l'échantillon 'train' et 'test'
        # --- Train
    train_proba = modele_propre.predict_proba(x_train)[:,1]

        # --- Test
    test_proba = modele_propre.predict_proba(x_test)[:,1]


    print("Entrainement du modèle sur l'échantillon 'train' et 'test' : OK")
    print("="*50)


    # Sauvegarde des objets JOBLIB
    full_test = pd.concat([x_test,y_test], axis = 1)

    full_test_sample = stratified_sample(
        df=full_test,
        target="target",
        n=35000,
        random_state=123
    )

    x_test_sample = full_test_sample.drop(["target"], axis = 1)
    
    
        # Modèles 
    joblib.dump(modele, MODEL_DATA / "fraud_detection_xgb_pas_calibre.joblib")  
    joblib.dump(modele_propre, MODEL_DATA / "fraud_detection_model_xgboost.joblib") 

        # Train
    joblib.dump(y_train, JOBLIB_DATA / "y_train.joblib")
    joblib.dump(train_proba, JOBLIB_DATA / "train_proba.joblib")

        # Validation
    joblib.dump(data_fraud_2020_validation["target"], JOBLIB_DATA / "y_validation.joblib")
    joblib.dump(df_y_validation_proba, JOBLIB_DATA / "proba_validation.joblib")

        # Test complet
    joblib.dump(y_test, JOBLIB_DATA / "y_test.joblib")
    joblib.dump(test_proba, JOBLIB_DATA / "test_proba.joblib")

        # Objet CostOptimizer contenant le seuil optimal
    joblib.dump(cost_optimizer, JOBLIB_DATA / "cost_optimizer.joblib")
    joblib.dump(cost_optimizer_artifacts, JOBLIB_DATA / "cost_optimizer_artifacts.joblib")

        # Liste des modalités de la variable "etat_client"
    values_etats_client = df["etat_client"].value_counts().reset_index()["etat_client"]
    joblib.dump(list(values_etats_client), JOBLIB_DATA / "values_etat_client.joblib")

        # Echantillons de données pour l'onglet 3 de l'application Streamlit
    fraud_1 = full_test[full_test["target"]==1].sample(5)
    fraud_0 = full_test[full_test["target"]==0].sample(5)

    test_sample_onglet_3 = pd.concat([fraud_0,fraud_1], axis=0).sample(10)
    joblib.dump(test_sample_onglet_3, JOBLIB_DATA / "test_sample_onglet_3.joblib")
    
        # Echantillon de données "x_test_sample"
    joblib.dump(x_test_sample, JOBLIB_DATA / "x_test_sample.joblib")

        # Sauvegarde de l'explainer et des SHAP Values + x_test_processed
        
    x_test_proc = modele[:-1].transform(x_test_sample) # x_test_sample avec preprocessing uniquement (sans le modèle)
    xgb = modele.named_steps["model"]                  # juste le modèle sans les étapes de preprocessing

    explainer = shap.TreeExplainer(xgb, x_test_proc)   # Initialisation de l'explainer
    shap_values = explainer(x_test_proc)               # SHAP Values

        
    joblib.dump(x_test_proc, JOBLIB_DATA / "x_test_processed.joblib")
    joblib.dump(explainer, JOBLIB_DATA / "explainer.joblib")
    joblib.dump(shap_values, JOBLIB_DATA / "shap_values_modele.joblib")

        # Sauvegarde de full_test_sample
    joblib.dump(full_test_sample, JOBLIB_DATA / "full_test_sample.joblib")

    print("Sauvegarde des objets JOBLIB : OK")
    print("="*50)
