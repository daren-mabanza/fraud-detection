from cleaning.feature_engineering_and_selection import data_fraud_cleaning
from cleaning.post_eda import cleaning_post_eda
from machine_learning_pipeline.xgb_fraud import xgboost_model


def full_fraud_detection_projet():

    # Nettoyage des données
    data_fraud_cleaning()

    print("FEATURE ENGINEERING/SELECTION ET PREMIER NETTOYAGE DES DONNEES : OK")
    print("="*70)

    # Nettoygage des données post EDA
    cleaning_post_eda()

    print("NETTOYAGE DES DONNEES POST EDA : OK")
    print("="*70)


    # Pipeline ML XGBoost
    xgboost_model()

    print("PIPELINE ML XGBOOST : OK")
    print("="*70)
