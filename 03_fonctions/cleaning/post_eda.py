# ======================
from pathlib import Path
import pandas as pd
import numpy as np
# ======================

ROOT = Path.cwd().parents[0]

RAW_DATA = ROOT / "01_data" / "01_raw"
PROCESSED_DATA = ROOT / "01_data" / "02_processed"

# ======================

def cleaning_post_eda():

    # Import des données
    data_fraud = pd.read_parquet(PROCESSED_DATA / "data_fraud_processed.parquet")
    df = data_fraud.copy()

    print("Import des données : OK")
    print("="*50)


    # Nettoyage de la variable 'etat_client'
    df["etat_client"] = np.where(
        df["etat_client"].isin(["DC","HI","AK","RI"]),
        "other",
        df["etat_client"]
    )

    print("Nettoyage de la variable 'etat_client' : OK")
    print("="*50)


    # Export de la table modifiée
    df.to_parquet(PROCESSED_DATA / "data_fraud_eda_processed.parquet")


    print("Export de la table modifiée : OK")
    print("="*50)



