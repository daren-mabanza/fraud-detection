# =============================================================
from pathlib import Path
import pandas as pd
import great_expectations as gx
from fonctions_perso.geo_localisation import calcul_distances
from fonctions_perso.data_quality import initialisation_gx, afficher_resultats_validation

# =============================================================

ROOT = Path.cwd().parents[0]

RAW_DATA = ROOT / "01_data" / "01_raw"
PROCESSED_DATA = ROOT / "01_data" / "02_processed"

# =============================================================

def data_fraud_cleaning():

    # Import des données et modification des types
    data_fraud_1 = (pd.read_csv(RAW_DATA / "fraudTest.csv")
    .astype({
        "cc_num":"object",
        "zip":"object",
        "trans_date_trans_time":"datetime64[ns]",
        "dob":"datetime64[ns]"
    })
    .drop(["Unnamed: 0"], axis = 1))

    data_fraud_2 = (pd.read_csv(RAW_DATA / "fraudTrain.csv")
    .astype({
        "cc_num":"object",
        "zip":"object",
        "trans_date_trans_time":"datetime64[ns]",
        "dob":"datetime64[ns]"
    })
    .drop(["Unnamed: 0"], axis = 1))

    data_fraud = pd.concat([data_fraud_1,data_fraud_2])

    df = data_fraud.copy()

    print("Import des données et modification des types : OK")
    print("="*50)

    # Renommage des variables
    df = df.rename(columns={
        "trans_date_trans_time":"date_heure_transaction","cc_num":"numero_carte","merchant":"nom_magasin",
        "category":"type_magasin","amt":"montant_transaction","first":"prenom","last":"nom","gender":"sexe",
        "street":"adresse_client","city":"ville_client","state":"etat_client","zip":"code_postal_client",
        "lat":"latitude_domicile_client","long":"longitude_domicile_client","trans_num":"numero_transaction",
        "city_pop":"population_ville_client","job":"profession_client","dob":"date_naissance_client",
        "unix_time":"timestamp_unix_transacation","merch_lat":"latitude_magasin","merch_long":"longitude_magasin",
        "is_fraud":"target"
        })
    
    print("Renommage des variables : OK")
    print("="*50)

    # Nettoyage de la variable "nom_magasin"
    df["nom_magasin"] = df["nom_magasin"].str.slice_replace(0,6,"")

    print("Nettoyage de la variable 'nom_magasin' : OK")
    print("="*50)


    # Création de nouvelles variables 

        # Année de la transaction
    df["annee_transaction"] = df["date_heure_transaction"].dt.year

        # Mois de la transaction
    df["mois_transaction"] = df["date_heure_transaction"].dt.month_name()

        # Jour de la transaction
    df["jour_transaction"] = df["date_heure_transaction"].dt.day_name()

        # Heure de la transaction
    df["heure_transaction"] = df["date_heure_transaction"].dt.hour

        # Distance "domicile" - "magasin"
    calcul_distances(
        df,
        "latitude_domicile_client","longitude_domicile_client",
        "latitude_magasin","longitude_magasin",
        unit="km",
        out_col="distance_domicile_magasin"
    )

        # Age client
    df["age_client"] = df["date_heure_transaction"].dt.year-df["date_naissance_client"].dt.year


    print("Création des nouvelles variables : OK")
    print("="*50)

    # Suppressions des variables non utilisables
    variables_a_supprimer = ["numero_carte","prenom","nom",
                             "sexe","adresse_client","code_postal_client",
                             "timestamp_unix_transacation","date_heure_transaction",
                             "date_naissance_client"]

    df = df.drop(variables_a_supprimer, axis=1)

    print("Suppressions des variables non utilisables : OK")
    print("="*50)

    # Controle qualité des données avec Great Expectations
    context = gx.get_context()

         # Initialisation du batch
    batch_request = initialisation_gx(
        context=context,
        dataframe=df,
        ds_name="data_fraud",
        asset_name="cleaning_engineering_selection"
    )

    validator = context.get_validator(batch_request=batch_request)

        # Paramétrage des conditions
    validator.expect_column_to_exist('target')
    validator.expect_column_values_to_not_be_null('target')
    validator.expect_table_column_count_to_equal(19) 

        # Export des données sous condition
    resultats = validator.validate()

    afficher_resultats_validation(resultats)

    if resultats.success:
        print("Les conditions sont validés, export possible")
        df.to_parquet(PROCESSED_DATA / "data_fraud_processed.parquet")
        print("Export des données : OK")
        print("="*50)
    else:
        print("Les conditions de sont pas validées !")

    return df
    

    


    




    
     

