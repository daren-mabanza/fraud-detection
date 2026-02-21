import numpy as np
import pandas as pd


def stratified_sample(df, strat_cols, sample_size, random_state=123):
    """
    Tirage aléatoire stratifié de `sample_size` lignes à partir de `df`,
    en respectant la distribution des strates définies par `strat_cols`.

    Parameters
    ----------
    df : pd.DataFrame
        Table de départ.
    strat_cols : list of str
        Noms des variables de stratification.
    sample_size : int
        Taille totale souhaitée de l'échantillon.
    random_state : int, default 123
        Graine de reproductibilité.

    Returns
    -------
    pd.DataFrame
        Échantillon stratifié de taille `sample_size`.
    """

    # 1) Taille de chaque strate
    strata_sizes = (
        df
        .groupby(strat_cols)
        .size()
        .rename("n")
        .reset_index()
    )

    # 2) Poids + allocation proportionnelle
    strata_sizes["weight"] = strata_sizes["n"] / strata_sizes["n"].sum()
    strata_sizes["n_sample"] = np.floor(
        strata_sizes["weight"] * sample_size
    ).astype(int)

    # 3) Ajustement pour avoir exactement sample_size
    delta = sample_size - strata_sizes["n_sample"].sum()
    if delta > 0:
        # On ajoute 1 aux strates les plus grosses (ou au hasard, comme tu veux)
        strata_sizes.loc[
            strata_sizes["n"].nlargest(delta).index,
            "n_sample"
        ] += 1

    # 4) Tirage stratifié
    strat_sample = (
        df
        .merge(
            strata_sizes[strat_cols + ["n_sample"]],
            on=strat_cols,
            how="left"
        )
        .groupby(strat_cols, group_keys=False)
        .apply(lambda g: g.sample(
            n=int(g["n_sample"].iloc[0]),
            random_state=random_state
        ))
    )

    # Par sécurité, on peut tronquer si jamais un léger dépassement arrive
    return strat_sample.iloc[:sample_size].copy()
