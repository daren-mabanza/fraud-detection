# ================================
# Import des packages nécéssaires
# ================================

from config import ROOT
import plotly.graph_objects as go
import numpy as np

# =======================
# Création des fonctions
# =======================


def plot_local_shap_vertical(
    shap_values,
    feature_names,
    top_k: int = 10,
    title: str = "Contributions locales des variables"
):
    """
    Fonction permettant d'obtenir un Waterfall Plot "SHAP Look Like".

    shap_values : array 1D (n_features,) pour 1 observation
    feature_names : liste de noms (n_features)
    top_k : nb de variables à afficher (par |valeur SHAP|)
    """

    shap_arr = np.asarray(shap_values).astype(float).flatten()
    feature_names = list(feature_names)

    # 1. Top k par importance absolue
    idx_sorted = np.argsort(-np.abs(shap_arr))
    idx_top = idx_sorted[:top_k]

    vals = shap_arr[idx_top]
    names = [feature_names[i] for i in idx_top]

    # 2. Couleur selon signe
    colors = ["#e74c3c" if v > 0 else "#3498db" for v in vals]

    fig = go.Figure()

    # 3. Barres horizontales (variables sur Y, valeurs sur X)
    fig.add_trace(
        go.Bar(
            x=vals,
            y=names,
            orientation="h",
            marker_color=colors,
            text=[f"{v:+.2f}" for v in vals],
            textposition="outside",
            textfont=dict(size=16),  # texte au bout des barres
            hovertemplate="<b>%{y}</b><br>Contribution: %{x:.3f}<extra></extra>",
        )
    )

    # 4. Mise en forme
    fig.update_layout(
        title=dict(text=title, font=dict(size=23)),
        xaxis=dict(
            title=dict(text="Contribution au score du modèle",
                       font=dict(size=16)),
            tickfont=dict(size=14),
        ),
        yaxis=dict(
            title=dict(text="Variables", font=dict(size=18)),
            tickfont=dict(size=14),
        ),
        template="plotly_white",
        showlegend=False,
        bargap=0.3,
        margin=dict(l=220, r=60, t=70, b=50),
        hoverlabel=dict(font_size=18),
    )

    # ligne verticale à 0 pour repère
    fig.add_vline(x=0, line_color="grey", line_width=1, line_dash="dash")

    return fig

