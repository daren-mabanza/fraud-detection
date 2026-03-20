# ================================
# Import des packages nécéssaires
# ================================

import plotly.graph_objects as go

# =======================
# Création des fonctions
# =======================

def plot_cost_threshold_curve(seuils, costs, best_t, best_cost):
    """
    Construit la figure Plotly du coût total en fonction du seuil,
    avec point et trait vertical au seuil optimal.
    """
    fig = go.Figure()

    # Courbe de coût total
    fig.add_trace(
        go.Scatter(
            x=seuils,
            y=costs,
            mode="lines",
            name="Coût total",
            line=dict(color="#1f77b4", width=3),
            hovertemplate=(
                "Seuil: %{x:.4f}<br>"
                "Coût total: %{y:,.0f} €<extra></extra>"
            ),
        )
    )

    # Point seuil optimal
    fig.add_trace(
        go.Scatter(
            x=[best_t],
            y=[best_cost],
            mode="markers",
            name=f"Seuil optimal ({best_t:.4f})",
            marker=dict(color="red", size=13),
            hovertemplate=(
                "Seuil optimal: %{x:.4f}<br>"
                "Coût minimum: %{y:,.0f} €<extra></extra>"
            ),
        )
    )

    # Trait vertical rouge au seuil optimal
    fig.add_vline(
        x=best_t,
        line_width=2,
        line_dash="dash",
        line_color="red",
        annotation_text=f"Seuil optimal = {best_t:.3f}",
        annotation_position="top right",
        annotation_font=dict(color="red", size=16),
    )

    fig.update_layout(
        xaxis=dict(
            title=dict(text="Seuil de décision", font=dict(size=18)),
            tickfont=dict(size=14),
        ),
        yaxis=dict(
            title=dict(text="Coût total (FP/FN) en €", font=dict(size=18)),
            tickfont=dict(size=14),
        ),
        title=dict(
            text="Coût total en fonction du seuil de décision",
            x=0.5,
            xanchor="center",
            y=0.97,
            yanchor="top",
            font=dict(size=20),
        ),
        hovermode="x unified",
        hoverlabel=dict(font_size=18),
        legend=dict(
            orientation="v",
            yanchor="top",
            y=0.98,
            xanchor="left",
            x=1.02,
            font=dict(size=14),
        ),
        margin=dict(l=70, r=140, t=90, b=70),
        height=590,
    )

    return fig







