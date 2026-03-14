import plotly.graph_objects as go

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
            marker=dict(color="red", size=10),
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
        annotation_font=dict(color="red", size=14),
    )

    fig.update_layout(
        xaxis_title="Seuil de décision",
        yaxis_title="Coût total (FP/FN) en €",
        title=dict(
            text="Coût total en fonction du seuil de décision",
            x=0.5,
            xanchor="center",
            y=0.97,
            yanchor="top",
            font=dict(size=16),
        ),
        hovermode="x unified",
        legend=dict(
            orientation="v",
            yanchor="top",
            y=0.98,
            xanchor="left",
            x=1.02,
            font=dict(size=14),
        ),
        margin=dict(l=60, r=120, t=80, b=70),
        height=520,
    )

    return fig
