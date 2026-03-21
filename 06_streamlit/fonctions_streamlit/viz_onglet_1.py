# ================================
# Import des packages nécéssaires
# ================================

from config import ROOT
import streamlit as st
import numpy as np
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef,
    roc_auc_score,
    brier_score_loss,
    average_precision_score,
    confusion_matrix,
    precision_recall_curve,
    brier_score_loss
)
from sklearn.calibration import calibration_curve

# =======================
# Création des fonctions
# =======================


def afficher_metriques_binaires(
    y_true,
    y_proba,
    seuil_decision: float = 0.5,
    title: str = "Métriques globales (test)",
):
    """
    Calcule et affiche les métriques d'un modèle de classification binaire
    à partir des vraies étiquettes et des probabilités prédites.
    Affichage sous forme de tableau HTML.
    """

    y_true = np.asarray(y_true)
    y_proba = np.asarray(y_proba)
    y_pred = (y_proba >= seuil_decision).astype(int)

    # Calcul des métriques
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    mcc = matthews_corrcoef(y_true, y_pred)
    roc_auc = roc_auc_score(y_true, y_proba)
    brier = brier_score_loss(y_true, y_proba)
    avg_prec = average_precision_score(y_true, y_proba)

    st.subheader(title)
    st.markdown(f"*Seuil de décision utilisé :* {seuil_decision:.3f}")

    html = f"""
<style>
    .metrics-table {{
        border-collapse: collapse;
        margin-top: 0.75rem;
        background-color: #f9fafb;
        color: #111827;
        min-width: 420px;
        margin-left: auto;
        margin-right: auto;
    }}
    .metrics-table th, .metrics-table td {{
        border: 1px solid #e5e7eb;
        padding: 10px 16px;
        text-align: left;
        font-size: 1.0rem;
    }}
    .metrics-header {{
        background-color: #e5e7eb;
        font-weight: 600;
    }}
    .metrics-name {{
        font-weight: 500;
    }}
</style>

<table class="metrics-table">
    <tr class="metrics-header">
        <th>Métrique</th>
        <th>Valeur</th>
    </tr>
    <tr>
        <td class="metrics-name">Accuracy</td>
        <td>{accuracy:.4f}</td>
    </tr>
    <tr>
        <td class="metrics-name">Précision</td>
        <td>{precision:.4f}</td>
    </tr>
    <tr>
        <td class="metrics-name">Rappel</td>
        <td>{recall:.4f}</td>
    </tr>
    <tr>
        <td class="metrics-name">F1-Score</td>
        <td>{f1:.4f}</td>
    </tr>
    <tr>
        <td class="metrics-name">MCC</td>
        <td>{mcc:.4f}</td>
    </tr>
    <tr>
        <td class="metrics-name">ROC-AUC</td>
        <td>{roc_auc:.4f}</td>
    </tr>
    <tr>
        <td class="metrics-name">Brier score</td>
        <td>{brier:.4f}</td>
    </tr>
    <tr>
        <td class="metrics-name">Average precision</td>
        <td>{avg_prec:.4f}</td>
    </tr>
</table>
"""

    st.markdown(html, unsafe_allow_html=True)



def afficher_matrice_confusion(
    y_true,
    y_proba,
    seuil_decision: float = 0.5,
    title: str = "Matrice de confusion (test)",
):
    """
    Affiche une matrice de confusion sous forme de tableau HTML
    propre dans Streamlit, avec st.markdown.
    """

    y_true = np.asarray(y_true)
    y_proba = np.asarray(y_proba)
    y_pred = (y_proba >= seuil_decision).astype(int)

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()

    st.subheader(title)

    html = f"""
<style>
    .cm-table {{
        border-collapse: collapse;
        margin-top: 0.75rem;
        background-color: #f9fafb;
        color: #111827;
        min-width: 420px;
        margin-left: auto;
        margin-right: auto;
    }}
    .cm-table th, .cm-table td {{
        border: 1px solid #e5e7eb;
        padding: 10px 16px;
        text-align: center;
        font-size: 1.0rem;
    }}
    .cm-header {{
        background-color: #e5e7eb;
        font-weight: 600;
    }}
    .cm-row-label {{
        background-color: #e5e7eb;
        font-weight: 500;
    }}
</style>

<table class="cm-table">
    <tr class="cm-header">
        <th></th>
        <th>Prédit Négatif</th>
        <th>Prédit Positif</th>
    </tr>
    <tr>
        <th class="cm-row-label">Réel Négatif</th>
        <td>{tn}</td>
        <td>{fp}</td>
    </tr>
    <tr>
        <th class="cm-row-label">Réel Positif</th>
        <td>{fn}</td>
        <td>{tp}</td>
    </tr>
</table>
"""

    st.markdown(html, unsafe_allow_html=True)





def afficher_courbe_pr_train_test(
    y_train,
    train_proba,
    y_test,
    test_proba,
    figsize=(6, 5),
    title="Courbes Précision–Rappel (train vs test)",
):
    """
    Trace les courbes Précision–Rappel pour train et test,
    avec AP en légende et baseline positive sur le test.
    """

    y_train = np.asarray(y_train)
    train_proba = np.asarray(train_proba)
    y_test = np.asarray(y_test)
    test_proba = np.asarray(test_proba)

    # Courbes PR
    precision_train, recall_train, _ = precision_recall_curve(y_train, train_proba)
    precision_test, recall_test, _ = precision_recall_curve(y_test, test_proba)

    # Average precision
    ap_train = average_precision_score(y_train, train_proba)
    ap_test = average_precision_score(y_test, test_proba)

    # Baseline (prévalence) sur le test
    baseline_test = y_test.mean()

    fig, ax = plt.subplots(figsize=figsize)

    ax.plot(
        recall_train,
        precision_train,
        color="blue",
        lw=2,
        label=f"Train (AP = {ap_train:.3f})",
    )
    ax.plot(
        recall_test,
        precision_test,
        color="red",
        lw=2,
        linestyle="--",
        label=f"Test (AP = {ap_test:.3f})",
    )
    ax.axhline(
        y=baseline_test,
        color="gray",
        linestyle=":",
        lw=1,
        label=f"Baseline test = {baseline_test:.3f}",
    )

    ax.set_xlabel("Rappel (Recall)")
    ax.set_ylabel("Précision (Precision)")
    ax.set_title(title)
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.05)
    ax.grid(alpha=0.3, linestyle=":", linewidth=0.5)
    ax.legend(loc="best", fontsize=9)

    plt.tight_layout()
    return fig



def afficher_courbe_calibration_train_test(
    y_train,
    train_proba,
    y_test,
    test_proba,
    n_bins: int = 10,
    figsize=(6, 5),
    title="Courbes de calibration (train vs test)",
):
    """
    Trace les courbes de calibration pour train et test
    + Brier scores en légende.
    """

    y_train = np.asarray(y_train)
    train_proba = np.asarray(train_proba)
    y_test = np.asarray(y_test)
    test_proba = np.asarray(test_proba)

    # Brier scores
    brier_train = brier_score_loss(y_train, train_proba)
    brier_test = brier_score_loss(y_test, test_proba)

    # Courbes de calibration
    prob_true_train, prob_pred_train = calibration_curve(
        y_train, train_proba, n_bins=n_bins
    )
    prob_true_test, prob_pred_test = calibration_curve(
        y_test, test_proba, n_bins=n_bins
    )

    fig, ax = plt.subplots(figsize=figsize)

    ax.plot(
        prob_pred_train,
        prob_true_train,
        marker="o",
        color="blue",
        lw=2,
        label=f"Train (Brier = {brier_train:.3f})",
    )
    ax.plot(
        prob_pred_test,
        prob_true_test,
        marker="s",
        color="red",
        lw=2,
        linestyle="--",
        label=f"Test (Brier = {brier_test:.3f})",
    )
    ax.plot(
        [0, 1],
        [0, 1],
        linestyle=":",
        color="gray",
        lw=1.5,
        label="Calibration parfaite",
    )

    ax.set_xlabel("Probabilité prédite")
    ax.set_ylabel("Probabilité observée")
    ax.set_title(title)
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.grid(alpha=0.3, linestyle=":", linewidth=0.5)
    ax.legend(loc="best", fontsize=9)

    plt.tight_layout()
    return fig