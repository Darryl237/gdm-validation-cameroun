"""
visualization.py
==================
Fonctions de visualisation standardisées (palette cohérente pour la thèse).
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import roc_curve, confusion_matrix

# Palette cohérente avec les dashboards HTML déjà créés (Section 6)
PALETTE = {
    'primary': '#667eea',
    'secondary': '#764ba2',
    'success': '#43e97b',
    'warning': '#f5576c',
    'critical': '#dc3545',
}

sns.set_theme(style='whitegrid', palette='muted')
plt.rcParams['figure.dpi'] = 120
plt.rcParams['font.size'] = 11


def plot_missing_data_heatmap(df, title="Carte des données manquantes", save_path=None):
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.heatmap(df.isna(), cbar=False, cmap=['#e9ecef', PALETTE['warning']], ax=ax)
    ax.set_title(title, fontsize=14, fontweight='bold')
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
    return fig


def plot_roc_curve(y_true, y_proba, model_name="Modèle", save_path=None):
    fpr, tpr, _ = roc_curve(y_true, y_proba)
    from sklearn.metrics import auc
    roc_auc = auc(fpr, tpr)

    fig, ax = plt.subplots(figsize=(7, 7))
    ax.plot(fpr, tpr, color=PALETTE['primary'], lw=2.5, label=f'{model_name} (AUC={roc_auc:.3f})')
    ax.plot([0, 1], [0, 1], color='gray', lw=1, linestyle='--', label='Hasard (AUC=0.5)')
    ax.axhline(y=0.85, color=PALETTE['critical'], lw=1, linestyle=':', label='Seuil sensibilité KPI (85%)')
    ax.set_xlabel('Taux de Faux Positifs (1 - Spécificité)')
    ax.set_ylabel('Taux de Vrais Positifs (Sensibilité)')
    ax.set_title(f'Courbe ROC — Validation Externe Cameroun', fontweight='bold')
    ax.legend(loc='lower right')
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
    return fig


def plot_confusion_matrix_annotated(y_true, y_pred, save_path=None):
    cm = confusion_matrix(y_true, y_pred)
    labels = np.array([
        [f'TN\n{cm[0,0]}', f'FP\n{cm[0,1]}'],
        [f'FN\n{cm[1,0]}', f'TP\n{cm[1,1]}']
    ])
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(cm, annot=labels, fmt='', cmap='Blues', cbar=False,
                xticklabels=['Prédit Non', 'Prédit Oui'],
                yticklabels=['Réel Non', 'Réel Oui'], ax=ax)
    ax.set_title('Matrice de Confusion — Validation Cameroun', fontweight='bold')
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
    return fig


def plot_metrics_by_group(results_df, group_col, metric_col='auc_roc', save_path=None):
    """Pour l'analyse d'équité (notebook 07) — barplot métrique par sous-groupe démographique."""
    fig, ax = plt.subplots(figsize=(9, 5))
    colors = [PALETTE['primary'] if v >= results_df[metric_col].mean() - 0.05
              else PALETTE['critical'] for v in results_df[metric_col]]
    ax.bar(results_df[group_col].astype(str), results_df[metric_col], color=colors)
    ax.axhline(y=results_df[metric_col].mean(), color='gray', linestyle='--', label='Moyenne globale')
    ax.set_ylabel(metric_col.upper())
    ax.set_title(f'{metric_col.upper()} par {group_col} — Analyse d\'équité', fontweight='bold')
    ax.legend()
    plt.xticks(rotation=30, ha='right')
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
    return fig
