"""
model_utils.py
================
Fonctions d'entraînement, d'évaluation et de calcul des métriques.

KPIs cibles (Section 5 thèse) :
  - Sensibilité >= 85% (PRIORITAIRE, raison médicale)
  - Spécificité >= 75%
  - AUC >= 0.85
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.metrics import (
    confusion_matrix, roc_auc_score, accuracy_score,
    precision_score, recall_score, f1_score, roc_curve
)


def get_model_zoo(random_state=42) -> dict:
    """
    Retourne les 5 modèles à comparer, avec grilles d'hyperparamètres pour GridSearchCV.
    class_weight='balanced' partout : le dataset France est déséquilibré (11.3% positifs).
    """
    return {
        'RandomForest': {
            'estimator': RandomForestClassifier(random_state=random_state, class_weight='balanced'),
            'param_grid': {
                'n_estimators': [200, 400],
                'max_depth': [5, 10, None],
                'min_samples_leaf': [1, 5, 10],
            }
        },
        'GradientBoosting': {
            'estimator': GradientBoostingClassifier(random_state=random_state),
            'param_grid': {
                'n_estimators': [200, 400],
                'max_depth': [3, 5],
                'learning_rate': [0.05, 0.1],
            }
        },
        'XGBoost': {
            'estimator': XGBClassifier(random_state=random_state, eval_metric='logloss',
                                         use_label_encoder=False),
            'param_grid': {
                'n_estimators': [200, 400],
                'max_depth': [3, 5, 7],
                'learning_rate': [0.05, 0.1],
                'scale_pos_weight': [1, 7.9],  # ratio 88.7/11.3 pour déséquilibre
            }
        },
        'LogisticRegression': {
            'estimator': LogisticRegression(random_state=random_state, class_weight='balanced',
                                              max_iter=1000),
            'param_grid': {
                'C': [0.01, 0.1, 1, 10],
                'penalty': ['l2'],
            }
        },
        'SVM': {
            'estimator': SVC(random_state=random_state, class_weight='balanced', probability=True),
            'param_grid': {
                'C': [0.1, 1, 10],
                'kernel': ['rbf', 'linear'],
            }
        },
    }


def train_with_gridsearch(estimator, param_grid, X_train, y_train, cv_folds=5, scoring='f1_weighted', sample_weight=None):
    """Entraîne un modèle avec GridSearchCV et validation croisée stratifiée.
    sample_weight : à utiliser UNIQUEMENT pour les modèles sans class_weight natif
    (ex: GradientBoostingClassifier n'a pas de paramètre class_weight)."""
    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
    grid = GridSearchCV(estimator, param_grid, cv=cv, scoring=scoring, n_jobs=-1, verbose=1)
    if sample_weight is not None:
        grid.fit(X_train, y_train, sample_weight=sample_weight)
    else:
        grid.fit(X_train, y_train)
    return grid.best_estimator_, grid.best_params_, grid.best_score_


def compute_clinical_metrics(y_true, y_pred, y_proba) -> dict:
    """
    Calcule toutes les métriques cliniques nécessaires pour Section 7.
    y_true, y_pred : 0/1 (0=Non, 1=Oui)
    y_proba : probabilité prédite classe positive
    """
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

    sensibilite = tp / (tp + fn) if (tp + fn) > 0 else np.nan  # = Recall
    specificite = tn / (tn + fp) if (tn + fp) > 0 else np.nan
    ppv = tp / (tp + fp) if (tp + fp) > 0 else np.nan  # Precision
    npv = tn / (tn + fn) if (tn + fn) > 0 else np.nan

    return {
        'n_total': len(y_true),
        'n_positifs': int((y_true == 1).sum()),
        'n_negatifs': int((y_true == 0).sum()),
        'TP': int(tp), 'TN': int(tn), 'FP': int(fp), 'FN': int(fn),
        'sensibilite': round(sensibilite, 4),
        'specificite': round(specificite, 4),
        'accuracy': round(accuracy_score(y_true, y_pred), 4),
        'precision_PPV': round(ppv, 4),
        'npv': round(npv, 4),
        'f1_score': round(f1_score(y_true, y_pred), 4),
        'auc_roc': round(roc_auc_score(y_true, y_proba), 4),
        'kpi_sensibilite_85_atteint': bool(sensibilite >= 0.85),
        'kpi_specificite_75_atteint': bool(specificite >= 0.75),
        'kpi_auc_085_atteint': bool(roc_auc_score(y_true, y_proba) >= 0.85),
    }


def find_threshold_for_target_sensitivity(y_true, y_proba, target_sensitivity=0.85):
    """
    Trouve le seuil de décision qui atteint la sensibilité cible.
    Utile car le seuil par défaut (0.5) ne garantit PAS sensibilité >= 85%.
    C'est une décision clinique : on accepte plus de faux positifs pour
    ne rater aucun cas DG.
    """
    fpr, tpr, thresholds = roc_curve(y_true, y_proba)
    valid_idx = np.where(tpr >= target_sensitivity)[0]
    if len(valid_idx) == 0:
        return None, None
    # Prendre le seuil le plus élevé qui atteint encore la sensibilité cible
    # (minimise les faux positifs tout en respectant la contrainte)
    best_idx = valid_idx[np.argmin(fpr[valid_idx])]
    return thresholds[best_idx], tpr[best_idx]


def threshold_sweep(y_true, y_proba, thresholds=None):
    """
    Balaie une plage de seuils de décision et calcule toutes les métriques
    cliniques pour chacun. Permet de visualiser le VRAI compromis
    sensibilité/spécificité plutôt qu'un seul point de recherche ciblée.

    Ajoute l'indice de Youden (J = sensibilité + spécificité - 1), une
    référence standard en biostatistique pour identifier le seuil qui
    équilibre au mieux les deux erreurs (sans favoriser l'une sur l'autre) —
    utile pour comparer objectivement au seuil retenu pour raisons cliniques
    (priorité sensibilité, cf. Section 5).
    """
    if thresholds is None:
        thresholds = np.arange(0.05, 0.96, 0.05)
    rows = []
    for t in thresholds:
        y_pred = (y_proba >= t).astype(int)
        m = compute_clinical_metrics(y_true, y_pred, y_proba)
        m['threshold'] = round(float(t), 3)
        m['youden_j'] = round(m['sensibilite'] + m['specificite'] - 1, 4)
        rows.append(m)
    return pd.DataFrame(rows)


def calibrate_subgroup_thresholds(y_true, y_proba, group_labels, target_sensitivity=0.85, min_n=20):
    """
    ⚠️ ANALYSE EXPLORATOIRE — PAS une validation externe pure (Section 5.2.4).

    Calibre un seuil de décision SPÉCIFIQUE à chaque sous-groupe pour que
    CHAQUE sous-groupe atteigne individuellement la sensibilité cible, au lieu
    d'un seuil global unique mal calibré aux extrêmes (cf. notebook 07,
    collapse de discrimination sur IMC/âge).

    ATTENTION MÉTHODOLOGIQUE : cette calibration utilise les données de
    validation Cameroun elles-mêmes pour ajuster la fonction de décision —
    c'est une forme d'adaptation post-hoc, pas une validation externe stricte.
    À présenter comme piste exploratoire, jamais comme remplacement des
    résultats de validation officiels (seuil global, notebook 05).

    min_n : sous-groupes trop petits exclus (seuil calibré sur <20 échantillons
    = statistiquement fragile, non fiable).
    """
    y_true = np.asarray(y_true)
    y_proba = np.asarray(y_proba)
    group_labels = pd.Series(group_labels).reset_index(drop=True)

    results = []
    thresholds_by_group = {}

    for g in group_labels.dropna().unique():
        mask = (group_labels == g).values
        n = int(mask.sum())
        if n < min_n:
            continue
        t, achieved_sens = find_threshold_for_target_sensitivity(
            y_true[mask], y_proba[mask], target_sensitivity)
        if t is None:
            # Cible inatteignable même au seuil minimal du balayage ROC -> repli prudent
            t = 0.0
        thresholds_by_group[str(g)] = round(float(t), 4)
        y_pred_g = (y_proba[mask] >= t).astype(int)
        m = compute_clinical_metrics(y_true[mask], y_pred_g, y_proba[mask])
        m['groupe'] = g
        m['n'] = n
        m['seuil_calibre'] = round(float(t), 4)
        m['fiabilite_statistique'] = 'Fragile (n<50)' if n < 50 else 'Correcte'
        results.append(m)

    return pd.DataFrame(results), thresholds_by_group
