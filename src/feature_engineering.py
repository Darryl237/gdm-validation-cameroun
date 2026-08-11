"""
feature_engineering.py
========================
Construction du pipeline de preprocessing.

RÈGLE D'OR (rappel) : le pipeline est TOUJOURS fit() sur France uniquement,
puis appliqué en transform() seul sur Cameroun. Ne jamais fit sur Cameroun.
"""

import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.experimental import enable_iterative_imputer  # ✅ DOIT être en premier
from sklearn.impute import IterativeImputer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler


# ============================================================
# # MODÈLE A — 12 variables clinique/biologique (PRINCIPAL)
# ============================================================
MODEL_A_NUMERIC = [
    'age_maternel',
    'imc_ordinal',
    'tension_ordinal',
    'sa_premiere_consult',
]

MODEL_A_CATEGORICAL = [
    'parite',
    'atcd_familial_diabete_1er_deg',
    'atcd_gdm',
    'atcd_macrosomie',
    'sopk',
    'sedentarite',
    'tabagisme',
    'hta_ou_preeclampsie',
]

MODEL_A_FEATURES = MODEL_A_NUMERIC + MODEL_A_CATEGORICAL

# ============================================================
# MODÈLE B — Modèle A + niveau_etude (ABLATION STUDY)
# ============================================================
MODEL_B_CATEGORICAL = MODEL_A_CATEGORICAL + ['niveau_etude']
MODEL_B_FEATURES = MODEL_A_NUMERIC + MODEL_B_CATEGORICAL


def build_preprocessing_pipeline(numeric_cols, categorical_cols, use_mice=True):
    """
    Construit le ColumnTransformer de preprocessing.

    - Numériques : imputation (MICE si use_mice=True, sinon médiane) + StandardScaler
    - Catégorielles : imputation (mode) + OneHotEncoder

    IMPORTANT : imc_ordinal et tension_ordinal sont déjà des entiers ordinaux
    (1-5 et 1-3), on les traite comme numériques pour préserver l'ordre clinique
    (un OneHot perdrait l'information d'ordre "plus grave que").
    """
    numeric_imputer = (
        IterativeImputer(random_state=42, max_iter=15)
        if use_mice else
        SimpleImputer(strategy='median')
    )

    numeric_transformer = Pipeline(steps=[
        ('imputer', numeric_imputer),
        ('scaler', StandardScaler()),
    ])

    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False)),
    ])

    preprocessor = ColumnTransformer(transformers=[
        ('num', numeric_transformer, numeric_cols),
        ('cat', categorical_transformer, categorical_cols),
    ])

    return preprocessor


def get_feature_names(preprocessor: ColumnTransformer) -> list:
    """Récupère les noms de features après transformation (pour SHAP / feature importance)."""
    return list(preprocessor.get_feature_names_out())
