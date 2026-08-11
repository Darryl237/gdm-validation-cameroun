"""
data_cleaning.py
==================
Fonctions de nettoyage et d'harmonisation pour les 3 datasets GDM.
Décisions verrouillées après discussion stratégique (voir README.md).

RÈGLE D'OR : ces fonctions sont appliquées à CHAQUE dataset séparément,
mais les règles de binning (IMC, tension) sont définies une seule fois
ici pour garantir une cohérence stricte entre France et Cameroun.
"""

import pandas as pd
import numpy as np


# ============================================================
# 1. BINNING IMC — Catégories OMS (PAS de midpoint)
# ============================================================

IMC_BINS = [0, 18.5, 25, 30, 35, 100]
IMC_LABELS = ['Maigreur', 'Normal', 'Surpoids', 'Obesite_I', 'Obesite_II_III']
IMC_ORDINAL_MAP = {'Maigreur': 1, 'Normal': 2, 'Surpoids': 3, 'Obesite_I': 4, 'Obesite_II_III': 5}

def bin_imc_continu(imc_series: pd.Series) -> pd.Series:
    """Binning OMS pour IMC continu (France, Négatifs)."""
    cats = pd.cut(imc_series, bins=IMC_BINS, labels=IMC_LABELS, right=False)
    return cats.map(IMC_ORDINAL_MAP)

def parse_imc_categoriel_positifs(imc_str_series: pd.Series) -> pd.Series:
    """
    Parse l'IMC catégoriel texte des Positifs vers le MÊME encodage ordinal OMS.
    Valeurs attendues: '<18', '18-24,9', '25-29,9', '30-34,9', '>35', et quelques floats isolés.
    """
    mapping = {
        '<18': 1,
        '18-24,9': 2,
        '25-29,9': 3,
        '30-34,9': 4,
        '>35': 5,
    }
    def convert(val):
        if pd.isna(val):
            return np.nan
        val = str(val).strip()
        if val in mapping:
            return mapping[val]
        # Cas valeurs numériques isolées (ex: "30.5")
        try:
            f = float(val.replace(',', '.'))
            if f < 18.5: return 1
            elif f < 25: return 2
            elif f < 30: return 3
            elif f < 35: return 4
            else: return 5
        except ValueError:
            return np.nan
    return imc_str_series.apply(convert)


# ============================================================
# 2. BINNING TENSION ARTÉRIELLE — Profil ordinal 3 niveaux
# ============================================================
# ⚠️ Seuils à CONFIRMER avec David Ben Zaza (protocole clinique exact CHU)
# Seuils provisoires basés sur les tranches observées dans Positifs :
#   Normal          : systolique < 120 ET diastolique < 90
#   Élevé            : 120 <= systolique < 140
#   Hypertension     : systolique >= 140 OU diastolique >= 90

TENSION_ORDINAL_MAP = {'Normal': 1, 'Eleve': 2, 'Hypertension': 3}

def bin_tension_continue(ta_sys: pd.Series, ta_dia: pd.Series) -> pd.Series:
    """Binning ordinal pour tension continue (France, Négatifs)."""
    def classify(sys, dia):
        if pd.isna(sys) or pd.isna(dia):
            return np.nan
        if sys >= 140 or dia >= 90:
            return TENSION_ORDINAL_MAP['Hypertension']
        elif sys >= 120:
            return TENSION_ORDINAL_MAP['Eleve']
        else:
            return TENSION_ORDINAL_MAP['Normal']
    return pd.Series(
        [classify(s, d) for s, d in zip(ta_sys, ta_dia)],
        index=ta_sys.index
    )

def parse_tension_categorielle_positifs(ta_str_series: pd.Series) -> pd.Series:
    """Parse la tension catégorielle texte des Positifs vers le même encodage ordinal."""
    mapping = {
        '90-120/60-90': TENSION_ORDINAL_MAP['Normal'],
        '120-139/90': TENSION_ORDINAL_MAP['Eleve'],
        '>140/>90': TENSION_ORDINAL_MAP['Hypertension'],
    }
    return ta_str_series.map(mapping)


# ============================================================
# 3. VARIABLE COMBINÉE HTA + PRÉ-ÉCLAMPSIE
# ============================================================

def combine_hta_preeclampsie(hta: pd.Series, preeclampsie: pd.Series) -> pd.Series:
    """
    Crée une variable unifiée 'hta_ou_preeclampsie'.
    Oui si hta_chronique=='Oui' OU atcd_preeclampsie=='Oui'.
    NaN si les deux sont NaN.
    """
    def combine(h, p):
        if pd.isna(h) and pd.isna(p):
            return np.nan
        h_bool = (str(h).strip().lower() == 'oui')
        p_bool = (str(p).strip().lower() == 'oui')
        return 'Oui' if (h_bool or p_bool) else 'Non'
    return pd.Series([combine(h, p) for h, p in zip(hta, preeclampsie)], index=hta.index)


# ============================================================
# 4. NORMALISATION VALEURS TEXTE (casse, espaces, virgules)
# ============================================================

def normalize_oui_non(series: pd.Series) -> pd.Series:
    """Normalise Oui/Non/oui/non/OUI -> 'Oui'/'Non', garde NaN."""
    def norm(v):
        if pd.isna(v):
            return np.nan
        v = str(v).strip().lower()
        if v == 'oui':
            return 'Oui'
        elif v == 'non':
            return 'Non'
        return np.nan  # valeur inattendue -> traité comme manquant, à auditer
    return series.apply(norm)


def parse_age_positifs(age_series: pd.Series) -> pd.Series:
    """Parse '25 ans' -> 25.0, garde les valeurs déjà numériques."""
    def parse(v):
        if pd.isna(v):
            return np.nan
        v = str(v).strip().replace(' ans', '').replace('ans', '')
        try:
            return float(v)
        except ValueError:
            return np.nan
    return age_series.apply(parse)


def parse_sa_positifs(sa_series: pd.Series) -> pd.Series:
    """Parse '11SA' -> 11.0."""
    def parse(v):
        if pd.isna(v):
            return np.nan
        v = str(v).strip().upper().replace('SA', '').strip()
        try:
            return float(v)
        except ValueError:
            return np.nan
    return sa_series.apply(parse)


def parse_parite_positifs(parite_series: pd.Series) -> pd.Series:
    """
    Convertit parité numérique texte des Positifs vers catégories France/Négatifs.
    '0'->Nullipare, '1'->Primipare, '2'->Multipare_2, '3+'->Multipare_3
    Valeurs aberrantes (ex: 'A') -> NaN (à auditer avec David).
    """
    def convert(v):
        if pd.isna(v):
            return np.nan
        v = str(v).strip()
        try:
            n = int(float(v))
        except ValueError:
            return np.nan  # ex: 'A' — anomalie de saisie, flaguée
        if n == 0:
            return 'Nullipare'
        elif n == 1:
            return 'Primipare'
        elif n == 2:
            return 'Multipare_2'
        else:
            return 'Multipare_3'
    return parite_series.apply(convert)


def parse_glycemie_virgule(series: pd.Series) -> pd.Series:
    """Parse '0,97' (virgule française) -> 0.97 (float)."""
    def parse(v):
        if pd.isna(v):
            return np.nan
        v = str(v).strip().replace(',', '.')
        try:
            return float(v)
        except ValueError:
            return np.nan
    return series.apply(parse)


# ============================================================
# 5. GESTION "Non_renseigne" -> NaN (dataset Négatifs)
# ============================================================

def non_renseigne_to_nan(df: pd.DataFrame, cols: list) -> pd.DataFrame:
    """Remplace 'Non_renseigne' par NaN sur les colonnes spécifiées."""
    df = df.copy()
    for col in cols:
        df[col] = df[col].replace('Non_renseigne', np.nan)
    return df


def create_flag_source_batch(df: pd.DataFrame, check_cols: list) -> pd.Series:
    """
    Crée le flag_source_batch : 1 si TOUTES les check_cols sont manquantes
    (signature du bloc des 108 premières lignes Négatifs), 0 sinon.
    USAGE : test de sensibilité interne uniquement (notebook 06),
    JAMAIS comme feature du modèle final.
    """
    return df[check_cols].isna().all(axis=1).astype(int)
