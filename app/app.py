"""
app.py — Application IA de prédiction du risque de Diabète Gestationnel (v3)
==============================================================================
CHU Yaoundé — Outil de support à la décision clinique

v3 : authentification (comptes praticiens pré-configurés), navigation multi-pages,
historique des prédictions avec graphiques SVG natifs (zéro dépendance CDN — le
service fonctionne offline, conformément au principe d'infrastructure légère
posé en Section 5.2.3 et au risque de connectivité documenté en Section 6).

Principe éthique (Charte Section 6, Principe 4) :
Cet outil SUPPORTE la décision clinique, il ne la REMPLACE jamais.
"""

from flask import Flask, render_template, request, jsonify, session, redirect, url_for
from werkzeug.security import generate_password_hash, check_password_hash
from werkzeug.utils import secure_filename
from functools import wraps
import joblib
import numpy as np
import pandas as pd
import os
import json
import sqlite3
from datetime import datetime

app = Flask(__name__)
app.secret_key = os.environ.get('GDM_APP_SECRET', 'dev-secret-key-a-changer-en-production')

UPLOAD_FOLDER = 'static/uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

# Fiabilité du modèle par sous-groupe — résultats VALIDÉS Section 7.7 (notebook 07).
# Utilisés pour visualiser concrètement la fiabilité attendue pour CE profil précis,
# plutôt qu'un simple avertissement textuel générique.
IMC_FAIRNESS = {
    1: None,  # Maigreur — effectif insuffisant en validation (Section 7.7), non estimable
    2: {'sensibilite': 0.2632, 'specificite': 0.8312, 'label': 'IMC Normal'},
    3: {'sensibilite': 0.7308, 'specificite': 0.3882, 'label': 'Surpoids'},
    4: {'sensibilite': 1.0000, 'specificite': 0.0455, 'label': 'Obésité I'},
    5: {'sensibilite': 1.0000, 'specificite': 0.0000, 'label': 'Obésité II/III'},
}
IMC_FAIRNESS_EN = {
    1: None,
    2: {'sensibilite': 0.2632, 'specificite': 0.8312, 'label': 'Normal BMI'},
    3: {'sensibilite': 0.7308, 'specificite': 0.3882, 'label': 'Overweight'},
    4: {'sensibilite': 1.0000, 'specificite': 0.0455, 'label': 'Obesity I'},
    5: {'sensibilite': 1.0000, 'specificite': 0.0000, 'label': 'Obesity II/III'},
}

def get_age_fairness(age, lang='fr'):
    try:
        age_val = float(age)
    except (TypeError, ValueError):
        return None
    if lang == 'en':
        if age_val < 25:
            return {'sensibilite': 0.5400, 'specificite': 0.8605, 'label': '< 25 years'}
        elif age_val <= 35:
            return {'sensibilite': 0.8992, 'specificite': 0.4345, 'label': '25-35 years'}
        else:
            return {'sensibilite': 1.0000, 'specificite': 0.0000, 'label': '> 35 years'}
    if age_val < 25:
        return {'sensibilite': 0.5400, 'specificite': 0.8605, 'label': '< 25 ans'}
    elif age_val <= 35:
        return {'sensibilite': 0.8992, 'specificite': 0.4345, 'label': '25-35 ans'}
    else:
        return {'sensibilite': 1.0000, 'specificite': 0.0000, 'label': '> 35 ans'}


MODEL_PATH = '../models/model_A.pkl'
PREPROCESSOR_PATH = '../models/preprocessor_A.pkl'
VALIDATION_METRICS_PATH = '../results/validation_metrics.json'
DB_PATH = 'gdm_app.db'

model = None
preprocessor = None
SEUIL_DECISION = 0.5
VALIDATION_INFO = {
    'sensibilite': None, 'specificite': None, 'auc_roc': None,
    'n_total': 455, 'n_positifs': 243, 'n_negatifs': 212
}

# ============================================================
# BASE DE DONNÉES — Comptes praticiens & historique (SQLite local)
# ============================================================

def get_db():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    conn = get_db()
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS users (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        username TEXT UNIQUE NOT NULL,
        password_hash TEXT NOT NULL,
        nom_complet TEXT,
        role TEXT,
        photo_path TEXT,
        is_admin INTEGER DEFAULT 0
    )''')
    # Migration sûre pour les bases déjà créées avant l'ajout de photo_path / is_admin
    for col_def in ['photo_path TEXT', 'is_admin INTEGER DEFAULT 0']:
        try:
            c.execute(f'ALTER TABLE users ADD COLUMN {col_def}')
            conn.commit()
        except sqlite3.OperationalError:
            pass  # colonne déjà présente
    c.execute('''CREATE TABLE IF NOT EXISTS predictions (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id INTEGER,
        timestamp TEXT,
        probabilite REAL,
        classification TEXT,
        age_maternel REAL,
        imc_ordinal INTEGER,
        recommandation TEXT,
        contributions_json TEXT,
        equity_caveats_json TEXT,
        fiabilite_imc_json TEXT,
        fiabilite_age_json TEXT,
        seuil_utilise REAL
    )''')
    # Migration sûre pour les bases déjà créées avant l'ajout de ces colonnes
    for col_def in ['recommandation TEXT', 'contributions_json TEXT', 'equity_caveats_json TEXT',
                     'fiabilite_imc_json TEXT', 'fiabilite_age_json TEXT', 'seuil_utilise REAL']:
        try:
            c.execute(f'ALTER TABLE predictions ADD COLUMN {col_def}')
            conn.commit()
        except sqlite3.OperationalError:
            pass  # colonne déjà présente
    conn.commit()

    c.execute('SELECT COUNT(*) as n FROM users')
    if c.fetchone()['n'] == 0:
        # Comptes de démonstration — à remplacer par un vrai processus d'onboarding
        # avant tout déploiement réel (hors périmètre de cette thèse).
        demo_accounts = [
            ('dr.fotso', 'CHU2026!', 'Dr. Fotso', 'Gynécologue-Obstétricien', 0),
            ('sage.femme', 'CHU2026!', 'Mme Nkolo', 'Sage-femme', 0),
            ('admin', 'AdminCHU2026!', 'Administrateur Système', 'Administrateur', 1),
        ]
        for username, pwd, nom, role, is_admin in demo_accounts:
            c.execute('INSERT INTO users (username, password_hash, nom_complet, role, is_admin) VALUES (?,?,?,?,?)',
                       (username, generate_password_hash(pwd), nom, role, is_admin))
        conn.commit()
        print("✅ Comptes de démonstration créés : dr.fotso / sage.femme / admin (mots de passe : CHU2026! / CHU2026! / AdminCHU2026!)")
    conn.close()


def login_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        if 'user_id' not in session:
            return redirect(url_for('login'))
        return f(*args, **kwargs)
    return decorated


def admin_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        if 'user_id' not in session:
            return redirect(url_for('login'))
        if not session.get('is_admin'):
            return redirect(url_for('home'))
        return f(*args, **kwargs)
    return decorated


# ============================================================
# CONFIGURATION DES CHAMPS DU FORMULAIRE
# ============================================================
FEATURES_INFO_EN = {
    'age_maternel': {'label': 'Maternal age (years)', 'type': 'number', 'min': 15, 'max': 55,
                      'section': 'demographique', 'icon': '🎂'},
    'imc_ordinal': {'label': 'BMI category (WHO)', 'type': 'select', 'section': 'demographique', 'icon': '⚖️',
                     'options': [(1, 'Underweight (<18.5)'), (2, 'Normal (18.5-24.9)'),
                                 (3, 'Overweight (25-29.9)'), (4, 'Obesity I (30-34.9)'),
                                 (5, 'Obesity II/III (≥35)')]},
    'tension_ordinal': {'label': 'Blood pressure profile', 'type': 'select', 'section': 'demographique', 'icon': '💓',
                         'options': [(1, 'Normal'), (2, 'Elevated'), (3, 'Hypertension')]},
    'sa_premiere_consult': {'label': "Gestational weeks at first visit", 'type': 'number',
                             'min': 4, 'max': 42, 'section': 'demographique', 'icon': '📅'},
    'parite': {'label': 'Parity', 'type': 'select', 'section': 'demographique', 'icon': '👶',
               'options': [('Nullipare', 'Nulliparous'), ('Primipare', 'Primiparous'),
                           ('Multipare_2', 'Multiparous (2)'), ('Multipare_3', 'Multiparous (3+)')]},
    'atcd_familial_diabete_1er_deg': {'label': 'Family history of diabetes (1st degree)', 'type': 'boolean',
                                       'section': 'antecedents', 'icon': '🧬'},
    'atcd_gdm': {'label': 'History of gestational diabetes', 'type': 'boolean',
                 'section': 'antecedents', 'icon': '📋'},
    'atcd_macrosomie': {'label': 'History of macrosomia (>4kg)', 'type': 'boolean',
                         'section': 'antecedents', 'icon': '📋'},
    'sopk': {'label': 'History of PCOS', 'type': 'boolean', 'section': 'antecedents', 'icon': '📋'},
    'hta_ou_preeclampsie': {'label': 'History of hypertension or pre-eclampsia', 'type': 'boolean',
                             'section': 'antecedents', 'icon': '📋'},
    'sedentarite': {'label': 'Sedentary lifestyle (<150 min activity/week)', 'type': 'boolean',
                     'section': 'mode_de_vie', 'icon': '🏃'},
    'tabagisme': {'label': 'Smoking (active/passive)', 'type': 'boolean',
                  'section': 'mode_de_vie', 'icon': '🚬'},
}

SECTIONS_EN = [
    {'id': 'demographique', 'label': 'Demographic & Clinical Profile', 'icon': '👤'},
    {'id': 'antecedents', 'label': 'Medical History', 'icon': '🏥'},
    {'id': 'mode_de_vie', 'label': 'Lifestyle', 'icon': '🌱'},
]

LABEL_MAP_EN = {
    'num__age_maternel': "maternal age",
    'num__imc_ordinal': "BMI category",
    'num__tension_ordinal': "blood pressure profile",
    'num__sa_premiere_consult': "gestational age at first visit",
    'cat__parite_Nullipare': "nulliparity",
    'cat__parite_Primipare': "primiparity",
    'cat__parite_Multipare_2': "multiparity (2)",
    'cat__parite_Multipare_3': "multiparity (3+)",
    'cat__atcd_familial_diabete_1er_deg_Oui': "family history of diabetes",
    'cat__atcd_familial_diabete_1er_deg_Non': "absence of family history of diabetes",
    'cat__atcd_gdm_Oui': "history of gestational diabetes",
    'cat__atcd_gdm_Non': "absence of history of gestational diabetes",
    'cat__atcd_macrosomie_Oui': "history of macrosomia",
    'cat__atcd_macrosomie_Non': "absence of history of macrosomia",
    'cat__sopk_Oui': "history of PCOS",
    'cat__sopk_Non': "absence of history of PCOS",
    'cat__sedentarite_Oui': "sedentary lifestyle",
    'cat__sedentarite_Non': "sufficient physical activity",
    'cat__tabagisme_Oui': "smoking",
    'cat__tabagisme_Non': "non-smoking status",
    'cat__hta_ou_preeclampsie_Oui': "history of hypertension/pre-eclampsia",
    'cat__hta_ou_preeclampsie_Non': "absence of history of hypertension/pre-eclampsia",
}

DEMO_PATIENTS_EN_LABELS = {
    'profil_bas_risque': '🟢 Profile A — Young, normal weight, no history',
    'profil_haut_risque': '🔴 Profile B — Older age, obesity, multiple risk factors',
    'profil_zone_incertaine': "🟡 Profile C — Overweight, mixed profile (model's uncertainty zone)",
}

FEATURES_INFO = {
    'age_maternel': {'label': 'Âge maternel (années)', 'type': 'number', 'min': 15, 'max': 55,
                      'section': 'demographique', 'icon': '🎂'},
    'imc_ordinal': {'label': 'Catégorie IMC (OMS)', 'type': 'select', 'section': 'demographique', 'icon': '⚖️',
                     'options': [(1, 'Maigreur (<18.5)'), (2, 'Normal (18.5-24.9)'),
                                 (3, 'Surpoids (25-29.9)'), (4, 'Obésité I (30-34.9)'),
                                 (5, 'Obésité II/III (≥35)')]},
    'tension_ordinal': {'label': 'Profil tensionnel', 'type': 'select', 'section': 'demographique', 'icon': '💓',
                         'options': [(1, 'Normal'), (2, 'Élevé'), (3, 'Hypertension')]},
    'sa_premiere_consult': {'label': "Semaines d'aménorrhée à la 1ère consultation", 'type': 'number',
                             'min': 4, 'max': 42, 'section': 'demographique', 'icon': '📅'},
    'parite': {'label': 'Parité', 'type': 'select', 'section': 'demographique', 'icon': '👶',
               'options': [('Nullipare', 'Nullipare'), ('Primipare', 'Primipare'),
                           ('Multipare_2', 'Multipare (2)'), ('Multipare_3', 'Multipare (3+)')]},
    'atcd_familial_diabete_1er_deg': {'label': 'Antécédent familial de diabète (1er degré)', 'type': 'boolean',
                                       'section': 'antecedents', 'icon': '🧬'},
    'atcd_gdm': {'label': 'Antécédent de diabète gestationnel', 'type': 'boolean',
                 'section': 'antecedents', 'icon': '📋'},
    'atcd_macrosomie': {'label': 'Antécédent de macrosomie (>4kg)', 'type': 'boolean',
                         'section': 'antecedents', 'icon': '📋'},
    'sopk': {'label': 'Antécédent de SOPK', 'type': 'boolean', 'section': 'antecedents', 'icon': '📋'},
    'hta_ou_preeclampsie': {'label': 'Antécédent HTA ou pré-éclampsie', 'type': 'boolean',
                             'section': 'antecedents', 'icon': '📋'},
    'sedentarite': {'label': 'Sédentarité (<150 min activité/semaine)', 'type': 'boolean',
                     'section': 'mode_de_vie', 'icon': '🏃'},
    'tabagisme': {'label': 'Tabagisme (actif/passif)', 'type': 'boolean',
                  'section': 'mode_de_vie', 'icon': '🚬'},
}

SECTIONS = [
    {'id': 'demographique', 'label': 'Profil Démographique & Clinique', 'icon': '👤'},
    {'id': 'antecedents', 'label': 'Antécédents Médicaux', 'icon': '🏥'},
    {'id': 'mode_de_vie', 'label': 'Mode de Vie', 'icon': '🌱'},
]

LABEL_MAP = {
    'num__age_maternel': "l'âge maternel",
    'num__imc_ordinal': "la catégorie d'IMC",
    'num__tension_ordinal': "le profil tensionnel",
    'num__sa_premiere_consult': "l'âge gestationnel à la 1ère consultation",
    'cat__parite_Nullipare': "la nulliparité",
    'cat__parite_Primipare': "la primiparité",
    'cat__parite_Multipare_2': "la multiparité (2)",
    'cat__parite_Multipare_3': "la multiparité (3+)",
    'cat__atcd_familial_diabete_1er_deg_Oui': "l'antécédent familial de diabète",
    'cat__atcd_familial_diabete_1er_deg_Non': "l'absence d'antécédent familial de diabète",
    'cat__atcd_gdm_Oui': "l'antécédent de diabète gestationnel",
    'cat__atcd_gdm_Non': "l'absence d'antécédent de diabète gestationnel",
    'cat__atcd_macrosomie_Oui': "l'antécédent de macrosomie",
    'cat__atcd_macrosomie_Non': "l'absence d'antécédent de macrosomie",
    'cat__sopk_Oui': "l'antécédent de SOPK",
    'cat__sopk_Non': "l'absence d'antécédent de SOPK",
    'cat__sedentarite_Oui': "la sédentarité",
    'cat__sedentarite_Non': "l'activité physique suffisante",
    'cat__tabagisme_Oui': "le tabagisme",
    'cat__tabagisme_Non': "le statut non-fumeuse",
    'cat__hta_ou_preeclampsie_Oui': "l'antécédent d'HTA/pré-éclampsie",
    'cat__hta_ou_preeclampsie_Non': "l'absence d'antécédent d'HTA/pré-éclampsie",
}

def get_demo_patients(lang='fr'):
    if lang != 'en':
        return DEMO_PATIENTS
    translated = {}
    for key, patient in DEMO_PATIENTS.items():
        translated[key] = {'label': DEMO_PATIENTS_EN_LABELS.get(key, patient['label']),
                            'data': patient['data']}
    return translated


# ============================================================
# TRADUCTIONS UI (chrome, navigation, textes fixes des templates)
# ============================================================
UI_STRINGS = {
    'fr': {
        'nav_accueil': 'Accueil', 'nav_prediction': 'Nouvelle Prédiction',
        'nav_historique': 'Historique', 'nav_parametres': 'Paramètres', 'nav_logout': 'Déconnexion',
        'nav_admin': '🛡️ Back Office',
        'admin_title': '🛡️ Back Office Administrateur',
        'admin_subtitle': "Vue d'ensemble des comptes praticiens et de l'activité de la plateforme.",
        'admin_col_user': "Nom d'utilisateur", 'admin_col_nom': 'Nom complet', 'admin_col_role': 'Rôle',
        'admin_col_npred': 'Prédictions', 'admin_col_derniere': 'Dernière activité', 'admin_col_type': 'Type de compte',
        'admin_total_preds': 'Prédictions totales (tous comptes)', 'admin_total_users': 'Comptes praticiens',
        'admin_db_info': 'Informations techniques base de données',
        'login_title': '🏥 GDM Predict', 'login_subtitle': 'Outil de support à la décision — CHU Yaoundé',
        'login_username': "Nom d'utilisateur", 'login_password': 'Mot de passe',
        'login_button': 'Se connecter', 'login_error': 'Identifiants incorrects.',
        'login_demo_hint': '🔑 Comptes de démonstration (thèse) :',
        'home_welcome': 'Bienvenue', 'home_subtitle': "Outil de support à la décision pour le dépistage précoce du diabète gestationnel — validé sur données réelles du CHU Yaoundé.",
        'home_perf_title': '📊 Performance du modèle (validation externe)',
        'home_sensibilite': 'Sensibilité', 'home_specificite': 'Spécificité', 'home_auc': 'AUC (ROC)',
        'home_cas_valides': 'Cas réels validés',
        'home_quickstart': '🚀 Démarrage rapide', 'home_new_pred': '➕ Nouvelle prédiction',
        'home_history_link': 'prédiction(s) dans votre historique',
        'home_warning_title': '⚠️ Rappel important',
        'home_warning_text': "Cet outil est une aide à la décision clinique. Il ne remplace en aucun cas le jugement du praticien. Sa fiabilité varie selon le profil de la patiente — consultez la Section 7.7 de la thèse pour le détail des limites documentées par sous-groupe.",
        'predict_title': '➕ Nouvelle Prédiction',
        'predict_subtitle': "Remplissez les champs disponibles — les champs non renseignés seront gérés automatiquement par le modèle (imputation validée, Section 7.3).",
        'predict_submit': 'Calculer le risque →', 'predict_calculating': 'Calcul en cours...',
        'predict_error_connexion': '⚠️ Erreur de connexion au serveur.',
        'result_high_risk': 'Haut risque', 'result_low_risk': 'Bas risque',
        'result_reco_title': 'Recommandation Clinique',
        'result_factors_title': '📊 Facteurs Ayant le Plus Influencé Cette Prédiction',
        'result_explanation_title': '📖 Explication clinique détaillée',
        'result_reliability_title': '🎯 Fiabilité Mesurée du Modèle pour ce Profil',
        'result_reliability_note': "Résultats de validation externe réels (Section 7.7) pour la catégorie IMC et la tranche d'âge de cette patiente, comparés aux cibles KPI de la thèse (lignes pointillées : 85% sensibilité, 75% spécificité).",
        'result_equity_title': '⚠️ Avertissement de Fiabilité (Section 7.7)',
        'result_new_pred': '➕ Nouvelle Prédiction', 'result_view_history': "📜 Voir l'Historique",
        'result_print': '🖨️ Imprimer ce Résultat',
        'result_no_factors': "Aucun facteur renseigné n'a eu de contribution significative (formulaire majoritairement vide).",
        'result_augmente': 'augmente le risque estimé', 'result_diminue': 'diminue le risque estimé',
        'result_disclaimer': "⚠️ Cet outil est une aide à la décision. Il ne remplace pas le jugement clinique. Seuil de décision appliqué",
        'historique_title': '📜 Historique des Prédictions',
        'historique_empty': 'Aucune prédiction enregistrée pour votre compte.',
        'historique_empty_link': 'Faire une première prédiction →',
        'historique_predictions': 'prédictions',
        'historique_chart_title': 'Probabilité par prédiction (chronologique)',
        'historique_chart_note': "Ligne pointillée = seuil de décision (85% sensibilité cible). Chaque barre = une prédiction, de la plus ancienne (gauche) à la plus récente (droite).",
        'historique_detail': 'Détail', 'historique_col_date': 'Date', 'historique_col_age': 'Âge',
        'historique_col_imc': 'IMC (cat.)', 'historique_col_prob': 'Probabilité',
        'historique_col_result': 'Résultat', 'historique_view_detail': 'Voir le détail →',
        'parametres_title': '⚙️ Paramètres & Informations Modèle',
        'parametres_subtitle': "Ces paramètres sont en lecture seule. Le seuil de décision et les caractéristiques du modèle sont déterminés par la validation externe rigoureuse documentée en Section 7 de la thèse — ils ne sont volontairement pas modifiables par l'utilisateur pour des raisons de sécurité clinique.",
        'parametres_algo': 'Algorithme', 'parametres_nfeatures': 'Nombre de variables',
        'parametres_seuil': 'Seuil de décision actuel', 'parametres_sensibilite': 'Sensibilité validée',
        'parametres_specificite': 'Spécificité validée', 'parametres_auc': 'AUC (ROC)',
        'parametres_train': "Jeu d'entraînement", 'parametres_val': 'Jeu de validation externe',
        'parametres_account_title': '👤 Mon Compte', 'parametres_photo_btn': 'Mettre à jour la photo',
        'parametres_name': 'Nom', 'parametres_role': 'Rôle',
        'parametres_train_desc': 'France (synthétique, 30 000 femmes)',
        'parametres_val_desc': 'Cameroun (réel)', 'parametres_account_note': "Gestion des comptes (création, modification, réinitialisation de mot de passe) hors périmètre de cette démonstration — prévue pour un déploiement réel encadré par le service informatique du CHU (cf. Section 9, perspectives).",
        'result_sens_label': 'Sensibilité', 'result_spec_label': 'Spécificité',
        'result_predicted_on': 'Prédiction du', 'result_age_label': 'Âge', 'result_imc_cat_label': 'IMC catégorie',
        'lang_switch': 'English', 'form_not_specified': '-- Non renseigné --',
        'form_placeholder_empty': 'Non renseigné si vide', 'form_oui': 'Oui', 'form_non': 'Non',
    },
    'en': {
        'nav_accueil': 'Home', 'nav_prediction': 'New Prediction',
        'nav_historique': 'History', 'nav_parametres': 'Settings', 'nav_logout': 'Log out',
        'nav_admin': '🛡️ Back Office',
        'admin_title': '🛡️ Administrator Back Office',
        'admin_subtitle': 'Overview of practitioner accounts and platform activity.',
        'admin_col_user': 'Username', 'admin_col_nom': 'Full name', 'admin_col_role': 'Role',
        'admin_col_npred': 'Predictions', 'admin_col_derniere': 'Last activity', 'admin_col_type': 'Account type',
        'admin_total_preds': 'Total predictions (all accounts)', 'admin_total_users': 'Practitioner accounts',
        'admin_db_info': 'Technical database information',
        'login_title': '🏥 GDM Predict', 'login_subtitle': 'Clinical decision support tool — CHU Yaoundé',
        'login_username': 'Username', 'login_password': 'Password',
        'login_button': 'Sign in', 'login_error': 'Incorrect credentials.',
        'login_demo_hint': '🔑 Demo accounts (thesis):',
        'home_welcome': 'Welcome', 'home_subtitle': "Decision support tool for early screening of gestational diabetes — validated on real CHU Yaoundé data.",
        'home_perf_title': '📊 Model Performance (external validation)',
        'home_sensibilite': 'Sensitivity', 'home_specificite': 'Specificity', 'home_auc': 'AUC (ROC)',
        'home_cas_valides': 'Real validated cases',
        'home_quickstart': '🚀 Quick Start', 'home_new_pred': '➕ New prediction',
        'home_history_link': 'prediction(s) in your history',
        'home_warning_title': '⚠️ Important reminder',
        'home_warning_text': "This tool is a clinical decision support aid. It never replaces the practitioner's judgment. Its reliability varies according to the patient's profile — see Section 7.7 of the thesis for details on limitations documented by subgroup.",
        'predict_title': '➕ New Prediction',
        'predict_subtitle': "Fill in the available fields — unanswered fields are handled automatically by the model (validated imputation, Section 7.3).",
        'predict_submit': 'Calculate risk →', 'predict_calculating': 'Calculating...',
        'predict_error_connexion': '⚠️ Server connection error.',
        'result_high_risk': 'High risk', 'result_low_risk': 'Low risk',
        'result_reco_title': 'Clinical Recommendation',
        'result_factors_title': '📊 Factors That Most Influenced This Prediction',
        'result_explanation_title': '📖 Detailed clinical explanation',
        'result_reliability_title': '🎯 Measured Model Reliability for this Profile',
        'result_reliability_note': "Real external validation results (Section 7.7) for this patient's BMI category and age group, compared to the thesis KPI targets (dotted lines: 85% sensitivity, 75% specificity).",
        'result_equity_title': '⚠️ Reliability Warning (Section 7.7)',
        'result_new_pred': '➕ New Prediction', 'result_view_history': "📜 View History",
        'result_print': '🖨️ Print This Result',
        'result_no_factors': "No provided factor had a significant contribution (mostly empty form).",
        'result_augmente': 'increases the estimated risk', 'result_diminue': 'decreases the estimated risk',
        'result_disclaimer': "⚠️ This tool is a decision support aid. It does not replace clinical judgment. Decision threshold applied",
        'historique_title': '📜 Prediction History',
        'historique_empty': 'No predictions recorded for your account.',
        'historique_empty_link': 'Make a first prediction →',
        'historique_predictions': 'predictions',
        'historique_chart_title': 'Probability per prediction (chronological)',
        'historique_chart_note': "Dotted line = decision threshold (85% target sensitivity). Each bar = one prediction, from oldest (left) to most recent (right).",
        'historique_detail': 'Detail', 'historique_col_date': 'Date', 'historique_col_age': 'Age',
        'historique_col_imc': 'BMI (cat.)', 'historique_col_prob': 'Probability',
        'historique_col_result': 'Result', 'historique_view_detail': 'View detail →',
        'parametres_title': '⚙️ Settings & Model Information',
        'parametres_subtitle': "These settings are read-only. The decision threshold and model characteristics are determined by the rigorous external validation documented in Section 7 of the thesis — they are intentionally not user-editable for clinical safety reasons.",
        'parametres_algo': 'Algorithm', 'parametres_nfeatures': 'Number of features',
        'parametres_seuil': 'Current decision threshold', 'parametres_sensibilite': 'Validated sensitivity',
        'parametres_specificite': 'Validated specificity', 'parametres_auc': 'AUC (ROC)',
        'parametres_train': 'Training set', 'parametres_val': 'External validation set',
        'parametres_account_title': '👤 My Account', 'parametres_photo_btn': 'Update photo',
        'parametres_name': 'Name', 'parametres_role': 'Role',
        'parametres_train_desc': 'France (synthetic, 30,000 women)',
        'parametres_val_desc': 'Cameroon (real)', 'parametres_account_note': "Account management (creation, editing, password reset) is out of scope for this demonstration — planned for a real deployment overseen by the CHU IT department (see Section 9, perspectives).",
        'result_sens_label': 'Sensitivity', 'result_spec_label': 'Specificity',
        'result_predicted_on': 'Prediction from', 'result_age_label': 'Age', 'result_imc_cat_label': 'BMI category',
        'lang_switch': 'Français', 'form_not_specified': '-- Not specified --',
        'form_placeholder_empty': 'Leave empty if unknown', 'form_oui': 'Yes', 'form_non': 'No',
    },
}


def current_lang():
    return session.get('lang', 'fr')


@app.context_processor
def inject_i18n():
    lang = current_lang()
    def ui(key):
        return UI_STRINGS.get(lang, UI_STRINGS['fr']).get(key, UI_STRINGS['fr'].get(key, key))
    return dict(ui=ui, current_lang=lang)


@app.route('/set-language/<lang_code>')
def set_language(lang_code):
    if lang_code in ('fr', 'en'):
        session['lang'] = lang_code
    return redirect(request.referrer or url_for('home'))


DEMO_PATIENTS = {

    'profil_bas_risque': {
        'label': '🟢 Profil A — Jeune, poids normal, sans antécédents',
        'data': {'age_maternel': 24, 'imc_ordinal': 2, 'tension_ordinal': 1, 'sa_premiere_consult': 10,
                  'parite': 'Nullipare', 'atcd_familial_diabete_1er_deg': 'Non', 'atcd_gdm': 'Non',
                  'atcd_macrosomie': 'Non', 'sopk': 'Non', 'hta_ou_preeclampsie': 'Non',
                  'sedentarite': 'Non', 'tabagisme': 'Non'}
    },
    'profil_haut_risque': {
        'label': '🔴 Profil B — Âge élevé, obésité, antécédents multiples',
        'data': {'age_maternel': 38, 'imc_ordinal': 5, 'tension_ordinal': 3, 'sa_premiere_consult': 22,
                  'parite': 'Multipare_3', 'atcd_familial_diabete_1er_deg': 'Oui', 'atcd_gdm': 'Oui',
                  'atcd_macrosomie': 'Oui', 'sopk': 'Non', 'hta_ou_preeclampsie': 'Oui',
                  'sedentarite': 'Oui', 'tabagisme': 'Non'}
    },
    'profil_zone_incertaine': {
        'label': "🟡 Profil C — Surpoids, profil mixte (zone d'incertitude du modèle)",
        'data': {'age_maternel': 29, 'imc_ordinal': 3, 'tension_ordinal': 2, 'sa_premiere_consult': 16,
                  'parite': 'Primipare', 'atcd_familial_diabete_1er_deg': 'Non', 'atcd_gdm': 'Non',
                  'atcd_macrosomie': 'Non', 'sopk': 'Oui', 'hta_ou_preeclampsie': 'Non',
                  'sedentarite': 'Non', 'tabagisme': 'Non'}
    },
}


def load_model():
    global model, preprocessor
    if os.path.exists(MODEL_PATH) and os.path.exists(PREPROCESSOR_PATH):
        model = joblib.load(MODEL_PATH)
        preprocessor = joblib.load(PREPROCESSOR_PATH)
        return True
    return False


def load_seuil_officiel():
    global SEUIL_DECISION, VALIDATION_INFO
    if os.path.exists(VALIDATION_METRICS_PATH):
        with open(VALIDATION_METRICS_PATH) as f:
            metrics = json.load(f)
        SEUIL_DECISION = metrics['seuil_ajuste_A']
        m = metrics.get('modele_A_seuil_ajuste_85pct', {})
        VALIDATION_INFO['sensibilite'] = m.get('sensibilite')
        VALIDATION_INFO['specificite'] = m.get('specificite')
        VALIDATION_INFO['auc_roc'] = m.get('auc_roc')
        VALIDATION_INFO['n_total'] = m.get('n_total', 455)
        VALIDATION_INFO['n_positifs'] = m.get('n_positifs', 243)
        VALIDATION_INFO['n_negatifs'] = m.get('n_negatifs', 212)
        print(f"✅ Seuil officiel chargé depuis notebook 05 : {SEUIL_DECISION:.4f}")
        return True
    print(f"⚠️ {VALIDATION_METRICS_PATH} introuvable — seuil de repli 0.5 utilisé.")
    return False


# Explications cliniques détaillées par facteur (texte pré-écrit, déterministe — pas de LLM,
# décision actée après discussion des risques). Chaque texte reste général et prudent
# (pas de statistique chiffrée non vérifiée) pour rester défendable méthodologiquement.
DETAILED_EXPLANATIONS = {
    'num__age_maternel_haut': "Un âge maternel avancé est associé dans la littérature clinique à une sensibilité à l'insuline généralement réduite, ce qui peut favoriser l'apparition d'un diabète gestationnel.",
    'num__age_maternel_bas': "Un âge maternel plus jeune est habituellement associé à un risque métabolique plus faible dans la littérature générale.",
    'num__imc_ordinal_haut': "Une catégorie d'IMC élevée (surpoids ou obésité) est l'un des facteurs de risque les plus établis du diabète gestationnel : l'excès de tissu adipeux est associé à une résistance accrue à l'insuline. Note : la Section 7.7 de cette thèse documente une spécificité du modèle très faible pour les catégories d'obésité — ce facteur doit être interprété avec prudence.",
    'num__imc_ordinal_bas': "Une catégorie d'IMC plus basse est habituellement rassurante sur le plan métabolique. Note : la Section 7.7 documente une sensibilité réduite du modèle pour l'IMC normal — un résultat bas risque doit être interprété avec prudence clinique renforcée pour ce profil.",
    'num__tension_ordinal_haut': "Un profil tensionnel élevé ou hypertensif peut s'accompagner d'altérations métaboliques partagées avec le diabète gestationnel.",
    'num__tension_ordinal_bas': "Un profil tensionnel normal est un facteur rassurant, sans association attendue avec un risque métabolique accru.",
    'num__sa_premiere_consult_haut': "Une première consultation tardive limite les données disponibles au moment du dépistage précoce — la Section 7.2.3 de cette thèse documente une proportion notable de consultations tardives dans le contexte camerounais.",
    'num__sa_premiere_consult_bas': "Une première consultation précoce permet une évaluation dans des conditions proches de celles du jeu d'entraînement (France, consultation moyenne à 9.5 SA).",
    'cat__parite_Nullipare': "La nulliparité (première grossesse) ne constitue pas en soi un facteur de risque établi du diabète gestationnel, contrairement à la multiparité élevée.",
    'cat__parite_Primipare': "La primiparité est associée à un profil de risque de référence dans la littérature clinique.",
    'cat__parite_Multipare_2': "Une parité de rang 2 peut être associée à un risque légèrement accru, en lien avec l'âge cumulé et l'historique métabolique des grossesses précédentes.",
    'cat__parite_Multipare_3': "Une multiparité élevée (3 grossesses ou plus) est associée dans la littérature à un risque légèrement accru de diabète gestationnel.",
    'cat__atcd_familial_diabete_1er_deg_Oui': "Un antécédent familial de diabète au premier degré (parent, frère/sœur) constitue un facteur de risque génétique reconnu et documenté du diabète gestationnel.",
    'cat__atcd_familial_diabete_1er_deg_Non': "L'absence d'antécédent familial de diabète est un facteur rassurant qui réduit la probabilité estimée.",
    'cat__atcd_gdm_Oui': "Un antécédent personnel de diabète gestationnel lors d'une grossesse précédente est l'un des facteurs prédictifs les plus robustes de récidive documentés dans la littérature clinique.",
    'cat__atcd_gdm_Non': "L'absence d'antécédent personnel de diabète gestationnel est un facteur rassurant important.",
    'cat__atcd_macrosomie_Oui': "Un antécédent de macrosomie fœtale (nouveau-né de plus de 4kg) peut être un signe indirect d'un diabète gestationnel non diagnostiqué lors d'une grossesse antérieure.",
    'cat__atcd_macrosomie_Non': "L'absence d'antécédent de macrosomie est un facteur rassurant.",
    'cat__sopk_Oui': "Le syndrome des ovaires polykystiques (SOPK) est associé à une insulino-résistance qui peut favoriser le développement d'un diabète gestationnel.",
    'cat__sopk_Non': "L'absence de SOPK est un facteur rassurant sur le plan métabolique.",
    'cat__sedentarite_Oui': "La sédentarité (moins de 150 minutes d'activité physique hebdomadaire) est un facteur de risque modifiable associé à une moindre sensibilité à l'insuline.",
    'cat__sedentarite_Non': "Une activité physique suffisante est un facteur protecteur reconnu vis-à-vis du risque métabolique.",
    'cat__tabagisme_Oui': "Le tabagisme peut interagir avec le métabolisme glucidique, bien que son association avec le diabète gestationnel spécifiquement soit moins établie que pour d'autres complications de grossesse.",
    'cat__tabagisme_Non': "L'absence de tabagisme est favorable au profil métabolique général.",
    'cat__hta_ou_preeclampsie_Oui': "Un antécédent d'hypertension artérielle ou de pré-éclampsie partage des mécanismes physiopathologiques vasculaires et métaboliques avec le diabète gestationnel.",
    'cat__hta_ou_preeclampsie_Non': "L'absence d'antécédent d'HTA ou de pré-éclampsie est un facteur rassurant.",
}


DETAILED_EXPLANATIONS_EN = {
    'num__age_maternel_haut': "Advanced maternal age is associated in the clinical literature with generally reduced insulin sensitivity, which may favor the onset of gestational diabetes.",
    'num__age_maternel_bas': "Younger maternal age is usually associated with lower metabolic risk in the general literature.",
    'num__imc_ordinal_haut': "A high BMI category (overweight or obesity) is one of the most established risk factors for gestational diabetes: excess adipose tissue is associated with increased insulin resistance. Note: Section 7.7 of this thesis documents very low model specificity for obesity categories — this factor should be interpreted with caution.",
    'num__imc_ordinal_bas': "A lower BMI category is usually reassuring from a metabolic standpoint. Note: Section 7.7 documents reduced model sensitivity for normal BMI — a low-risk result should be interpreted with heightened clinical caution for this profile.",
    'num__tension_ordinal_haut': "An elevated or hypertensive blood pressure profile may accompany metabolic changes shared with gestational diabetes.",
    'num__tension_ordinal_bas': "A normal blood pressure profile is a reassuring factor, with no expected association with increased metabolic risk.",
    'num__sa_premiere_consult_haut': "A late first antenatal visit limits the data available at the time of early screening — Section 7.2.3 of this thesis documents a notable proportion of late visits in the Cameroonian context.",
    'num__sa_premiere_consult_bas': "An early first visit allows evaluation under conditions close to those of the training set (France, average visit at 9.5 weeks).",
    'cat__parite_Nullipare': "Nulliparity (first pregnancy) is not in itself an established risk factor for gestational diabetes, unlike high multiparity.",
    'cat__parite_Primipare': "Primiparity is associated with a reference risk profile in the clinical literature.",
    'cat__parite_Multipare_2': "A parity of rank 2 may be associated with a slightly increased risk, related to cumulative age and metabolic history of previous pregnancies.",
    'cat__parite_Multipare_3': "High multiparity (3 or more pregnancies) is associated in the literature with a slightly increased risk of gestational diabetes.",
    'cat__atcd_familial_diabete_1er_deg_Oui': "A first-degree family history of diabetes (parent, sibling) is a recognized and well-documented genetic risk factor for gestational diabetes.",
    'cat__atcd_familial_diabete_1er_deg_Non': "The absence of a family history of diabetes is a reassuring factor that reduces the estimated probability.",
    'cat__atcd_gdm_Oui': "A personal history of gestational diabetes in a previous pregnancy is one of the most robust predictors of recurrence documented in the clinical literature.",
    'cat__atcd_gdm_Non': "The absence of a personal history of gestational diabetes is an important reassuring factor.",
    'cat__atcd_macrosomie_Oui': "A history of fetal macrosomia (newborn over 4kg) can be an indirect sign of undiagnosed gestational diabetes in a previous pregnancy.",
    'cat__atcd_macrosomie_Non': "The absence of a history of macrosomia is a reassuring factor.",
    'cat__sopk_Oui': "Polycystic ovary syndrome (PCOS) is associated with insulin resistance that may favor the development of gestational diabetes.",
    'cat__sopk_Non': "The absence of PCOS is a reassuring metabolic factor.",
    'cat__sedentarite_Oui': "A sedentary lifestyle (less than 150 minutes of weekly physical activity) is a modifiable risk factor associated with lower insulin sensitivity.",
    'cat__sedentarite_Non': "Sufficient physical activity is a recognized protective factor against metabolic risk.",
    'cat__tabagisme_Oui': "Smoking may interact with glucose metabolism, although its association with gestational diabetes specifically is less established than for other pregnancy complications.",
    'cat__tabagisme_Non': "The absence of smoking is favorable to the overall metabolic profile.",
    'cat__hta_ou_preeclampsie_Oui': "A history of hypertension or pre-eclampsia shares vascular and metabolic pathophysiological mechanisms with gestational diabetes.",
    'cat__hta_ou_preeclampsie_Non': "The absence of a history of hypertension or pre-eclampsia is a reassuring factor.",
}


def compute_raw_contributions(X_transformed_row, top_n=4):
    """Calcule les contributions BRUTES (langue-indépendantes) : juste la clé technique
    de la variable et sa valeur de contribution. Ne contient AUCUN texte traduit — c'est
    ce qui doit être stocké en base, pour permettre un ré-affichage dans n'importe quelle
    langue plus tard (voir enrich_contributions)."""
    if not hasattr(model, 'coef_'):
        return []
    feature_names = preprocessor.get_feature_names_out()
    coefs = model.coef_[0]
    contributions = coefs * X_transformed_row
    items = []
    for name, contrib in zip(feature_names, contributions):
        if abs(contrib) < 1e-6:
            continue
        items.append({'feature_key': name, 'contribution': float(contrib)})
    items.sort(key=lambda x: abs(x['contribution']), reverse=True)
    return items[:top_n]


def enrich_contributions(raw_contributions, lang='fr'):
    """Traduit des contributions BRUTES (feature_key + valeur) en éléments affichables
    (label, direction, explication détaillée) dans la langue demandée. Appelée à chaque
    AFFICHAGE (pas au moment du calcul) pour que les résultats passés se retraduisent
    correctement même après un changement de langue."""
    label_map = LABEL_MAP_EN if lang == 'en' else LABEL_MAP
    explanations = DETAILED_EXPLANATIONS_EN if lang == 'en' else DETAILED_EXPLANATIONS
    items = []
    for raw in raw_contributions:
        name = raw['feature_key']
        contrib = raw['contribution']
        explain_key = name
        if name.startswith('num__'):
            explain_key = f"{name}_haut" if contrib > 0 else f"{name}_bas"
        detail = explanations.get(explain_key)
        items.append({'label': label_map.get(name, name), 'contribution': contrib,
                       'direction': 'augmente' if contrib > 0 else 'diminue',
                       'detail': detail})
    return items


def generate_recommendation(risque_eleve, contributions, equity_caveats, lang='fr'):
    """

    Génère une recommandation déterministe (basée sur règles, PAS de LLM — décision actée
    après discussion des risques : dépendance Internet contraire au principe offline
    Section 5.2.3/Section 6, non-validation clinique d'un LLM génératif, dilution de
    l'argument concurrentiel vs Delfina Care Section 4.2).
    """
    if lang == 'en':
        base = ("Priority OGTT recommended — high-risk profile for gestational diabetes."
                if risque_eleve else
                "Routine follow-up recommended. OGTT if otherwise clinically indicated.")
        facteur_texte = ""
        if contributions:
            principal = contributions[0]
            verbe = "mainly due to" if principal['direction'] == 'augmente' else "despite"
            facteur_texte = f" This result is explained {verbe} {principal['label']}."
        equite_texte = ""
        if equity_caveats:
            equite_texte = " ⚠️ See the reliability warning below before acting on this result."
        return base + facteur_texte + equite_texte

    if risque_eleve:
        base = "OGTT prioritaire recommandé — profil à risque élevé de diabète gestationnel."
    else:
        base = "Suivi de routine recommandé. OGTT si indication clinique par ailleurs."

    facteur_texte = ""
    if contributions:
        principal = contributions[0]
        verbe = "principalement en raison de" if principal['direction'] == 'augmente' else "malgré"
        facteur_texte = f" Ce résultat s'explique {verbe} {principal['label']}."

    equite_texte = ""
    if equity_caveats:
        equite_texte = " ⚠️ Voir l'avertissement de fiabilité ci-dessous avant d'agir sur ce résultat."

    return base + facteur_texte + equite_texte


def get_equity_caveat(imc_ordinal, age, lang='fr'):
    caveats = []
    if lang == 'en':
        try:
            imc = int(imc_ordinal)
            if imc == 2:
                caveats.append("For women with normal BMI, the model's measured sensitivity "
                                "is lower (26%, Section 7.7): a 'low risk' result should be "
                                "interpreted with heightened clinical caution.")
            elif imc in (4, 5):
                caveats.append("For women with obesity, the model's measured specificity "
                                "is very low (Section 7.7): the model over-flags this profile. "
                                "A 'high risk' result should not replace overall clinical assessment.")
        except (TypeError, ValueError):
            pass
        try:
            age_val = float(age)
            if age_val < 25:
                caveats.append("For women under 25, the model's measured sensitivity is "
                                "reduced (54%, Section 7.7).")
            elif age_val > 35:
                caveats.append("For women over 35, the model's measured specificity is "
                                "very low (0%, Section 7.7): the model systematically over-flags this profile.")
        except (TypeError, ValueError):
            pass
        return caveats

    try:
        imc = int(imc_ordinal)
        if imc == 2:
            caveats.append("Chez les femmes d'IMC normal, la sensibilité mesurée du modèle "
                            "est plus faible (26 %, Section 7.7) : un résultat 'bas risque' doit "
                            "être interprété avec prudence clinique renforcée.")
        elif imc in (4, 5):
            caveats.append("Chez les femmes en situation d'obésité, la spécificité mesurée du "
                            "modèle est très faible (Section 7.7) : le modèle sur-signale ce "
                            "profil. Un résultat 'haut risque' ne doit pas se substituer à "
                            "l'évaluation clinique globale.")
    except (TypeError, ValueError):
        pass
    try:
        age_val = float(age)
        if age_val < 25:
            caveats.append("Chez les femmes de moins de 25 ans, la sensibilité mesurée du "
                            "modèle est réduite (54 %, Section 7.7).")
        elif age_val > 35:
            caveats.append("Chez les femmes de plus de 35 ans, la spécificité mesurée du "
                            "modèle est très faible (0 %, Section 7.7) : le modèle sur-signale "
                            "systématiquement ce profil.")
    except (TypeError, ValueError):
        pass
    return caveats


# ============================================================
# ROUTES — AUTHENTIFICATION
# ============================================================

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form.get('username', '').strip()
        password = request.form.get('password', '')
        conn = get_db()
        user = conn.execute('SELECT * FROM users WHERE username = ?', (username,)).fetchone()
        conn.close()
        if user and check_password_hash(user['password_hash'], password):
            session['user_id'] = user['id']
            session['nom_complet'] = user['nom_complet']
            session['role'] = user['role']
            session['photo_path'] = user['photo_path']
            session['is_admin'] = bool(user['is_admin'])
            return redirect(url_for('home'))
        return render_template('login.html', error="Identifiants incorrects.")
    return render_template('login.html', error=None)


@app.route('/logout')
def logout():
    session.clear()
    return redirect(url_for('login'))


# ============================================================
# ROUTES — PAGES PRINCIPALES
# ============================================================

@app.route('/')
@login_required
def home():
    conn = get_db()
    n_predictions = conn.execute('SELECT COUNT(*) as n FROM predictions WHERE user_id = ?',
                                  (session['user_id'],)).fetchone()['n']
    conn.close()
    return render_template('home.html', validation_info=VALIDATION_INFO, n_predictions=n_predictions)


@app.route('/predire')
@login_required
def predire_page():
    lang = current_lang()
    features = FEATURES_INFO_EN if lang == 'en' else FEATURES_INFO
    sections = SECTIONS_EN if lang == 'en' else SECTIONS
    return render_template('predict.html', features=features, sections=sections,
                            demo_patients=get_demo_patients(lang), validation_info=VALIDATION_INFO,
                            seuil=SEUIL_DECISION)


@app.route('/predict', methods=['POST'])
@login_required
def predict():
    lang = current_lang()
    if model is None:
        error_msg = "Model not loaded. Run notebooks 01 to 05 first." if lang == 'en' else \
                    "Modèle non chargé. Exécuter les notebooks 01 à 05 d'abord."
        return jsonify({'error': error_msg}), 503

    data = request.get_json()
    input_dict = {}
    for feat in FEATURES_INFO.keys():
        val = data.get(feat, None)
        input_dict[feat] = [val if val not in ('', None) else np.nan]
    X_input = pd.DataFrame(input_dict)

    try:
        X_transformed = preprocessor.transform(X_input)
        proba = model.predict_proba(X_transformed)[0, 1]
        risque_eleve = proba >= SEUIL_DECISION

        # 🔒 IMPORTANT : on ne stocke QUE des données brutes langue-indépendantes
        # (clé technique de variable + valeur numérique). Le texte affiché (labels,
        # explications, recommandation, avertissements) est généré à la VOLÉE au moment
        # de l'affichage (route /resultat), dans la langue active à CE moment-là — pas
        # celle active au moment du calcul. Ça permet de rouvrir un résultat ancien et
        # de le voir correctement traduit même après un changement de langue.
        raw_contributions = compute_raw_contributions(X_transformed[0])

        conn = get_db()
        cursor = conn.execute(
            'INSERT INTO predictions (user_id, timestamp, probabilite, classification, age_maternel, '
            'imc_ordinal, contributions_json, seuil_utilise) VALUES (?,?,?,?,?,?,?,?)',
            (session['user_id'], datetime.now().isoformat(), float(proba),
             'Haut risque' if risque_eleve else 'Bas risque',
             data.get('age_maternel'), data.get('imc_ordinal'),
             json.dumps(raw_contributions), float(SEUIL_DECISION))
        )
        conn.commit()
        prediction_id = cursor.lastrowid
        conn.close()

        return jsonify({'prediction_id': prediction_id})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/resultat/<int:pred_id>')
@login_required
def resultat(pred_id):
    lang = current_lang()
    conn = get_db()
    row = conn.execute('SELECT * FROM predictions WHERE id = ? AND user_id = ?',
                        (pred_id, session['user_id'])).fetchone()
    conn.close()
    if row is None:
        return redirect(url_for('historique'))

    pred = dict(row)

    # 🔒 Régénération complète du texte affiché dans la langue ACTIVE au moment de la
    # consultation (pas celle active au moment du calcul) — à partir des seules données
    # brutes stockées (contributions_json = clés techniques, age_maternel, imc_ordinal).
    raw_contributions = json.loads(pred['contributions_json']) if pred['contributions_json'] else []
    pred['contributions'] = enrich_contributions(raw_contributions, lang=lang)

    pred['equity_caveats'] = get_equity_caveat(pred['imc_ordinal'], pred['age_maternel'], lang=lang)

    imc_fairness_table = IMC_FAIRNESS_EN if lang == 'en' else IMC_FAIRNESS
    pred['fiabilite_imc'] = (imc_fairness_table.get(int(pred['imc_ordinal']))
                              if pred['imc_ordinal'] not in (None, '') else None)
    pred['fiabilite_age'] = get_age_fairness(pred['age_maternel'], lang=lang)

    risque_eleve = pred['classification'] == 'Haut risque'
    pred['recommandation'] = generate_recommendation(risque_eleve, pred['contributions'],
                                                       pred['equity_caveats'], lang=lang)

    return render_template('resultat.html', pred=pred)


@app.route('/historique')
@login_required
def historique():
    conn = get_db()
    rows = conn.execute('SELECT * FROM predictions WHERE user_id = ? ORDER BY timestamp DESC',
                         (session['user_id'],)).fetchall()
    conn.close()
    predictions = [dict(r) for r in rows]

    n_haut = sum(1 for p in predictions if p['classification'] == 'Haut risque')
    n_bas = len(predictions) - n_haut

    return render_template('historique.html', predictions=predictions, n_haut=n_haut, n_bas=n_bas)


@app.route('/admin')
@admin_required
def admin_panel():
    conn = get_db()
    users = conn.execute('''
        SELECT u.id, u.username, u.nom_complet, u.role, u.is_admin,
               COUNT(p.id) as n_predictions,
               MAX(p.timestamp) as derniere_prediction
        FROM users u
        LEFT JOIN predictions p ON p.user_id = u.id
        GROUP BY u.id
        ORDER BY u.id
    ''').fetchall()

    stats = conn.execute('''
        SELECT COUNT(*) as total,
               SUM(CASE WHEN classification='Haut risque' THEN 1 ELSE 0 END) as n_haut
        FROM predictions
    ''').fetchone()
    conn.close()

    return render_template('admin.html', users=users, stats=stats,
                            db_path=os.path.abspath(DB_PATH))


@app.route('/parametres')
@login_required
def parametres():
    return render_template('parametres.html', validation_info=VALIDATION_INFO, seuil=SEUIL_DECISION,
                            model_type='Régression Logistique (class_weight=balanced)',
                            n_features=12)


@app.route('/parametres/photo', methods=['POST'])
@login_required
def upload_photo():
    file = request.files.get('photo')
    if file and file.filename and '.' in file.filename and \
       file.filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS:
        os.makedirs(UPLOAD_FOLDER, exist_ok=True)
        ext = file.filename.rsplit('.', 1)[1].lower()
        filename = secure_filename(f"user_{session['user_id']}.{ext}")
        filepath = os.path.join(UPLOAD_FOLDER, filename)
        file.save(filepath)
        photo_url = '/' + filepath.replace('\\', '/')

        conn = get_db()
        conn.execute('UPDATE users SET photo_path = ? WHERE id = ?', (photo_url, session['user_id']))
        conn.commit()
        conn.close()
        session['photo_path'] = photo_url
    return redirect(url_for('parametres'))


@app.route('/health')
def health():
    return jsonify({'status': 'ok', 'model_loaded': model is not None,
                     'seuil': SEUIL_DECISION, 'validation_info': VALIDATION_INFO})


# Initialisation exécutée au chargement du module (pas seulement en lancement direct)
# — indispensable pour un déploiement distant via Gunicorn (gunicorn app:app), qui importe
# ce fichier sans jamais passer par le bloc if __name__ == '__main__'.
try:
    init_db()
    print("✅ init_db() terminé sans exception.", flush=True)
except Exception as e:
    import traceback
    print("❌❌❌ ERREUR CRITIQUE dans init_db() :", str(e), flush=True)
    traceback.print_exc()

# Vérification explicite post-init : la table users existe-t-elle vraiment ?
try:
    _check_conn = get_db()
    _check = _check_conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()
    print(f"📋 Tables présentes dans {DB_PATH} après init_db() : {[r['name'] for r in _check]}", flush=True)
    _check_conn.close()
except Exception as e:
    print("❌ Impossible de vérifier les tables :", str(e), flush=True)

_model_loaded = load_model()
load_seuil_officiel()
if not _model_loaded:
    print("⚠️ ATTENTION: models/model_A.pkl introuvable.")
    print("   Exécuter les notebooks 01 à 05 avant de lancer l'application.")
else:
    print("✅ Modèle chargé avec succès.")
print("\n🔑 Comptes de démonstration : dr.fotso / sage.femme — mot de passe : CHU2026!")


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
