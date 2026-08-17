# 🏥 GDM Validation Project

**Validation externe rigoureuse de modèles IA entraînés sur données synthétiques : Application à la prédiction du diabète gestationnel au CHU de Yaoundé**

Thèse Mastère DIA — Nexa Digital School
Auteur : Darryl MOMO | Contexte clinique : David Ben Zaza (Université de Yaoundé I)

---

## 🚀 Application en ligne

**[👉 gdm-validation-cameroun.onrender.com](https://gdm-validation-cameroun.onrender.com)**

Comptes de démonstration :

| Utilisateur | Mot de passe | Rôle |
|---|---|---|
| `dr.fotso` | `CHU2026!` | Gynécologue-Obstétricien |
| `sage.femme` | `CHU2026!` | Sage-femme |

> ⚠️ Hébergement sur tier gratuit Render : l'application peut mettre jusqu'à 50 secondes à se réveiller après une période d'inactivité (comportement normal, pas un bug).

---

## 📋 À propos

Ce projet valide, sur 455 cas réels de patientes camerounaises (CHU Yaoundé et HOGOPY), un modèle d'apprentissage supervisé entraîné exclusivement sur un jeu de données synthétique français (30 000 femmes). L'objectif : déterminer si un modèle IA entraîné sur des données idéales peut prédire de façon fiable le diabète gestationnel lorsqu'il est confronté à des données réelles, partielles et hétérogènes d'un contexte africain à ressources limitées.

Le dépôt contient l'ensemble du pipeline (nettoyage des données, entraînement, validation externe, analyse de robustesse et d'équité) ainsi qu'une application web fonctionnelle (Flask) intégrant le modèle retenu, déployée en accès public pour démonstration.

---

## 🎯 Principe méthodologique fondamental

> **Entraînement** exclusivement sur `dataset_dg_france_30000_final.csv` (synthétique, France)
> **Validation** exclusivement sur les 455 cas réels camerounais (243 positifs + 212 négatifs)
> **Aucune donnée camerounaise n'entre dans l'entraînement.** C'est de la validation externe pure.

---

## 📁 Structure du projet

```
GDM_VALIDATION_PROJECT/
├── data/
│   ├── raw/                  ← CSV originaux (NE JAMAIS MODIFIER)
│   └── processed/            ← Générés par les notebooks 02-03
├── notebooks/                ← 8 notebooks séquentiels (voir ci-dessous)
├── src/                      ← Fonctions réutilisables (imports dans notebooks)
├── models/                   ← Modèles entraînés sauvegardés (.pkl)
├── figures/                  ← Figures exportées (pour la thèse)
├── results/                  ← Tableaux de résultats (.csv, .json)
├── app/                      ← Application IA Flask (Section 8)
├── requirements.txt
└── README.md                 ← Ce fichier
```

## 📓 Ordre d'exécution des notebooks

| # | Notebook | Objectif | Sortie |
|---|----------|----------|--------|
| 01 | `01_EDA_exploration.ipynb` | Exploration des 3 datasets bruts | Figures EDA |
| 02 | `02_data_cleaning.ipynb` | Nettoyage, harmonisation, binning IMC/tension | `data/processed/*.csv` |
| 03 | `03_feature_engineering.ipynb` | Pipeline preprocessing (fit sur France, transform sur Cameroun) | `models/preprocessing_pipeline.pkl` |
| 04 | `04_model_training.ipynb` | Entraînement Modèle A (11 var.) + Modèle B (+ niveau_etude) sur France | `models/model_A.pkl`, `models/model_B.pkl` |
| 05 | `05_external_validation.ipynb` | Validation sur Cameroun (455 cas) — sensibilité, spécificité, AUC | `results/validation_metrics.csv` |
| 06 | `06_robustness_analysis.ipynb` | Robustesse données manquantes + test `flag_source_batch` | `results/robustness_results.csv` |
| 07 | `07_fairness_interpretability.ipynb` | Équité par groupe démographique + SHAP | `figures/shap_*.png` |
| 08 | `08_results_summary.ipynb` | Synthèse finale — tableaux/figures pour Section 7 thèse | `results/summary_thesis.xlsx` |

**⚠️ Exécuter dans l'ordre.** Chaque notebook dépend des sorties du précédent (`data/processed/`, `models/`).

---

## 🖥️ Installation — Windows + VSCode

### 1. Prérequis
- Python 3.10 ou 3.11 installé ([python.org](https://www.python.org/downloads/))
- VSCode installé avec l'extension **Jupyter** et **Python** (Microsoft)

### 2. Ouvrir le projet
```powershell
cd C:\Users\<TonNom>\Documents
# Dézippe GDM_VALIDATION_PROJECT.zip ici, puis :
cd GDM_VALIDATION_PROJECT
code .
```

### 3. Créer l'environnement virtuel
Dans le terminal VSCode (PowerShell) :
```powershell
python -m venv gdm_env
gdm_env\Scripts\activate
```
> Si erreur "impossible de charger le fichier script" → lancer PowerShell en admin et exécuter :
> `Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser`

### 4. Installer les dépendances
```powershell
pip install --upgrade pip
pip install -r requirements.txt
```

### 5. Enregistrer le kernel Jupyter
```powershell
python -m ipykernel install --user --name gdm_env --display-name "GDM Validation"
```

### 6. Ouvrir un notebook
- Dans VSCode, ouvrir `notebooks/01_EDA_exploration.ipynb`
- En haut à droite : sélectionner le kernel **"GDM Validation"**
- Exécuter les cellules avec `Shift+Enter`

---

## 🔒 Règles de nettoyage des données (verrouillées après discussion)

| Sujet | Décision |
|---|---|
| IMC | Binning OMS ordinal (PAS de conversion midpoint — fausse précision) |
| Tension artérielle | Binning ordinal 3 niveaux (Normal/Élevé/Hypertension) |
| HTA + pré-éclampsie | Variable combinée unique |
| Glycémie à jeun, taille, poids, alcool, grossesse multiple | Exclues (absentes/quasi-100% manquantes côté Cameroun) |
| sopk, sédentarité, tabagisme, hta_preeclampsie | Gardées + `flag_source_batch` en test de sensibilité interne uniquement |
| Niveau d'étude | Modèle B seulement (ablation study) |
| Zone de résidence | Exclue (trop de manquants) |
| OGTT | Exclu des features (analyse robustesse séparée, notebook 06) |

**Découverte critique (notebook 02) :** les 212 cas négatifs contiennent 2 sous-cohortes fusionnées (108 premières lignes = 5 variables systématiquement "Non_renseigne"). Documenté et testé pour risque de fuite d'information.

---

## 📊 KPIs cibles (Section 5 de la thèse)

| KPI | Seuil |
|---|---|
| Sensibilité | ≥ 85% |
| Spécificité | ≥ 75% |
| AUC (ROC) | ≥ 0.85 |
| Dégradation performance avec 65% OGTT manquant | < 10% |
| Écart AUC inter-groupe démographique | < 5% |
| Temps de prédiction (app) | < 1 seconde |

---

## 📞 Contact

- **Lead Technique Solution IA :** Darryl MOMO
- **Contexte clinique & données :** David Ben Zaza (Université de Yaoundé I)
