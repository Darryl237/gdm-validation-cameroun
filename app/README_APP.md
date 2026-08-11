# Application IA v3 — Prédiction Risque DG

## Prérequis
Exécuter d'abord les notebooks `01` à `05` (génère `models/model_A.pkl`, `models/preprocessor_A.pkl`,
`results/validation_metrics.json`).

## Lancement local (Windows)
```powershell
cd app
..\gdm_env\Scripts\activate
python app.py
```

Au premier lancement, une base SQLite locale (`gdm_app.db`) est créée automatiquement avec
2 comptes de démonstration :

| Utilisateur | Mot de passe | Rôle |
|---|---|---|
| `dr.fotso` | `CHU2026!` | Gynécologue-Obstétricien |
| `sage.femme` | `CHU2026!` | Sage-femme |

Ouvrir : http://localhost:5000 — redirige automatiquement vers la page de connexion.

## Structure de l'application (v3)
- `/login` — authentification (comptes pré-configurés, mots de passe hashés)
- `/` — tableau de bord (métriques de validation, accès rapide)
- `/predire` — formulaire de prédiction (3 sections, profils de démo, résultat avec jauge,
  facteurs contributifs, avertissements d'équité)
- `/historique` — historique des prédictions de l'utilisateur connecté, avec graphiques SVG natifs
- `/parametres` — informations modèle en lecture seule + infos compte

## ⚠️ Avant tout déploiement réel (hors périmètre de cette thèse)
1. Remplacer `app.secret_key` (actuellement une valeur de développement) par une vraie clé secrète
2. Remplacer les comptes de démonstration par un vrai processus de création de comptes encadré
   par le service informatique du CHU
3. Ajouter HTTPS (actuellement HTTP local uniquement)
4. Tester le temps de réponse sur le matériel cible (KPI < 1 seconde, Section 5)

## 🌐 Bilingue FR/EN (v3.4)
L'application est désormais entièrement bilingue français/anglais :
- Sélecteur "🌐" dans la barre de navigation (et sur la page de connexion)
- Couvre : navigation, formulaire, recommandations, les 26 explications cliniques détaillées,
  avertissements d'équité, historique, paramètres
- Système léger (dictionnaires Python, pas de dépendance externe type Flask-Babel) — cohérent
  avec le principe d'infrastructure légère (Section 5.2.3)
- La langue est mémorisée en session ; chaque prédiction est recalculée dans la langue active
  au moment du calcul (stockée telle quelle en base pour cohérence historique)
