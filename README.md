# Store Sales Forecasting

Prévision des ventes quotidiennes pour la compétition Kaggle [Store Sales - Time Series Forecasting](https://www.kaggle.com/competitions/store-sales-time-series-forecasting).

Ce projet compare une régression directe à un modèle en deux étapes et combine leurs prédictions pour produire une soumission Kaggle. Le notebook présenté est la version portfolio ; les datasets et les fichiers de soumission ne sont pas inclus.

## Résultats

La métrique de la compétition est le **RMSLE** (plus bas est meilleur).

| Modèle | RMSLE OOF temporel | Score public Kaggle |
|---|---:|---:|
| LightGBM direct, CPU | 0,412539 | — |
| XGBoost en deux étapes, GPU | 0,409131 | — |
| Mélange 25 % LightGBM / 75 % XGBoost | 0,407937 | **0,40665** |
| Mélange candidat 35 % LightGBM / 65 % XGBoost | **0,407842** | Non soumis |

La soumission 25/75 a également atteint le **104e rang sur 1 520 participants** au classement Kaggle (environ le top **6,8 %**), selon le résultat rapporté. Le score public de **0,40665** est celui de cette soumission ; il s'agit d'un score de leaderboard public, pas d'une mesure indépendante. Le meilleur résultat OOF du candidat 65/35 est très proche et a été obtenu en comparant les mêmes fenêtres : il n'est donc pas présenté comme un gain confirmé.

## Approche

- Transformation de la cible avec `log1p(sales)` ; évaluation en RMSLE.
- Variables calendaires, informations magasins et jours fériés, transactions, promotions, lags et statistiques glissantes.
- Validation chronologique sur sept fenêtres non chevauchantes ; aucune validation aléatoire des dates.
  1. classifier la probabilité que les ventes soient positives ;
  2. estimer les ventes logarithmiques conditionnellement aux ventes positives.
- Mélange des sorties des deux modèles dans l'espace logarithmique, avec un poids de 25 % LightGBM et 75 % XGBoost.

Les fenêtres historiques couvrent 2016-10 à 2017-08. Le dernier fold contient 15 jours étiquetés. Les itérations et poids ont été choisis à partir d'expériences antérieures ; les fenêtres ne constituent pas un holdout indépendant.

## Fichiers

```text
.
├── README.md
├── requirements.txt
└── store_sales_forecasting.ipynb
```

## Exécution sur Kaggle

1. Ouvrez le notebook dans Kaggle.
2. Ajoutez le dataset de la compétition comme entrée du notebook.
3. Dans **Settings → Accelerator**, activez l'accélérateur graphique.
4. Exécuteé toutes les cellules dans l'ordre.

Le notebook vérifie que la version installée de XGBoost est compilée avec CUDA. L'entraînement XGBoost utilise le GPU à la fois ; le modèle LightGBM reste sur le CPU.

## Exécution locale

Python 3.11 est recommandé. Installez les dépendances :

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

Placez les six fichiers de données fournis par Kaggle dans `data/` :

```text
data/
├── train.csv
├── test.csv
├── oil.csv
├── stores.csv
├── holidays_events.csv
└── transactions.csv
```

Puis lancez Jupyter et exécutez les cellules dans l'ordre. L'entraînement XGBoost requiert une installation compatible CUDA et un GPU disponible ; l'environnement local CPU seul ne suffit pas pour le pipeline complet tel qu'il est configuré, une modification du notebook est requise

## Données et licence

Les données de la compétition ne sont pas redistribuées dans ce dépôt. Elles sont téléchargeables depuis la [page Kaggle de la compétition](https://www.kaggle.com/competitions/store-sales-time-series-forecasting/data) et sous le respecte des règles d'utilisation. Aucune licence de réutilisation du code n'est définie pour le moment.
