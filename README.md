# Détection de fraude sur transactions par carte bancaire

## 1. Contexte

Ce projet porte sur la détection de transactions frauduleuses à partir d’un jeu de données simulé issu de Kaggle.

Le dataset contient environ 1,85 million de transactions réalisées entre 2019 et 2020, avec une forte asymétrie : une majorité écrasante de transactions légitimes et un taux de fraude d’environ 0,05 %.

Malgré ce faible taux, les enjeux restent importants : pertes financières, coûts opérationnels et dégradation de l’expérience client. L’objectif est donc de détecter efficacement les fraudes tout en limitant les faux positifs.

## 2. Objectif

Mettre en place un pipeline complet de détection de fraude couvrant l’ensemble des étapes : préparation des données, analyse exploratoire, modélisation, interprétation et mise à disposition via une application.

L’objectif n’est pas uniquement d’obtenir de bonnes performances, mais de trouver un compromis réaliste entre détection des fraudes, coûts opérationnels et expérience client.

## 3. Données

Source : Kaggle  
https://www.kaggle.com/datasets/kartik2112/fraud-detection

Le dataset contient des transactions simulées mais construites pour reproduire des comportements réalistes.

Caractéristiques principales :
- données transactionnelles bancaires  
- période 2019–2020  
- classification binaire (fraude vs non fraude)  
- très fort déséquilibre  

## 4. Méthodologie

### Préparation des données

- fusion des fichiers train et test  
- nettoyage et renommage des variables  
- suppression des variables inutiles ou sensibles  
- création de variables (temps, âge, distance)

### Analyse exploratoire

Objectifs : comprendre les données, identifier les variables pertinentes et vérifier la stabilité temporelle.

Principaux résultats :
- taux de fraude très faible (~0,05 %)  
- variables les plus discriminantes : montant, heure, type de magasin  
- suppression des variables peu informatives ou trop bruitées  

### Stratégie de validation

Découpage temporel :
- train : 2019  
- validation : janvier à septembre 2020  
- test : octobre à décembre 2020  

Ce choix permet de simuler un cadre réel de détection.

### Modélisation

Pipeline basé sur :

- encodage des variables catégorielles  
- ajout d’un score d’anomalie via Isolation Forest (`if_score`)  
- modèle principal : XGBoost  
- tuning des hyperparamètres  

Le score issu de l’Isolation Forest permet de capter le caractère atypique d’une transaction. Dans un contexte fortement déséquilibré, ce signal non supervisé est particulièrement pertinent pour détecter des comportements rares.

XGBoost est utilisé pour sa capacité à capturer des relations non linéaires et des interactions complexes.

Les hyperparamètres sont optimisés afin d’obtenir un bon compromis entre performance et stabilité des probabilités, avec un focus sur la qualité des scores produits.

La validation repose sur un découpage temporel de type TimeSeriesSplit, ce qui permet d’éviter toute fuite d’information et de rester cohérent avec un contexte réel.

### Calibration et seuil

Deux éléments structurants du pipeline :

**Calibration des probabilités**

Une étape de calibration est appliquée afin d’obtenir des probabilités cohérentes et interprétables. L’objectif est que les scores reflètent correctement le risque réel, ce qui est essentiel dans une logique de prise de décision.

**Optimisation du seuil de décision**

Le seuil est optimisé selon une fonction de coût métier :

- faux positif : 25 €  
- faux négatif : 125 €  

Une fraude non détectée étant 5 fois plus coûteuse, le modèle est orienté vers un bon rappel.

Le seuil est déterminé sur la validation en minimisant le coût total, puis appliqué au test.

## 5. Résultats

Le modèle présente des performances élevées, à la fois en termes de métriques statistiques et d’impact opérationnel :

- ROC-AUC ≈ 0,99  
- Recall ≈ 78,5 % (≈ 8 fraudes sur 10 détectées)  
- Précision ≈ 51 % (≈ 1 alerte sur 2 pertinente)  
- Alert rate ≈ 0,51 % des transactions  

Ces résultats traduisent un bon compromis entre détection et charge opérationnelle.

![Precision-Recall curve](./05_visualisations/courbe_pr.png)

![Calibration curve](./05_visualisations/courbe_calibration.png)

Un drift du taux de fraude est observé entre validation et test, impactant la précision et nécessitant un ajustement du seuil.

## 6. Interprétabilité du modèle (SHAP)

Le modèle est analysé avec SHAP afin de comprendre ses décisions.

### Importance globale

Variables principales :
- montant  
- heure  
- score d’anomalie  
- type de magasin  

![Feature importance](./05_visualisations/feature_importance.png)

### Effets moyens

- montants atypiques plus risqués  
- certaines heures plus exposées  
- transactions anormales plus suspectes  

![Beeswarm](./05_visualisations/beeswarm_plot.png)

### Explications locales

Chaque décision est interprétable :

Transaction non frauduleuse :
- comportement cohérent  
- faible probabilité  

![Non fraude](./05_visualisations/fraud_target_0.png)

Transaction frauduleuse :
- comportement atypique  
- signaux de risque cumulés  

![Fraude](./05_visualisations/fraud_target_1.png)

### Interactions

Le modèle combine plusieurs variables pour estimer le risque.

![Dependence plot](./05_visualisations/dependence_plot.png)

## 7. Pipeline

Le projet est structuré sous forme de pipeline exécutable :

full_fraud_detection_projet()

Il enchaîne :
- feature engineering  
- nettoyage post-EDA  
- modélisation complète  

Cette organisation garantit la reproductibilité et s’inscrit dans une logique proche des pratiques MLOps.

## 8. Structure du projet

- 01_data : données brutes et transformées  
- 02_notebooks : analyse complète  
- 03_fonctions : fonctions réutilisables  
- 04_model : modèles et artefacts  
- 05_visualisations : graphiques  
- 06_streamlit : application  
- full_pipeline.py : orchestration  

Cette structure sépare clairement exploration et production.

## 9. Application Streamlit

Une application interactive permet de rendre le modèle exploitable.

Fonctionnalités :
- visualisation des performances  
- analyse du coût  
- explicabilité (SHAP)  
- simulation de transactions  

Une brique d’IA générative (LLM Sonar – Perplexity) est intégrée pour traduire les explications en langage naturel.

L’application transforme le modèle en outil d’aide à la décision.

## 10. Conclusion

Le modèle présente une forte capacité de discrimination, une bonne cohérence probabiliste et un compromis pertinent entre détection et coût.

Il est exploitable en contexte réel, avec ajustement du seuil.

Ce projet illustre le passage d’une approche exploratoire à une solution structurée proche des pratiques industrielles.