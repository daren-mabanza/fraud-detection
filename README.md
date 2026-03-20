# Détection de fraude sur transactions par carte bancaire

## 1. Contexte

Ce projet porte sur la détection de **transactions frauduleuses** à partir d’un jeu de données simulé issu de Kaggle.

Le dataset contient environ **1,85 million de transactions** réalisées entre 2019 et 2020, avec une forte asymétrie : une majorité écrasante de transactions légitimes et un taux de fraude d’environ **0,05 %**.

Malgré ce faible taux, les enjeux restent importants : **pertes financières**, **coûts opérationnels** et **dégradation de l’expérience client**. L’objectif est donc de détecter efficacement les fraudes tout en limitant les faux positifs.

## 2. Objectif

Mettre en place un **pipeline complet de détection de fraude** couvrant l’ensemble des étapes : préparation des données, analyse exploratoire, modélisation, interprétation et mise à disposition via une application.

Au-delà de la performance pure du modèle, l’objectif est de construire une solution **exploitable en contexte réel**, capable de s’intégrer dans un processus opérationnel.

Cela implique de trouver un compromis réaliste entre :
- la capacité à détecter un maximum de fraudes  
- la limitation des faux positifs pour éviter une surcharge opérationnelle  
- la préservation de l’expérience client  

Le projet s’inscrit ainsi dans une logique **orientée métier**, où les décisions du modèle sont évaluées à la fois sur des métriques statistiques et leur **impact économique**.

## 3. Données

Source : Kaggle  
https://www.kaggle.com/datasets/kartik2112/fraud-detection

Le dataset contient des transactions simulées mais construites pour reproduire des comportements réalistes.

Caractéristiques principales :
- données transactionnelles bancaires  
- période 2019–2020  
- classification binaire (**fraude vs non fraude**)  
- **très fort déséquilibre**  

## 4. Méthodologie

### Préparation des données

- fusion des fichiers train et test  
- nettoyage et renommage des variables  
- suppression des variables inutiles ou sensibles  
- création de variables (temps, âge, distance)

### Analyse exploratoire

Objectifs : comprendre les données, identifier les variables pertinentes et vérifier la **stabilité temporelle**.

Principaux résultats :
- taux de fraude très faible (~0,05 %)  
- variables les plus discriminantes : **montant**, **heure**, **type de magasin**  
- suppression des variables peu informatives ou trop bruitées  

### Stratégie de validation

Découpage temporel :

- train : 2019  
- validation : janvier à septembre 2020  
- test : octobre à décembre 2020  

Ce choix permet de simuler un cadre **réel de détection** et d’éviter toute fuite d’information.

### Modélisation

Le pipeline repose sur les étapes suivantes :

- encodage des variables catégorielles  
- ajout d’un score d’anomalie via **Isolation Forest (`if_score`)**  
- modèle principal : **XGBoost**  
- tuning des hyperparamètres  

L’intégration de l’Isolation Forest permet de capturer le caractère **atypique** des transactions.  
Dans un contexte fortement déséquilibré, ce signal non supervisé apporte une information complémentaire pertinente pour identifier des comportements rares.

Le modèle **XGBoost** est retenu pour sa capacité à modéliser des **relations non linéaires** et des **interactions complexes**, particulièrement adaptées aux problématiques de fraude.

Les hyperparamètres sont optimisés avec un focus sur la **stabilité des probabilités**, en utilisant le **Brier score** comme critère.  
Ce choix permet de limiter le surapprentissage et d’obtenir des probabilités cohérentes, indispensables pour la phase de décision.

La validation repose sur un **TimeSeriesSplit**, garantissant l’absence de fuite d’information et une cohérence avec un contexte de détection en temps réel.

---

### Calibration et seuil

Deux éléments structurants :

**Calibration des probabilités**

Une calibration est appliquée afin d’obtenir des probabilités **fiables et interprétables**.  
C’est un point critique, car ces probabilités sont directement utilisées pour la prise de décision.

**Optimisation du seuil**

Le seuil est optimisé selon une **fonction de coût métier** :

- faux positif : 25 €  
- faux négatif : 125 €  

L’objectif est de minimiser le **coût total**, et non une métrique purement statistique.

Ce choix reflète une logique opérationnelle :  
une fraude non détectée étant nettement plus coûteuse qu’une fausse alerte, le modèle est orienté vers un bon **rappel**, tout en maîtrisant le volume d’alertes.

---

## 5. Résultats

Le modèle présente des performances élevées, à la fois en termes de métriques ML et d’impact métier :

- ROC-AUC ≈ **0,99**  
- Recall ≈ **78,5 %** (≈ 8 fraudes sur 10 détectées)  
- Précision ≈ **51 %** (≈ 1 alerte sur 2 pertinente)  
- Alert rate ≈ **0,51 %** des transactions  

Le **seuil optimal (0,226)** est déterminé sur l’échantillon de validation en minimisant le coût.

Appliqué au jeu de test, il conduit à un **coût total d’environ 134 675 €**, traduisant un compromis efficace entre faux positifs et faux négatifs.

Ces résultats montrent que le modèle :

- détecte une part importante des fraudes  
- génère un volume d’alertes maîtrisé  
- reste cohérent avec les contraintes opérationnelles  

![Precision-Recall curve](./05_visualisations/courbe_pr.png)

![Calibration curve](./05_visualisations/courbe_calibration.png)

Un **drift du taux de fraude** est observé entre validation et test (≈ 0,05 % → ≈ 0,03 %).  
Ce phénomène explique :

- une baisse de la précision  
- un seuil devenu légèrement trop permissif  

Un ajustement du seuil serait donc nécessaire en production pour rester optimal.

---

## 6. Interprétabilité du modèle (SHAP)

Le modèle est analysé avec **SHAP** afin de comprendre ses décisions, à la fois au niveau global et individuel.

### Importance globale

Variables principales :

- **montant de la transaction**  
- **heure de la transaction**  
- **score d’anomalie (`if_score`)**  
- **type de magasin**  

![Feature importance](./05_visualisations/feature_importance.png)

### Effets moyens

L’analyse des contributions met en évidence plusieurs comportements :

- des **montants atypiques** (faibles ou élevés) sont plus risqués  
- certaines **plages horaires** sont plus exposées  
- les transactions jugées **anormales** sont plus susceptibles d’être frauduleuses  

![Beeswarm](./05_visualisations/beeswarm_plot.png)

### Explications locales

Chaque décision du modèle est **interprétable individuellement**, ce qui permet de comprendre précisément les facteurs de risque.

Transaction non frauduleuse :
- comportement **cohérent avec l’historique**  
- **faible probabilité** de fraude  

![Non fraude](./05_visualisations/fraud_target_0.png)

Transaction frauduleuse :
- comportement **atypique**  
- **accumulation de signaux de risque**  

![Fraude](./05_visualisations/fraud_target_1.png)

### Interactions

Le modèle ne repose pas sur des règles simples.  
Il combine plusieurs variables pour estimer le risque, ce qui permet de capturer des **patterns de fraude complexes**.

![Dependence plot](./05_visualisations/dependence_plot.png)

## 7. Pipeline

Le projet est structuré sous forme de **pipeline exécutable** :

full_fraud_detection_projet()

Il enchaîne :

- feature engineering  
- nettoyage post-EDA  
- modélisation complète  

Cette organisation garantit la **reproductibilité des résultats** et s’inscrit dans une logique proche des pratiques **MLOps**.

## 8. Structure du projet

- `01_data` : données brutes et données transformées  
- `02_notebooks` : notebooks d’analyse (feature engineering, EDA, modélisation, interprétation)  
- `03_fonctions` : fonctions réutilisables (cleaning, pipeline ML, utilitaires)  
- `04_model` : modèles sauvegardés et artefacts  
- `05_visualisations` : graphiques utilisés dans le projet et le README  
- `06_streamlit` : application interactive  
- `full_pipeline.py` : orchestration complète du projet  

Cette structure permet de séparer clairement les phases **exploratoires** des composants **réutilisables**, facilitant la maintenance et l’évolution du projet.

## 9. Application Streamlit

Une application interactive permet de rendre le modèle exploitable dans un contexte métier.

Fonctionnalités :

- visualisation des performances  
- analyse du coût  
- explicabilité des décisions (SHAP)  
- simulation de transactions  

Une brique d’**IA générative (LLM Sonar - Perplexity)** est intégrée afin de traduire les explications techniques en **langage naturel**, facilitant leur compréhension par des utilisateurs non techniques.

L’application transforme ainsi le modèle en un véritable **outil d’aide à la décision**.

## 10. Conclusion

Le modèle présente une **forte capacité de discrimination**, une **bonne cohérence probabiliste** et un compromis pertinent entre détection et coût opérationnel.

Il est exploitable en contexte réel, sous réserve d’un **ajustement du seuil** en fonction des conditions observées.

Ce projet illustre le passage d’une approche exploratoire à une solution **structurée, reproductible et orientée métier**, proche des pratiques industrielles en data science.