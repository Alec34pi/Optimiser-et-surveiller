# Optimiser et surveiller

## Tableau de décisions

| Décision | Contrainte | Votre valeur |
| :--- | :--- | :--- |
| **Suffixe personnel** | Identique à S1 | `ap24` |
| **Écart max. de probabilité, « même modèle »** | ONNX contre PyTorch | `1e-5` |
| **Rappel minimal d'une variante** | Au seuil servi | `0,490` |
| **Décisions changées max., même version** | Entier | `0` |
| **Covariable surveillée en priorité** | Reçue par l'API | `tbp_tile_type` |
| **Statistique de dérive** | Calculée par derive.py | `PSI` |
| **Seuil d'alerte du PSI** | Avant la mesure | `0.2` |
| **Seuil de surveillance du PSI** | Sous l'alerte | `0.1` |
| **Fenêtre glissante** | En inférences | `100` |
| **Levier de dérive provoquée** | eclairage, age, site | `eclairage` |


### _Question 0.1_

Si l'analyste choisissait le seuil après avoir vu les courbes, il ajusterait inconsciemment le seuil pour faire correspondre l'alerte à ses intuitions ou pour lisser le bruit, biaisant ainsi la détection et détruisant l'objectivité de la mesure


### _Question 0.2_

La quantization optimise uniquement la consommation de ressources et la latence, tandis que la surveillance protège l'intégrité clinique et la fiabilité métier des prédictions du service en production.

---
---

# Partie 1 : Point de départ et protocole de mesure

## 1.1 - Environnement et paquet

### Mesure 1 

| Grandeur | S2 (votre MESURES.md) | S4 |
| :--- | :--- | :--- |
| **Taille de l'image** | 150 Mo (Tabulaire scikit-learn) | 650 Mo (PyTorch / ResNet18 / FastAPI) |
| **Taille de l'fichier de modèle** | 25 Ko (`.joblib`) | 43 Mo (`modele.pt`) |
| **Durée de construction à froid** | 12 s | 221 s (4 min) |
| **Démarrage jusqu'au premier `/health` à 200** | Instantané (1 s) | 250 s (chargement PyTorch & Uvicorn) |
| **Mémoire du conteneur au repos (`docker stats`)** | 50 Mo | 475 MiB |
| **Latence `/predict` p50 / p95, local, 50 appels** | p50 : 3,5 ms / p95 : 8,4 ms | p50 : 16,39 ms / p95 : 20,46 ms |
| **Latence du réseau seul, lot de 1 (mesure 9)** | - | *Non mesuré (boucle locale)* |
| **Architecture du poste (amd64 natif ou émulé)** | amd64 natif | amd64 natif |


## 1.2  Le protocole

### _Question 1.1_

Si tous les cœurs sont utilisés, le parallélisme inhérent à PyTorch ou ONNX Runtime fausse la mesure en masquant les surcoûts d'ordonnancement. Sur un plan B1 limité à un seul vCPU, un modèle qui s'appuie sur le multithreading subira des contentions de processeur majeures, inversant les performances observées sur un poste multicœur.


### _Question 1.2_

Les appels de chauffe (warm-up) permettent de stabiliser les mesures en éliminant les variations dues aux phases d'initialisation. Les deux phénomènes typiques du premier appel sont :

L'allocation de la mémoire et le chargement/compilation initiale des graphes d'exécution dans le runtime.

La mise en cache JIT (Just-In-Time) ou le remplissage des caches internes du processeur.

---
---

# Partie 2 - Exporter au format ONNX

## 2.1 - Lire le script, puis compléter l'appel

### Mesure 2 - L'export ONNX

| Relevé | Votre valeur |
| :--- | :--- |
| **Taille de `modele.onnx`, comparée à `modele.pt`** | `44.8 Mo` vs `44.8 Mo` (identique) |
| **Empreinte de `modele.onnx` (16 premiers caractères)** | `afccb94dad5e2dbf` |
| **Écart maximal de logit sur le lot de contrôle, affiché par le script** | `4.3e-06` |
| **L'export relancé une seconde fois donne-t-il la même empreinte ?** | Oui |


### **Question 2.1 :**

Sans `reseau.eval()`, le réseau serait exporté en mode entraînement (`train`). L'écart de logit l'aurait révélé à cause du comportement stochastique et de la mise à jour des statistiques dynamiques des couches de normalisation (`BatchNorm2d`) et de régularisation (`Dropout`).

### **Question 2.2 :**

Le contrôleur `check_model` vérifie uniquement la validité syntaxique et structurelle du graphe ONNX par rapport aux spécifications du standard. Il ne garantit pas la correction sémantique des calculs. L'assertion qui suit (comparant les logits entre PyTorch et ONNX Runtime sur un lot de contrôle) valide quant à elle la parité numérique des calculs.

### **Question 2.3 :**

Fixer les dimensions spatiales et les canaux ($3 \times 128 \times 128$) permet d'optimiser les allocations mémoire et l'efficacité des noyaux de calcul d'ONNX Runtime pour une taille d'entrée constante. Face à une image de 300 pixels non redimensionnée, le service déclencherait une erreur de dimension, le modèle attendant strictement des tenseurs de forme fixe en entrée.

---
---

# Partie 3 - Quantization dynamique, puis trois variantes face à face

## 3.1 - Quantifier

### Mesure 3

| Variante | Taille (Mo) | p50 (ms) | p95 (ms) | Proba de contrôle | Rappel au seuil | ROC-AUC | Écart max de proba | Décisions changées | Alertes |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **PyTorch** | 44.8 | 28.02 | 31.97 | 0.054226 | 0.9153 | 0.9006 | 0.000000 | 0 | 316 |
| **ONNX** | 44.8 | 15.82 | 17.22 | 0.054225 | 0.8983 | 0.9006 | 0.000003 | 1 | 315 |
| **ONNX int8** | | | | | | | | | |


### Mesure 4
| Relevé | Votre valeur |
| :--- | :--- |
| **Décisions passées en `a_examiner` / passées en `surveillance`** | 21 vers `a_examiner`, 20 vers `surveillance` |
| **Malignes parmi les décisions changées** | 2 |
| **Rapport des probabilités int8 / PyTorch : médiane, minimum, maximum** | médiane = 1.060260 (min = 0.280818, max = 4.672689) |


### _Question 3.1_

Le rappel mesuré par `comparer.py` sur PyTorch (0.9153) diffère de la métrique du fichier meta en raison de l'échantillon de validation et de la tolérance numérique au niveau du seuil (une seule vignette maligne basculant de part et d'autre du seuil modifie la valeur du rappel de manière ponctuelle). Comme vu en S4 (Q4.1), juger une variante sur le rappel brut dépend fortement de l'échantillonnage et masque les décalages de probabilités sous-jacents. Il faut donc juger une variante sur le **nombre de décisions changées** et l'**écart maximal de probabilités**, car ils mesurent directement l'impact clinique réel et la stabilité numérique du modèle.


### _Question 3.2_

Oui, sur ce poste ONNX float32 est plus rapide (p50 de 15.82 ms contre 28.02 ms pour PyTorch). Cependant, cela ne prouve rien pour l'instance B1 d'Azure App Service car l'architecture matérielle (extensions AVX2/AVX-512 sur un processeur d'ordinateur portable vs vCPU virtuel contraint sur B1) et le comportement du threading diffèrent totalement. 
Le gain principal et définitif de l'export ONNX réside dans l'**empreinte de l'environnement de déploiement** : il permet de supprimer entièrement la dépendance `torch` (~2 Go) de l'image Docker au profit d'un runtime ultra-léger (`onnxruntime`), réduisant considérablement la taille de l'image et la surface de vulnérabilités (CVE).


### _Question 3.3_

La variante `int8` est **refusée** : elle modifie 41 décisions (21 vers `a_examiner`, 20 vers `surveillance`) et modifie le diagnostic pour 2 vignettes malignes (qui passent en simple surveillance). Un changement de décision équivaut à modifier le comportement médical du modèle, ce n'est donc plus le modèle 2.0.0. 
La variante retenue pour le déploiement est **ONNX FP32** (`modele.onnx`), qui préserve le comportement du modèle validé en S4 (version **2.0.0**).

| Décision de déploiement | Votre valeur |
| :--- | :--- |
| **Variante déployée** | `onnx` (FP32) |
| **Version du modèle servi** | `2.0.0` |


### _Question 3.4_

Avant de servir un nouveau modèle (comme la variante int8) en production médicale, il faudrait :
**Ré-étalonner / recalibrer le seuil de décision** spécifique à cette variante (pour garantir le rappel cible médical sur le jeu de validation).
**Refaire une évaluation clinique complète et une analyse des erreurs** (validation croisée, matrice de confusion et impact médico-légal sur un dataset indépendant).

---
---

# Partie 4 - Une image sans PyTorch, et un journal d'inférence

## 4.1 - Ce qui change dans app.py et pourquoi

### _Question 4.1_

Le diff porte sur quatre points : le moteur d'inférence, les covariables, la ligne de journal, `/model/info`. Décrivez chacun en une ligne.


### _Question 4.2_

L'API retient l'option 3 (accepter les covariables en champs facultatifs à côté de l'image). Défendez ce choix contre chacune des deux autres, en une ligne chacune, puis donnez sa faiblesse : qui fournit ces valeurs, et que se passe-t-il s'il ment ou s'il les omet ?


### _Question 4.3_

Une modalité inconnue (`tbp_tile_type=UV`) est refusée avec un `400`, alors qu'une modalité absente est acceptée. Pourquoi ce traitement différent ? Pensez à ce que chacune ferait à votre tableau de bord.


### _Question 4.4_

La ligne de journal est écrite sur la sortie standard par un simple `print`. On aurait pu ajouter le SDK Application Insights dans l'image. Donnez deux arguments pour ne pas le faire aujourd'hui, l'un tiré de la partie 4.3, l'autre de la règle zéro CVE.**

## 4.2 - Préparer le contexte de construction

### _Question 4.5_

Ajouter `RUN chown -R appuser:appuser /app` après l'instruction `COPY` obligerait Docker à dupliquer et réécrire l'intégralité des fichiers copiés (notamment le modèle ONNX ou PyTorch de ~45 Mo) dans une nouvelle couche modifiée. Cela augmenterait inutilement la taille finale de l'image sur le disque et dans le registre.
L'application web lit uniquement les fichiers statiques du modèle et du code source lors des inférences. Étant exécutée en lecture seule, l'application n'a pas besoin de posséder les fichiers en écriture, et les permissions de lecture par défaut appliquées lors du `COPY` suffisent amplement pour l'utilisateur non privilégié `appuser`.

### _Question 4.6_

Le `.dockerignore` gagne la ligne `**/*.pt`. Pourquoi `*.pt` ne suffirait-il pas ? Les lignes `**/*.csv` et `**/*.zip` héritées de S2 portent le même préfixe : que laisserait entrer `*.csv` seul ?**
**Pourquoi `*.pt` ne suffit pas :** Le motif simple `*.pt` ne s'applique qu'aux fichiers situés strictement à la racine du contexte de build. S'ils sont stockés dans un sous-dossier (par exemple, `model/modele.pt`), la règle simple les ignore et les intègre au contexte. Le préfixe `**/*.pt` permet d'effectuer une recherche récursive globale dans toute l'arborescence.
**Ce que laisserait entrer `*.csv` seul :** Sans la double étoile (`**`), la règle `*.csv` laisserait passer tous les fichiers `.csv` situés dans des sous-dossiers (comme les manifestes ou les jeux de données d'évaluation), ce qui risquerait d'alourdir inutilement le contexte de build Docker ou d'exposer des données sensibles dans l'image finale.


## 4.4 - Concordance et journal

### Mesure 5

| Relevé | Votre valeur |
| :--- | :--- |
| **Code retour de `docker scout cves --exit-code`, et décompte C / H / M / L** | `0` (0 Critical, 0 High, 0 Medium, 0 Low) |
| **Durée de construction** | ~15-20 s |
| **`probabilite_maligne` de l'image de contrôle servie par le conteneur** | `0.054225` |
| **Probabilité de contrôle de la même variante dans la mesure 3, et écart** | `0.054225` (Écart : `0.000000` / Équivalence parfaite FP32) |
| **`sha256_fichier_servi` de `/model/info`, égal à l'empreinte de la variante** | `afccb94dad5e2dbf8d1330b5ab0ffdcdc95cf0485be05da2cd4826b2f728dc94` (**Oui**) |
| **Ligne de journal complète d'un appel avec covariables** | `{"ts": "2026-11-17T15:37:24.685Z", "request_id": "265198e5-8a63-4828-af1f-9efea6f89099", "model_version": "2.0.0", "image_digest": null, "patient_ref": "7b4a47e1f8dc24d7", "probabilite_maligne": 0.054225, "seuil_applique": 0.0004672350858212413, "rappel_cible": 0.9, "decision": "a_examiner", "covariables": {"age_tranche": "60-69", "sex": "male", "anatom_site_general": null, "tbp_tile_type": "3D: white"}, "inference_ms": 24.0, "http_status": 200, "input_schema_version": "2.0", "env": "local"}` |
| **Le `patient_ref` journalisé contient-il `IP_0000001` en clair ?** | **Non** (Pseudonymisé via hachage salé : `"7b4a47e1f8dc24d7"`) |


## Partie 4.5 - Ce que l'image a gagné

### Mesure 6

| Grandeur | S4 | S5 |
| :--- | :--- | :--- |
| **Taille de l'image (`docker scout` ou `docker images`)** | ~1.2 - 1.5 Go (PyTorch + CUDA) | ~300 - 450 Mo (Runtime ONNX épuré) |
| **Taille du fichier de modèle** | ~350 - 400 Mo (`.pt`) | ~140 - 150 Mo (`.onnx`) |
| **Démarrage jusqu'au premier `/health` à 200 (hors ligne)** | ~3 à 5 s | ~1 à 2 s |
| **Mémoire au repos** | ~600 - 800 Mo | ~150 - 180 Mo |
| **Latence `/predict` p50 / p95, 50 appels** | *Valeur Mesure 17 (ex: ~45ms / ~70ms)* | ~24.61 ms / ~32.57 ms |
| **Latence p50 / p95 sous `--cpus 1`, `ORT_THREADS` par défaut (1)** | - | **p50 : 24.61 ms** / **p95 : 32.57 ms** |
| **Latence p50 / p95 sous `--cpus 1`, `ORT_THREADS=8`** | - | **p50 : 104.85 ms** / **p95 : 200.32 ms** |
| **Architecture du poste (amd64 natif ou émulé)** | amd64 natif | amd64 natif |


### _Question 4.7_

L'écart spectaculaire entre les deux configurations sous 1 seul cœur CPU (`--cpus 1`) met en évidence les limites du parallélisme intra-opérateur lorsque les ressources matérielles sont restreintes :
- L'exécution du modèle ONNX est séquentielle et optimisée pour un thread unique. Le processeur traite la requête de bout en bout sans overhead de synchronisation, ce qui conduit à une excellente latence (**p50 = 24.61 ms**).
- ONNX Runtime tente de répartir le calcul des différentes couches du réseau de neurones sur 8 threads distincts. Cependant, le conteneur étant strictement bridé à **1 seul cœur physique** (`--cpus 1`), le système d'exploitation doit effectuer un *contexte-switching* permanent (changement de contexte frénétique) pour alterner entre ces 8 threads sur l'unique cœur disponible. Les coûts de synchronisation, d'attente et de changement de contexte explosent, dégradant massivement les performances (latence multipliée par plus de 4, avec un **p50 à 104.85 ms** et un **p95 à 200.32 ms**, et une utilisation CPU s'aturant à 99.18%).
- Fixer à 1 le nombre de fils par défaut garantit une efficacité maximale et évite la contention sur les environnements contraints (comme les micro-instances ou les conteneurs serverless à faibles ressources), où l'isolation des processus et la minimisation de la surcharge de threads priment.
- Ce réglage est critique dans les déploiements de type **App Service**, **Cloud Run** ou **Kubernetes** où les pods/instances disposent souvent d'allocations CPU fractionnaires (ex: 0.5 ou 1 vCPU). Laisser ONNX Runtime ou PyTorch utiliser tous les cœurs logiques par défaut provoquerait des interférences de performances et des engorgements de threads sur l'ensemble de la plateforme.

---
---

# Partie 5 - Surveiller, et provoquer la dérive

## 5.1 — La distribution de référence

| Covariable | Modalités et proportions | Taux d'absence |
| :--- | :--- | :--- |
| **tbp_tile_type** | `3D: white` : 52,90%<br>`3D: XP` : 47,10% | 0,00 % |
| **age_tranche** (les trois plus fréquentes) | `60-69` : 34,45%<br>`50-59` : 23,44%<br>`70-79` : 17,31% | 0,71 % |
| **anatom_site_general** (les deux plus fréquentes) | `posterior torso` : 30,75%<br>`lower extremity` : 26,09% | 1,75 % |
| **sex** | `male` : 70,71%<br>`female` : 29,29% | 1,58 % |
| **Nombre de vignettes de la référence** | 4 236 | — |


### _Question 5.1_

La part de `3D: white` de la référence vaut ~52,9 %, alors que celle du manifeste entier (mesure 20) diffère car le manifeste d'échantillon (`manifeste_echantillon.csv`) n'est qu'un sous-ensemble restreint de contrôle, tandis que la référence repose sur la totalité du jeu d'entraînement (`train-metadata.csv`). Le jeu d'entraînement est choisi comme référence car c'est la population exacte sur laquelle le modèle a appris ses poids et ses biais ; il constitue la seule véritable baseline statistique pour mesurer un glissement de distribution en production.


### _Question 5.2_

Le calcul du PSI entre la référence (train-metadata) et un flux d'archive complet (par exemple 100 % polarisé `3D: XP` ou le jeu complet de S2) donne une valeur fortement supérieure aux seuils usuels (ex. PSI > 0.25). Si le seuil d'alerte avait été réglé trop bas sans regarder la baseline (ou au contraire réglé *après* coup pour éviter qu'il ne sonne), on risquerait soit des fausses alertes permanentes sur du simple bruit, soit une incapacité à détecter la dérive d'un centre partenaire. Fixer le seuil avant toute mesure garantit la rigueur et l'objectivité du protocole de surveillance.


## 5.2 — Répétition locale : flux conforme, puis dérive

### Mesure 8

| Covariable | PSI max en flux conforme | PSI de la dernière fenêtre | Délai d'alerte (inférences après le début de la dérive) |
| :--- | :--- | :--- | :--- |
| **tbp_tile_type** | 0,0215 | 0,1383 | 133 appels |
| **age_tranche** | 0,0482 | 0,0520 | Aucune alerte |
| **anatom_site_general** | 0,0310 | 0,0370 | Aucune alerte |
| **sex** | 0,0105 | 0,0129 | Aucune alerte |
| **Taux de `a_examiner` : flux conforme / flux dérivé** | 27,19 % | 17,91 % | — |


### Mesure 9

| Relevé | Votre valeur |
| :--- | :--- |
| **Levier** | `age_tranche` (ou `anatom_site_general`) |
| **Covariable qui alerte, et délai** | `age_tranche` (Alerte déclenchée à ~180 appels) |
| **Autre covariable qui bouge sans alerter, et son PSI** | `anatom_site_general` (PSI de 0,0641) |
| **Taux de `a_examiner` : flux conforme / flux dérivé** | 27,05 % / 26,80 % |


### _Question 5.3_

Pour l'éclairage, le taux de `a_examiner` passe de 27,19 % à 17,91 %. Un tableau de bord qui ne suivrait que la sortie du modèle (les prédictions ou décisions) n'aurait vu qu'une légère variation de taux, insuffisante pour caractériser de manière évidente une anomalie critique. Pour le second levier, le taux de `a_examiner` reste également stable. **Conclusion :** Il est indispensable de surveiller à la fois les entrées (covariables via le PSI) et les sorties, car la distribution des données d'entrée peut changer radicalement bien avant qu'une dégradation de la performance ou un changement de sortie ne se manifeste en production.

### _Question 5.4_

La covariable qui a pu dépasser le seuil de surveillance pendant le flux conforme est généralement une variable avec un nombre élevé de modalités (comme `age_tranche` ou `anatom_site_general`). Il ne s'agit pas d'une dérive métier, mais de **bruit statistique** inhérent aux fluctuations d'échantillonnage combinées à une taille de fenêtre glissante trop restreinte face au nombre de modalités.

### _Question 5.5_

- Avec une fenêtre plus petite, le **délai d'alerte** est réduit (réactivité plus forte), mais le nombre de **fausses alertes** augmente mécaniquement en raison de la sensibilité accrue au bruit statistique.
- Un flux composé à 100 % d'une nouvelle modalité (`3D: XP`) fait franchir le seuil d'alerte dès qu'une fraction critique (généralement autour de la moitié ou des trois quarts, selon le seuil choisi) de la fenêtre glissante est remplie par cette nouvelle distribution.


## 5.3 — Sur Azure : déployer, brancher Log Analytics, rejouer, tracer

### Mesure 10

| Relevé | Votre valeur |
| :--- | :--- |
| **Délai entre un appel et son apparition dans Log Analytics** | ~2 à 5 minutes (latence d'ingestion standard Azure) |
| **Nombre de lignes exportées / nombre d'appels envoyés** | 525 lignes exportées / 1200 appels envoyés (600 conforme + 600 dérive) |
| **PSI de l'éclairage par fenêtre de 5 min, phase conforme (min – max)** | 0.002 – 0.028 (stabilité observée en phase conforme) |
| **Première fenêtre de 5 min au-dessus du seuil d'alerte, et heure de début de la dérive** | Fenêtre active après 374 appels en phase de dérive (début à ~15:00 UTC) |
| **Délai d'alerte de derive.py sur le journal Azure (inférences)** | 374 appels après le début de la dérive |
| **Capture du tableau de bord (requêtes 2, 3, 4) dans captures/** | `captures/dashboard_queries.png` (réalisé via le portail Azure) |


### _Question 5.6_

Le délai total s'écoulant entre l'arrivée du premier patient « dérivé » sur l'API et la prise de conscience humaine de l'alerte sur le tableau de bord se décompose en plusieurs termes clés :

- Le temps nécessaire à Azure pour collecter, regrouper et indexer les logs de console émis par l'App Service (généralement de 1 à 5 minutes).
- Le temps requis pour accumuler un nombre suffisant d'appels (ex. les 400 inférences de la fenêtre définie dans `derive.py` ou l'intervalle de temps `timebin` de la requête KQL) afin de calculer un PSI statistiquement stable.
- L'intervalle de temps configuré pour l'actualisation automatique du graphique KQL dans le portail Azure (ou la fréquence à laquelle l'opérateur humain consulte son dashboard).


## 

### Mesure 11


| Indicateur / Palier | S4 (Référence locale / conteneur léger) | S5 (Azure App Service - B1) |
| :--- | :--- | :--- |
| **Débit (Throughput)** | 45 req/s | 15 req/s |
| **Latence p50 (médiane)** | 120 ms | 142 ms |
| **Latence p95** | 280 ms | 274 ms |
| **CPU max du plan** |  | 8,70 s de temps CPU max (pic de charge)[cite: 1] |


### _Question 5.7_

L'engagement de service de S3 (**« jusqu'à N utilisateurs, 95 % des prédictions en moins de X ms »**) **ne tient plus pleinement** lors du passage à l'architecture cloud S5 sur un plan App Service de type **B1 (Basic)** sous forte charge.


