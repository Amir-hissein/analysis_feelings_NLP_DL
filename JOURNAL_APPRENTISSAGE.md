# 📓 Journal d'apprentissage — Analyse de sentiments (NLP + Deep Learning)

> Ce fichier retrace **ce que j'ai fait**, **ce que j'ai appris** et les **décisions prises** à chaque phase.
> Il est mis à jour au fur et à mesure de notre avancement.

**Projet :** classifieur de sentiments (positive / negative) sur le dataset **IMDB** (50 000 avis de films), construit de bout en bout pour apprendre le NLP moderne.

---

## 🗺️ Progression globale

| Phase | Titre | Statut |
|-------|-------|--------|
| 1 | Mise en place du projet | ✅ Terminée |
| 2 | Comprendre le dataset (EDA) | ✅ Terminée |
| 3 | Nettoyage des données | ✅ Terminée |
| 4 | Prétraitement NLP | ✅ Terminée |
| 5 | Feature Engineering | ✅ Terminée |
| 6 | Machine Learning classique | ✅ Terminée |
| 7 | Évaluation | ✅ Terminée |
| 8 | Deep Learning (PyTorch) | ✅ Terminée |
| 9 | Transformers (DistilBERT/BERT) | ✅ Terminée |
| 10 | Sauvegarde du modèle / inférence | ✅ Terminée |
| 11 | API FastAPI | ✅ Terminée |
| 12 | Frontend React | ✅ Terminée |
| 13 | Déploiement (Docker) | ✅ Terminée |
| 14 | Améliorations (négation) | ✅ Terminée |

---

## ✅ Phase 1 — Mise en place du projet

**Objectif :** préparer un environnement de travail propre et professionnel avant d'écrire du code ML.

**Ce que j'ai fait :**
- Vérifié la structure de dossiers pro (`data/`, `src/`, `models/`, `reports/`, `notebooks/`...).
- Recréé le **virtual environment (`venv`)** proprement (l'ancien était cassé car déplacé depuis un autre dossier → `pip` ne marchait plus).
- Installé les librairies de base via `requirements.txt` : pandas, numpy, matplotlib, seaborn, plotly, jupyterlab.

**Ce que j'ai appris :**
- **venv** = environnement isolé, évite les conflits de versions entre projets.
- **requirements.txt** = liste des libs + versions → environnement reproductible.
- Un venv **n'est pas déplaçable** : si on renomme/déplace le dossier, il faut le recréer.
- Organisation d'un projet ML : séparer données brutes / code / modèles / rapports.

**Livrable :** projet propre et reproductible. ✅

---

## ✅ Phase 2 — Comprendre le dataset (EDA)

**Objectif :** regarder et diagnostiquer les données AVANT de modéliser (« un modèle ne vaut que ce que valent ses données »).

**Ce que j'ai fait :**
- Script `notebooks/01_eda_diagnostic.py` : diagnostic chiffré du dataset.
- Script `notebooks/02_eda_visualisation.py` : graphiques + rapport.
- Créé `src/config/config.py` : configuration centrale (chemins, noms de colonnes) = une seule source de vérité.

**Chiffres clés trouvés :**
| Question | Réponse | Décision |
|---|---|---|
| Taille | 50 000 avis, 2 colonnes (`review`, `sentiment`) | Volume OK ✅ |
| Valeurs manquantes | 0 | Rien à imputer ✅ |
| Doublons | 418 (0,84 %) | À supprimer (Phase 3) |
| Équilibre des classes | 50 % / 50 % | Accuracy fiable ✅ |
| Longueur des textes | moyenne 231 mots, médiane 173, max 2470 | Tronquer pour BERT (Phase 9) |

**Ce que j'ai appris :**
- **Pandas** : `read_csv`, `shape`, `info`, `isnull().sum()`, `duplicated().sum()`, `value_counts()`, `describe()`.
- **Seaborn/Matplotlib** : `countplot` (barres), `histplot` (histogramme), lignes de repère (`axvline`).
- **Concept clé — asymétrie (skew)** : quand **moyenne > médiane**, la distribution a une **traîne à droite** (quelques valeurs extrêmes tirent la moyenne vers le haut).
- **Pourquoi l'équilibre des classes compte** : sur un dataset 50/50, l'accuracy est fiable ; sur du déséquilibré (ex. 95/5), elle est trompeuse.

**Livrables :** `reports/EDA_report.md`, `reports/01_equilibre_classes.png`, `reports/02_distribution_longueur.png`. ✅

---

## ✅ Phase 3 — Nettoyage des données

**Objectif :** appliquer les décisions de la Phase 2 pour obtenir un texte propre, sans jamais toucher aux données brutes.

**Ce que j'ai fait :**
- Créé `src/preprocessing/cleaning.py` : fonction réutilisable `clean_text()` (réutilisée plus tard par l'API).
- Script `notebooks/03_nettoyage.py` : supprime les doublons, encode les labels, nettoie le texte, sauvegarde.
- Généré `data/processed/imdb_clean.csv` (49 582 lignes, 4 colonnes).

**Résultats :**
| Étape | Résultat |
|---|---|
| Doublons | 50 000 → 49 582 (418 retirés) |
| Labels | `positive → 1`, `negative → 0` |
| Nettoyage | minuscules, sans HTML/URL/ponctuation |
| Vérifs | 0 avis vide, 0 label mal encodé ✅ |

**Ce que j'ai appris :**
- **Le niveau de nettoyage dépend du modèle** : agressif pour ML classique (TF-IDF), minimal pour BERT.
- On garde **les deux** versions : `review` (original, pour BERT) et `review_clean` (nettoyé, pour ML classique). On ne jette jamais l'original.
- **Expressions régulières (`re`)** : `<[^>]+>` (balises HTML), `http\S+` (URLs), `[^a-z\s]` (tout sauf lettres/espaces), `\s+` (espaces multiples).
- Pandas : `drop_duplicates()`, `reset_index()`, `.map()` (encoder), `.apply()` (appliquer une fonction à chaque ligne).
- **Réflexe pro** : toujours vérifier qu'un nettoyage n'a pas détruit de données (avis vides, NaN).
- Séparation stricte `data/raw/` (jamais modifié) vs `data/processed/` (résultat traité).

**Livrable :** `data/processed/imdb_clean.csv`. ✅

---

## ✅ Phase 4 — Prétraitement NLP

**Objectif :** transformer le texte nettoyé (une grande chaîne) en une liste de **mots utiles** ramenés à leur forme de base, que le ML classique (TF-IDF) sait exploiter.

**Ce que j'ai fait :**
- Choisi **spaCy** (plutôt que NLTK) : pipeline NLP moderne, un seul objet fait tout.
- Créé `src/preprocessing/nlp_preprocessing.py` : `preprocess()` (1 texte) et `preprocess_batch()` (rapide, via `nlp.pipe`).
- Script `notebooks/04_pretraitement_nlp.py` : applique le prétraitement, mesure l'impact, sauvegarde.
- Généré `data/processed/imdb_preprocessed.csv` (colonne `review_nlp` ajoutée).

**Résultats :**
| Mesure | Valeur |
|---|---|
| Mots/avis avant → après | 234 → 109 (**−54 %**) |
| Avis devenus vides | 0 ✅ |
| Durée (49 582 avis) | ~709 s (~12 min), 70 avis/s |

**Ce que j'ai appris :**
- **spaCy = pipeline** : `texte → tokenizer → tagger → lemmatizer → Doc`. Chaque **token** est un objet riche : `.text`, `.lemma_`, `.is_stop`, `.is_alpha`, `.pos_`.
- **Tokenisation** : découper le texte en mots. **Stopwords** : mots vides (the, is, a...) qu'on retire car ils ajoutent du bruit. **Lemmatisation** : forme de base (`loved→love`, `was→be`, `better→well`).
- Tout le prétraitement tient en **une ligne** : `[t.lemma_ for t in doc if t.is_alpha and not t.is_stop]`.
- **`nlp.pipe()`** (traitement par lots) >> boucle `.apply()` : bien plus rapide sur des milliers de textes. On **désactive** `parser`+`ner` (inutiles ici) pour accélérer.
- La lemmatisation n'est **pas parfaite** (`discomforting→discomforte`) ; le ML classique tolère ce bruit.
- **Coût payé une fois** : on sauvegarde le résultat en CSV pour ne jamais recalculer.
- ⚠️ Ce prétraitement agressif est **pour le ML classique**. Pour **BERT** (Phase 9), on N'appliquera PAS ce module : les Transformers ont besoin des mots entiers, des stopwords et de l'ordre.

**Livrable :** `data/processed/imdb_preprocessed.csv`. ✅

---

## ✅ Phase 5 — Feature Engineering (TF-IDF) + découpage train/test

**Objectif :** transformer le texte en **vecteurs de nombres** exploitables par le ML, et préparer une évaluation honnête (train/test).

**Ce que j'ai fait :**
- Ajouté `scikit-learn` (amène scipy + joblib).
- Ajouté au `config.py` : `TEST_SIZE`, `TFIDF_MAX_FEATURES`, `VECTORIZER_PATH`.
- Créé `src/features/vectorizer.py` : `build_vectorizer()` (TF-IDF paramétré, réutilisé par l'API).
- Script `notebooks/05_feature_engineering.py` : split → fit TF-IDF sur train → transform → sauvegarde.
- Généré `X_train.npz`, `X_test.npz`, `y_train.npy`, `y_test.npy` + `models/tfidf_vectorizer.joblib`.

**Résultats :**
| Mesure | Valeur |
|---|---|
| Train / Test | 39 665 / 9 917 (80/20) |
| Équilibre préservé (stratify) | 0.502 / 0.502 ✅ |
| Vocabulaire TF-IDF | 5 000 termes (mots + bigrammes) |
| Matrice train | (39665 × 5000), densité 1.49 % (creuse) |

**Ce que j'ai appris :**
- **Le ML ne mange que des nombres** : il faut vectoriser le texte (Feature Engineering).
- **Bag of Words** = compter les mots. Défaut : les mots courants (movie) écrasent les rares.
- **TF-IDF** = TF (fréquent dans CET avis) × IDF (rare dans TOUS) → fait ressortir les mots **rares et discriminants**. Vérifié : le top-mots d'un avis ne contient JAMAIS movie/film/good.
- ⚠️ **Fuite de données (data leakage)** = LA règle d'or : `.fit()` sur le **train seulement**, `.transform()` sur les deux. Sinon l'évaluation est mensongère.
- **`train_test_split(..., stratify=y, random_state=42)`** : découpage reproductible qui préserve l'équilibre des classes.
- **Matrice creuse (sparse)** : 98,5 % de zéros → on ne stocke que le reste (scipy `.npz`). Indispensable en NLP.
- Hyperparamètres TF-IDF utiles : `max_features`, `ngram_range=(1,2)` (capte "not good"), `min_df`, `sublinear_tf`.
- On **sauvegarde le vectoriseur entraîné** (`joblib`) : l'API devra vectoriser un nouvel avis EXACTEMENT pareil.

**Livrables :** `X_train/X_test.npz`, `y_train/y_test.npy`, `models/tfidf_vectorizer.joblib`. ✅

---

## ✅ Phase 6 — Machine Learning classique (baseline)

**Objectif :** entraîner un premier vrai classifieur et obtenir un score de référence.

**Ce que j'ai fait :**
- Ajouté `LOGREG_MODEL_PATH` au `config.py`.
- Créé `src/models/classifier.py` : `build_model()` (Régression Logistique paramétrée).
- Script `notebooks/06_train_baseline.py` : charge les matrices, entraîne, évalue, inspecte les poids, sauvegarde.
- Généré `models/logreg_model.joblib`.

**Résultats :**
| Mesure | Valeur |
|---|---|
| Accuracy TRAIN | 90,6 % |
| **Accuracy TEST** | **88,6 %** (la vraie mesure) |
| Écart train-test | 2,1 pts → pas de surapprentissage ✅ |
| Temps d'entraînement | 0,1 s |

**Ce que j'ai appris :**
- **Régression Logistique** : chaque mot reçoit un **poids** (+ = pousse vers positif, − = vers négatif). Le modèle fait `somme(poids × tfidf)` → **sigmoïde** → probabilité entre 0 et 1.
- **« Entraîner » = `.fit()`** : trouver les 5000 poids qui séparent le mieux + de −.
- **Baseline d'abord** : toujours commencer par un modèle simple. 88,6 % en 0,1 s = le score à battre par le Deep Learning.
- **Overfitting (surapprentissage)** : si Train >> Test, le modèle a mémorisé au lieu d'apprendre. Ici écart de 2,1 pts = sain.
- **Interprétabilité** : `model.coef_` = les poids. On lit les mots décisifs. Positifs : excellent, great, perfect, brilliant. Négatifs : bad, waste, awful, boring. Le modèle a **déduit** le sentiment des données, sans qu'on le lui dise.
- Le modèle capte des **corrélations, pas du sens** (ex. `suppose` classé négatif via "supposed to be good but…").
- On **sauvegarde le modèle** (`joblib`) pour l'API (Phase 11).

**Livrable :** `models/logreg_model.joblib`. ✅

---

## ✅ Phase 7 — Évaluation détaillée

**Objectif :** aller au-delà de l'accuracy pour comprendre COMMENT le modèle se trompe.

**Ce que j'ai fait :**
- Script `notebooks/07_evaluation.py` : rapport de classification, matrice de confusion, courbe ROC, exemples mal classés.
- Généré `reports/03_confusion_matrix.png`, `reports/04_roc_curve.png`, `reports/EVAL_report.md`.

**Résultats :**
| Métrique | Valeur |
|---|---|
| Accuracy | 88,6 % |
| **AUC (ROC)** | **0,956** |
| F1 macro | 0,89 |
| Erreurs | 1 133 / 9 917 (626 FP, 507 FN) |

**Ce que j'ai appris :**
- **L'accuracy est une moyenne** qui cache la structure des erreurs → il faut creuser.
- **Matrice de confusion** : croise prédit × vérité → 4 cases (TN, FP, FN, TP). La diagonale = bonnes réponses.
- **Precision** = `TP/(TP+FP)` (« quand il dit positif, a-t-il raison ? »). **Recall** = `TP/(TP+FN)` (« attrape-t-il tous les positifs ? »). **F1** = moyenne harmonique des deux. Compromis fréquent entre précision et rappel.
- **Courbe ROC / AUC** : teste TOUS les seuils de décision → métrique reine, indépendante du seuil. AUC=1 parfait, 0.5 hasard. Ici 0,956 = excellent.
- **Analyse des erreurs (le plus instructif)** : les faux positifs sont des avis négatifs qui louent avant un retournement ; les faux négatifs contiennent des négations/nuances.
- ⚠️ **LIMITE FONDAMENTALE du TF-IDF** : il **compte des mots**, ignore l'**ordre** et le **contexte** (négation, sarcasme, retournement). → C'est LA raison de passer au Deep Learning (Phases 8-9), qui lit la phrase dans l'ordre.
- Dataviz : heatmap séquentielle à une teinte pour la confusion, ligne + diagonale de référence pour la ROC.

**Livrables :** `reports/03_confusion_matrix.png`, `reports/04_roc_curve.png`, `reports/EVAL_report.md`. ✅

---

## ✅ Phase 8 — Deep Learning (réseau LSTM, PyTorch)

**Objectif :** construire un premier réseau de neurones qui lit l'avis mot par mot, dans l'ordre.

**Ce que j'ai fait :**
- Ajouté `torch` + hyperparamètres LSTM au `config.py`.
- Créé `src/models/lstm.py` (`SentimentLSTM`) et `src/data/text_dataset.py` (vocab, encodage, Dataset).
- Script `notebooks/08_train_lstm.py` : split, DataLoaders, boucle d'entraînement, évaluation.
- Généré `models/lstm_model.pt` + `models/lstm_vocab.json`.
- Entraîné sur le GPU Apple Silicon (MPS).

**Résultats :**
| Epoch | perte | val acc |
|---|---|---|
| 1 | 0,635 | 73,1 % |
| 2 | 0,524 | 77,5 % |
| 3 | 0,434 | 82,1 % |
| 4 | 0,343 | 84,8 % |

Accuracy TEST : **84,6 %** (baseline logistique : 88,6 %). Val acc encore en hausse → sous-entraîné.

**Ce que j'ai appris :**
- **Embeddings** : chaque mot → vecteur dense appris (ex. 100 dim). Le sens devient de la géométrie (good ≈ great). Remplace les colonnes isolées du TF-IDF.
- **LSTM** : lit la séquence dans l'ordre avec une mémoire → peut capter la négation, contrairement au TF-IDF.
- **Boucle d'entraînement** (le cœur du DL) : `zero_grad → forward → loss → backward → step`, répétée sur des batches, sur plusieurs **epochs**. Le réseau apprend **progressivement** (≠ `.fit()` en un coup).
- **Pipeline données DL** : tokeniser → vocabulaire (mot→id) → encoder → padding/troncature à longueur fixe → `Dataset`/`DataLoader`. Tokens spéciaux `<pad>` et `<unk>`.
- ⚠️ **Preprocessing différent** : on utilise `review_clean` (ordre + négation `not` préservés), PAS `review_nlp` (spaCy avait supprimé `not`).
- 🐛 **BUG rencontré & corrigé** : au 1er essai le modèle stagnait à 51 % (hasard). Cause : je prenais `h_n` = état APRÈS des dizaines de tokens `<pad>` → le sens se diluait. 1ère correction (`pack_padded_sequence`) = correcte mais **non supportée sur MPS** → repli CPU ultra-lent. 2ᵉ correction retenue : faire tourner le LSTM sur tout, puis **`gather`** la sortie au dernier VRAI mot (padding à droite = n'affecte pas les positions antérieures). 51 % → 85 %.
- 🎓 **Leçon clé** : un modèle plus complexe n'est PAS automatiquement meilleur. Le DL « from scratch » exige beaucoup de données. La vraie puissance vient du **pré-entraînement** (transfer learning) → Phase 9.
- Techniques : `Dropout` (anti-surapprentissage), `BCEWithLogitsLoss`, optimiseur `Adam`, **gradient clipping** (stabilise le LSTM), GPU **MPS**.

**Livrables :** `models/lstm_model.pt`, `models/lstm_vocab.json`. ✅

---

## ✅ Phase 9 — Transformers (fine-tuning de DistilBERT)

**Objectif :** utiliser un Transformer PRÉ-ENTRAÎNÉ (DistilBERT) et l'ajuster à notre tâche (transfer learning).

**Ce que j'ai fait :**
- Ajouté `transformers` + hyperparamètres BERT au `config.py`.
- Script `notebooks/09_finetune_distilbert.py` : tokenizer WordPiece, `DistilBertForSequenceClassification`, boucle de fine-tuning (même boucle qu'en Phase 8), évaluation, sauvegarde.
- Généré `models/distilbert_sentiment/` (modèle + tokenizer au format Hugging Face).

**Résultats (le grand duel, sur le même test de 9 917 avis) :**
| Modèle | Accuracy | Données d'entraînement |
|---|---|---|
| **TF-IDF + Régression Logistique** | **88,6 %** | 39 665 avis, 0,1 s |
| DistilBERT (256 tokens, 8k) | 88,1 % | 8 000 avis, ~35 min |
| LSTM (from scratch) | 84,6 % | 35 698 avis |

**Ce que j'ai appris :**
- **Attention** : le Transformer regarde tous les mots à la fois et apprend quels mots comptent → capte la négation ("not good") et les dépendances longues, là où LSTM/TF-IDF peinent.
- **Pré-entraînement** : BERT a d'abord appris la langue sur des milliards de mots (mots masqués). Il "connaît" l'anglais avant de voir nos avis. C'est LA révolution.
- **Fine-tuning (transfer learning)** : on ajuste finement un modèle pré-entraîné (LR minuscule 2e-5) plutôt que de repartir de zéro. Quelques epochs suffisent.
- **Tokenizer WordPiece** : découpe les mots rares en sous-morceaux (`unbelievably → un ##bel ##ie ##va ##bly`) → plus jamais de `<unk>`. Jetons spéciaux `[CLS]`/`[SEP]`.
- On réutilise **la même boucle d'entraînement** qu'en Phase 8 ; seul le modèle change.
- 🐛 **Compromis calcul rencontré** : 1er essai à `max_len=128` → BERT ne lisait que le 1er tiers des avis → 85,0 % (sous la baseline). En repassant à `max_len=256` → 88,1 % (+3 pts). **Diagnostic confirmé : la troncature était le coupable.** Leçon : les compromis de calcul ont des conséquences directes.
- 🎓 **Leçon majeure** : BERT a ÉGALÉ (pas dépassé) une baseline entraînée sur 5× plus de données → il est bien plus **efficace par exemple**, mais **une bonne baseline classique est redoutable** sur une tâche binaire nette. On ne sort pas l'artillerie lourde par réflexe. BERT gagnerait sur des tâches plus dures ou avec le dataset complet (impraticable sans GPU dédié ici).

**Livrable :** `models/distilbert_sentiment/`. ✅

---

## ✅ Phase 10 — Emballage du meilleur modèle (inférence)

**Objectif :** transformer le meilleur modèle (baseline TF-IDF + Régression Logistique, 88,6 %) en une fonction simple et fiable, prête pour l'API.

**Ce que j'ai fait :**
- Créé `src/inference/predict.py` : `predict_sentiment(texte)` → `{label, proba_positif, confiance}`.
- Chargement paresseux des artefacts (vectoriseur + modèle chargés une seule fois).
- Script `notebooks/10_inference_demo.py` : test sur des phrases écrites à la main.

**Ce que j'ai appris :**
- ⚠️ **Cohérence entraînement ↔ prédiction (train/serve skew)** : un nouvel avis doit traverser EXACTEMENT le même pipeline qu'à l'entraînement. Notre TF-IDF a appris sur `review_nlp` = `preprocess(clean_text(avis))` → l'inférence rejoue ces 2 étapes avant de vectoriser (`transform`, jamais `fit`).
- **Résultat de la démo** : correct sur les cas nets (fantastic → positif 92 %, waste of time → négatif 100 %).
- 💥 **Erreur révélatrice** : *"The movie was not good at all"* → prédit POSITIF. Cause : `preprocess` a supprimé `not` (stopword) → il ne reste que "movie good" ; + le TF-IDF ignore l'ordre. Les limites des Phases 4 et 7 se combinent en une erreur concrète. C'est là que BERT (attention) serait qualitativement meilleur.
- **Réutilisabilité** : la logique vit dans `src/`, pas dans un notebook → l'API (Phase 11) l'importera telle quelle.

**Livrable :** `src/inference/predict.py`. ✅

---

## ✅ Phase 11 — API web (FastAPI)

**Objectif :** exposer le modèle via le web pour qu'une app cliente (frontend React) puisse l'utiliser.

**Ce que j'ai fait :**
- Ajouté `fastapi`, `uvicorn`, `pydantic` au `requirements.txt`.
- Créé `src/api/main.py` : routes `GET /`, `GET /health`, `POST /predict`.
- Schémas Pydantic `ReviewIn` / `PredictionOut` (validation + doc automatiques).
- Testé le serveur avec `curl` (4 tests OK).

**Ce que j'ai appris :**
- **API REST** : un client envoie une requête HTTP (`POST /predict` + JSON), le serveur répond en JSON. Découple totalement le modèle de l'app cliente.
- **FastAPI** : une fonction Python + un décorateur (`@app.post`) = un endpoint.
- **Pydantic** : on déclare les types/contraintes (`Field(min_length=1)`) → validation AUTOMATIQUE. Un texte vide est rejeté avec un message clair, sans code de validation manuel.
- **Docs interactives** générées seules sur `/docs` (Swagger).
- **Séparation des responsabilités** : la logique métier reste dans `src/inference/predict.py` ; l'API ne fait que l'emballer en routes HTTP.
- **Lancer le serveur** : `uvicorn src.api.main:app --reload`.

**Livrable :** `src/api/main.py`. ✅

---

## ✅ Phase 12 — Frontend React

**Objectif :** une interface web où l'utilisateur tape un avis et voit le sentiment s'afficher.

**Ce que j'ai fait :**
- Créé un projet **Vite + React** dans `frontend/`.
- Écrit `frontend/src/App.jsx` : champ de texte, bouton Analyser, appel `fetch` vers l'API, affichage vert/rouge + barre de confiance.
- Style dans `frontend/src/App.css` (+ `index.css` simplifié).
- Ajouté le **CORS** dans `src/api/main.py` (autorise `localhost:5173`).
- Testé de bout en bout : CORS OK, API répond, page servie.

**Ce que j'ai appris :**
- **Composant React** = fonction JS qui retourne du **JSX** (du HTML dans du JS).
- **`useState`** = la mémoire du composant ; `setXxx()` redessine l'UI automatiquement. Utilisé pour `texte`, `resultat`, `chargement`, `erreur`.
- **`fetch` + `async/await`** : appel HTTP asynchrone vers l'API, `await res.json()` pour lire la réponse.
- **Affichage conditionnel** (`{resultat && ...}`) et classes dynamiques (`className={estPositif ? 'positif' : 'negatif'}`).
- ⚠️ **CORS** : le navigateur bloque par défaut les appels frontend→API entre origines différentes. Il faut l'autoriser côté serveur (`CORSMiddleware` FastAPI).
- **Vite** : outil moderne de dev React (serveur `npm run dev` sur le port 5173, rechargement à chaud).

**Lancer l'appli** (2 terminaux) :
1. API : `uvicorn src.api.main:app --reload`
2. Front : `cd frontend && npm run dev` → http://localhost:5173

**Livrables :** `frontend/` (projet React), CORS dans `src/api/main.py`. ✅

### Bonus Phase 12 — Graphique des contributions (dataviz)
- **API** : `predict_sentiment()` renvoie maintenant `contributions` = pour chaque mot présent, `score TF-IDF × poids appris` (exploite l'**interprétabilité** de la régression logistique). Schéma Pydantic `Contribution` ajouté.
- **Frontend** : `frontend/src/Contributions.jsx` = graphe à **barres divergentes** (vert = pousse positif ▶, rouge = négatif ◀, longueur = importance). Montre POURQUOI le modèle décide.
- **Dataviz** : paire divergente vert/rouge + encodages secondaires (position gauche/droite, signe, valeur) → lisible même en cas de daltonisme ; ligne du zéro au centre ; thème clair/sombre.
- Vérifié : "waste time" (bigramme) capté à −2.16 ; "brilliant" +2.08. ✅

### Bonus Phase 12 — Histogramme de session
- `frontend/src/SessionHistogram.jsx` : compte positifs/négatifs + confiance moyenne sur tous les avis testés dans la session.
- Concept React : historique stocké dans `useState` ; chaque analyse l'enrichit → l'UI se redessine automatiquement.

---

## ✅ Phase 13 — Déploiement Docker

**Objectif :** empaqueter l'appli (API + frontend) dans des conteneurs pour qu'elle tourne à l'identique partout.

**Ce que j'ai fait :**
- `requirements-api.txt` : dépendances **minimales** de l'API (pas de torch/transformers → image plus légère).
- `Dockerfile.api` : image API (python:3.13-slim + spaCy + modèles .joblib).
- `frontend/Dockerfile` : **build multi-étapes** (Node compile React → nginx sert le statique).
- `.dockerignore` (racine + frontend) : exclut venv/data/gros modèles/node_modules.
- `docker-compose.yml` : orchestre API (8000) + frontend (8080).
- CORS : ajout de `localhost:8080` (origine du frontend Docker).
- Buildé et lancé : les 2 conteneurs tournent, testés OK.

**Résultats :** image API 952 Mo, frontend 92 Mo. Appli complète sur http://localhost:8080.

**Ce que j'ai appris :**
- **Image** (recette figée via Dockerfile) vs **conteneur** (image en cours d'exécution). « Marche partout ».
- **Couches Docker en cache** : copier `requirements` AVANT le code → l'install n'est pas refaite si seul le code change.
- **Build multi-étapes** : compiler avec Node, ne garder que le résultat servi par nginx → image finale minuscule (pas de node_modules).
- **`--host 0.0.0.0`** obligatoire dans un conteneur (écouter toutes les interfaces).
- **docker-compose** : lancer/relier plusieurs services (`up --build`, `down`, `ps`).
- **Optimisation prod** : image d'inférence sans les libs d'entraînement (torch/transformers).
- `.dockerignore` : indispensable pour ne pas envoyer venv/data (300 Mo+) au contexte de build.

**Lancer en prod :** `docker compose up --build` → http://localhost:8080

**Livrables :** `Dockerfile.api`, `frontend/Dockerfile`, `docker-compose.yml`, `requirements-api.txt`, `.dockerignore`. ✅

### Bonus Phase 13 — CI (GitHub Actions)
- `.github/workflows/ci.yml` : à chaque push sur main → build image API + **smoke test** (lance l'API et teste une prédiction) + build image frontend.
- Les 2 modèles baseline (222 Ko) sont versionnés (exception `.gitignore`) pour que la CI/Docker soit self-contained.
- CI vérifiée : **verte** ✅.

---

## ✅ Phase 14 — Amélioration : gestion de la négation

**Objectif :** corriger le bug découvert en Phase 10 ("not good at all" → prédit positif) et mesurer si ça aide.

**Ce que j'ai fait :**
- Ajouté `preprocess_neg()` / `preprocess_neg_batch()` dans `nlp_preprocessing.py` : **marquage de négation** (préfixe `neg_` après une négation, jusqu'à la ponctuation ou un mot de contraste comme "but").
- Script `notebooks/11_negation_retrain.py` : réentraîne TF-IDF + Régression Logistique sur le texte marqué, compare, sauvegarde.
- Branché le modèle amélioré sur l'API (`predict.py` utilise `preprocess_neg` + les modèles `*_neg`).
- Mis à jour `Dockerfile.api` + `.gitignore` pour embarquer le nouveau modèle. Testé dans le conteneur.

**Résultats :**
| Modèle | Accuracy | "not good at all" |
|---|---|---|
| Baseline | 88,6 % | positif ❌ |
| **Avec négation** | **89,3 %** (+0,7) | **négatif** ✅ |

**Ce que j'ai appris :**
- **Marquage de négation** : technique NLP classique. "not good" → "neg_good" (un mot distinct que le modèle apprend). Bornes = ponctuation + mots de contraste ("but").
- ⚠️ Il faut **RÉENTRAÎNER** : le modèle n'a aucun poids pour `neg_good` tant qu'il ne l'a pas vu à l'entraînement. Changer le prétraitement sans réentraîner ne sert à rien.
- Gain réel mais modeste (+0,7 pt), et le **bug principal est corrigé**.
- 🎓 **Limite honnête** : "not a bad film" (litote = positif) reste mal classé. Le marquage crée des mots, il ne "comprend" pas la double négation. C'est là que l'attention de BERT ferait mieux. Aucune méthode n'est magique.
- Bonus : le mot `neg_good` apparaît dans le graphe des contributions → la négation est **visible** dans l'UI.

**Livrables :** `models/tfidf_vectorizer_neg.joblib`, `models/logreg_model_neg.joblib`, `notebooks/11_negation_retrain.py`. ✅

---

## 🏁 PROJET TERMINÉ — 14 / 14 phases

De l'exploration des données à une application web déployée en Docker avec CI :
**EDA → nettoyage → NLP → TF-IDF → ML → évaluation → LSTM → BERT → inférence → API → React → Docker → CI → négation.**
Modèle final : TF-IDF + Régression Logistique **avec gestion de la négation (89,3 %)**, interprétable et déployé.

---
