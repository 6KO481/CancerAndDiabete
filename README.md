# API Python pour Prédictions Médicales

Cette API FastAPI héberge les modèles de machine learning pour les prédictions de cancer de la peau et de diabète.

## 📋 Prérequis

- Un compte sur [Render.com](https://render.com) (gratuit)
- Les fichiers de modèles dans le bon emplacement

## 🚀 Déploiement sur Render.com

### Étape 1: Préparer les fichiers

1. Créez un nouveau dépôt Git (GitHub, GitLab, ou Bitbucket)

2. Copiez tous les fichiers de ce dossier `api-python/` à la racine de votre dépôt

3. Créez la structure de dossiers pour les modèles:
   ```
   votre-depot/
   ├── main.py
   ├── requirements.txt
   ├── render.yaml
   ├── README.md
   └── models/
       ├── cancer/
       │   ├── model1.h5
       │   └── model2.h5
       └── diabete/
           └── xgb_model.pkl
   ```

4. Copiez vos modèles depuis le dossier `ml/` de votre projet:
   - `ml/cancer/model1.h5` → `models/cancer/model1.h5`
   - `ml/cancer/model2.h5` → `models/cancer/model2.h5`
   - `ml/diabete/xgb_model.pkl` → `models/diabete/xgb_model.pkl`

5. Commitez et pushez vers votre dépôt:
   ```bash
   git add .
   git commit -m "Initial commit - Medical AI API"
   git push origin main
   ```

### Étape 2: Déployer sur Render

1. Allez sur [Render.com](https://render.com) et connectez-vous

2. Cliquez sur **"New +"** → **"Web Service"**

3. Connectez votre dépôt Git

4. Configurez le service:
   - **Name**: `medical-ai-api` (ou un nom de votre choix)
   - **Runtime**: `Python 3`
   - **Build Command**: `pip install -r requirements.txt`
   - **Start Command**: `uvicorn main:app --host 0.0.0.0 --port $PORT`
   - **Plan**: Sélectionnez "Starter" (gratuit pour commencer)

5. Cliquez sur **"Create Web Service"**

6. Attendez que le déploiement se termine (5-10 minutes la première fois)

7. Une fois déployé, vous obtiendrez une URL comme:
   ```
   https://medical-ai-api.onrender.com
   ```

### Étape 3: Tester l'API

Testez que l'API fonctionne:

```bash
# Health check
curl https://votre-url.onrender.com/health

# Test prédiction diabète
curl -X POST https://votre-url.onrender.com/predict/diabetes \
  -H "Content-Type: application/json" \
  -d '{
    "gender": 1,
    "age": 45,
    "hypertension": 0,
    "heart_disease": 0,
    "smoking_history": 2,
    "bmi": 28.5,
    "HbA1c_level": 5.8,
    "blood_glucose_level": 120
  }'
```

### Étape 4: Configurer les Edge Functions

1. Ajoutez l'URL de votre API comme secret dans Lovable:
   - Dans Lovable, allez dans Settings → Secrets
   - Ajoutez un nouveau secret: `PYTHON_API_URL`
   - Valeur: `https://votre-url.onrender.com`

2. Les edge functions sont déjà configurées pour utiliser ce secret

## 📡 Endpoints de l'API

### GET /
Informations sur l'API

### GET /health
Vérification de l'état de santé des modèles

### POST /predict/skin-cancer
Prédiction de cancer de la peau
- **Body**: Fichier image (form-data)
- **Response**: 
  ```json
  {
    "prediction": "healthy|melanoma|...",
    "confidence": 0.95,
    "details": {...}
  }
  ```

### POST /predict/diabetes
Prédiction de diabète
- **Body**: JSON avec les données patient
  ```json
  {
    "gender": 1,
    "age": 45,
    "hypertension": 0,
    "heart_disease": 0,
    "smoking_history": 2,
    "bmi": 28.5,
    "HbA1c_level": 5.8,
    "blood_glucose_level": 120
  }
  ```
- **Response**:
  ```json
  {
    "prediction": "positive|negative",
    "confidence": 0.75,
    "details": {...}
  }
  ```

## 🔧 Développement Local

Pour tester localement:

```bash
# Installer les dépendances
pip install -r requirements.txt

# Lancer le serveur
uvicorn main:app --reload

# L'API sera disponible sur http://localhost:8000
# Documentation interactive: http://localhost:8000/docs
```

## 💰 Coûts Render.com

- **Plan Starter** (gratuit): 
  - 750 heures/mois
  - L'application peut "dormir" après 15 minutes d'inactivité
  - Premier démarrage peut prendre 30-60 secondes

- **Plan Payant** (à partir de $7/mois):
  - Application toujours active
  - Meilleure performance
  - Plus de ressources

## 🔐 Sécurité

Pour ajouter une authentification par API key (optionnel):

1. Ajoutez une variable d'environnement sur Render: `API_KEY=votre_clé_secrète`

2. Modifiez `main.py` pour vérifier la clé dans les headers

## 🐛 Dépannage

### Les modèles ne se chargent pas
- Vérifiez que les fichiers `.h5` et `.pkl` sont bien dans le dossier `models/`
- Vérifiez les logs sur Render pour voir les erreurs de chargement

### Timeout lors des prédictions
- Le plan gratuit a des ressources limitées
- Considérez upgrader vers un plan payant pour de meilleures performances

### L'application "dort"
- C'est normal sur le plan gratuit après 15 minutes d'inactivité
- Le premier appel prendra plus de temps (cold start)
- Utilisez un service comme UptimeRobot pour "pinger" votre API et la garder active

## 📚 Documentation

- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [Render Documentation](https://render.com/docs)
- [TensorFlow Documentation](https://www.tensorflow.org/)
