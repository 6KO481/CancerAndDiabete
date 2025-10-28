from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import numpy as np
import pickle
from PIL import Image
import io
import tensorflow as tf
from tensorflow.keras.models import load_model
import logging

# Configuration du logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Medical AI Prediction API")

# Configuration CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Chargement des modèles au démarrage
logger.info("Chargement des modèles...")

try:
    # Modèles de cancer de la peau
    model1 = load_model('models/cancer/model1.h5')
    model2 = load_model('models/cancer/model2.h5')
    logger.info("Modèles de cancer chargés avec succès")
except Exception as e:
    logger.error(f"Erreur lors du chargement des modèles de cancer: {e}")
    model1 = None
    model2 = None

try:
    # Modèle de diabète
    with open('models/diabete/xgb_model.pkl', 'rb') as f:
        diabetes_model = pickle.load(f)
    logger.info("Modèle de diabète chargé avec succès")
except Exception as e:
    logger.error(f"Erreur lors du chargement du modèle de diabète: {e}")
    diabetes_model = None

# Modèles Pydantic pour les requêtes
class DiabetesInput(BaseModel):
    gender: int  # 0 = Female, 1 = Male, 2 = Other
    age: float
    hypertension: int  # 0 = No, 1 = Yes
    heart_disease: int  # 0 = No, 1 = Yes
    smoking_history: int  # 0-5 encoded
    bmi: float
    HbA1c_level: float
    blood_glucose_level: float

class PredictionResponse(BaseModel):
    prediction: str
    confidence: float
    details: dict

@app.get("/")
def read_root():
    return {
        "message": "Medical AI Prediction API",
        "endpoints": {
            "/predict/skin-cancer": "POST - Prédiction de cancer de la peau (image)",
            "/predict/diabetes": "POST - Prédiction de diabète (données patient)"
        }
    }

@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "models": {
            "skin_cancer_model1": model1 is not None,
            "skin_cancer_model2": model2 is not None,
            "diabetes_model": diabetes_model is not None
        }
    }

@app.post("/predict/skin-cancer", response_model=PredictionResponse)
async def predict_skin_cancer(file: UploadFile = File(...)):
    """
    Prédiction de cancer de la peau à partir d'une image
    Implémentation basée sur cancer.py - cascade de deux modèles ViT
    """
    if model1 is None or model2 is None:
        raise HTTPException(status_code=503, detail="Modèles non disponibles")
    
    try:
        # Lire et prétraiter l'image
        contents = await file.read()
        image = Image.open(io.BytesIO(contents))
        
        # Convertir en RGB si nécessaire
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Redimensionner à 224x224 (requis par ViT)
        image = image.resize((224, 224))
        
        # Convertir en array et normaliser
        img_array = np.array(image) / 255.0
        img_array = np.expand_dims(img_array, axis=0)
        
        # Étape 1: Prédiction avec model1 (healthy vs malignant)
        pred1 = model1.predict(img_array, verbose=0)
        is_healthy = pred1[0][0] > 0.5
        
        if is_healthy:
            return PredictionResponse(
                prediction="healthy",
                confidence=float(pred1[0][0]),
                details={
                    "final_prediction": "healthy",
                    "model1_prediction": "healthy",
                    "model1_confidence": float(pred1[0][0]),
                    "description": "Peau saine - Aucune lésion détectée"
                }
            )
        
        # Étape 2: Classification détaillée avec model2
        pred2 = model2.predict(img_array, verbose=0)
        
        # Classes du model2
        classes = [
            'non_cancerous_lesion',
            'keratinocytes',
            'melanoma',
            'kaposi_sarcoma',
            'mycosis_fungoides'
        ]
        
        predicted_class_idx = np.argmax(pred2[0])
        predicted_class = classes[predicted_class_idx]
        confidence = float(pred2[0][predicted_class_idx])
        
        # Descriptions
        descriptions = {
            'non_cancerous_lesion': 'Lésion bénigne ou non-néoplasique',
            'keratinocytes': 'Carcinome kératinocytaire détecté',
            'melanoma': 'Mélanome détecté - Consultation urgente recommandée',
            'kaposi_sarcoma': 'Sarcome de Kaposi détecté',
            'mycosis_fungoides': 'Mycosis fungoïde détecté'
        }
        
        return PredictionResponse(
            prediction=predicted_class,
            confidence=confidence,
            details={
                "final_prediction": predicted_class,
                "model1_prediction": "malignant",
                "model1_confidence": float(pred1[0][0]),
                "model2_prediction": predicted_class,
                "model2_confidence": confidence,
                "all_probabilities": {cls: float(pred2[0][i]) for i, cls in enumerate(classes)},
                "description": descriptions[predicted_class]
            }
        )
        
    except Exception as e:
        logger.error(f"Erreur lors de la prédiction: {e}")
        raise HTTPException(status_code=500, detail=f"Erreur de prédiction: {str(e)}")

@app.post("/predict/diabetes", response_model=PredictionResponse)
async def predict_diabetes(data: DiabetesInput):
    """
    Prédiction de diabète à partir des données patient
    Utilise le modèle XGBoost entraîné dans train.ipynb
    """
    if diabetes_model is None:
        raise HTTPException(status_code=503, detail="Modèle de diabète non disponible")
    
    try:
        # Préparer les features dans le bon ordre
        features = np.array([[
            data.gender,
            data.age,
            data.hypertension,
            data.heart_disease,
            data.smoking_history,
            data.bmi,
            data.HbA1c_level,
            data.blood_glucose_level
        ]])
        
        # Prédiction
        prediction = diabetes_model.predict(features)[0]
        
        # Probabilités si disponible
        try:
            probabilities = diabetes_model.predict_proba(features)[0]
            confidence = float(probabilities[1]) if prediction == 1 else float(probabilities[0])
        except:
            # Si predict_proba n'est pas disponible, calculer un score simple
            risk_score = 0
            if data.age > 45: risk_score += 0.2
            if data.bmi > 30: risk_score += 0.25
            if data.hypertension == 1: risk_score += 0.15
            if data.heart_disease == 1: risk_score += 0.15
            if data.HbA1c_level > 6.5: risk_score += 0.15
            if data.blood_glucose_level > 140: risk_score += 0.1
            confidence = min(risk_score, 1.0)
        
        has_diabetes = prediction == 1
        risk_percentage = confidence * 100
        
        result = "positive" if has_diabetes else "negative"
        description = (
            f"Risque élevé de diabète (Score: {risk_percentage:.1f}%)" 
            if has_diabetes 
            else f"Risque faible de diabète (Score: {risk_percentage:.1f}%)"
        )
        
        return PredictionResponse(
            prediction=result,
            confidence=confidence,
            details={
                "has_diabetes": has_diabetes,
                "risk_score": risk_percentage,
                "patient_data": data.dict(),
                "description": description
            }
        )
        
    except Exception as e:
        logger.error(f"Erreur lors de la prédiction de diabète: {e}")
        raise HTTPException(status_code=500, detail=f"Erreur de prédiction: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
