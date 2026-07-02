from fastapi import APIRouter, HTTPException
import numpy as np
from app.services.anemia_service import predict_anemia
from app.services.pcos_service import predict_pcos
from app.services.risk_service import get_risk
from app.services.explanation_service import generate_explanation

router = APIRouter()

@router.post("/predict")
def predict(features: dict):
    try:
        if not isinstance(features, dict):
            raise ValueError("Features must be a JSON object")

        # Basic data cleaning: fill missing with reasonable defaults for model safety
        cleaned_features = features.copy()
        defaults = {
            "Hb": 12.0, "MCV": 85.0, "MCH": 28.0, "MCHC": 33.0,
            "Skin_darkening": 0, "Hair_growth": 0, "Weight_gain": 0,
            "Lifestyle_Risk": 0, "Cycle_Irregular": 0, "Fast_food": 0,
            "Pimples": 0, "Weight": 60.0, "BMI": 22.0, "Hair_loss": 0,
            "Waist": 80.0, "Hip": 95.0, "BMI_Category": 0, "Gender": 0
        }
        for k, v in defaults.items():
            if cleaned_features.get(k) is None:
                cleaned_features[k] = v

        a_pred, a_prob, _ = predict_anemia(cleaned_features)
        p_pred, p_prob, _ = predict_pcos(cleaned_features)
        
        # Calculate confidence
        a_conf = int(max(a_prob) * 100)
        p_conf = int(max(p_prob) * 100)
        
        anemia_res = {
            "prediction": int(a_pred),
            "confidence": f"{a_conf}%",
            "confidence_val": a_conf,
            "risk": get_risk(a_prob)
        }
        pcos_res = {
            "prediction": int(p_pred),
            "confidence": f"{p_conf}%",
            "confidence_val": p_conf,
            "risk": get_risk(p_prob)
        }
        
        # Generate RAG-backed explanation
        explanation_data = generate_explanation(anemia_res, pcos_res, cleaned_features)
        
        return {
            "anemia": anemia_res,
            "pcos": pcos_res,
            "explanation": explanation_data.get("explanation", ""),
            "dietary_recommendations": explanation_data.get("dietary_recommendations", []),
            "lifestyle_recommendations": explanation_data.get("lifestyle_recommendations", [])
        }
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=400, detail=f"Prediction error: {str(e)}")
