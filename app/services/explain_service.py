import shap
import joblib
import pandas as pd
import numpy as np

# Load models for explanation
anemia_model = joblib.load("model/anemia_model.pkl")
pcos_model = joblib.load("model/pcos_prediction_model.pkl")

# Define features for each model to ensure consistency
ANEMIA_FEATURES = ["Hb", "MCV", "MCH", "MCHC", "Gender"]
PCOS_FEATURES = [
    "Skin_darkening", "Hair_growth", "Weight_gain",
    "Lifestyle_Risk", "Cycle_Irregular", "Fast_food",
    "Pimples", "Weight", "BMI", "Hair_loss",
    "Waist", "Hip", "BMI_Category"
]

def get_top_contributors(model, features_dict, feature_names, top_k=3):
    """Calculates SHAP values and returns the top k contributing features."""
    try:
        # Create DataFrame with correct feature order
        df = pd.DataFrame([features_dict], columns=feature_names)
        
        # Initialize SHAP explainer
        # For tree-based models (like RandomForest/XGBoost usually saved in pkl), Explainer works well
        explainer = shap.Explainer(model)
        shap_values = explainer(df)
        
        # Extract values for the first (and only) row
        # shap_values.values shape is (n_samples, n_features) or (n_samples, n_features, n_classes)
        vals = shap_values.values[0]
        
        # If multi-class/output, we focus on the positive class or the predicted class
        if len(vals.shape) > 1:
            # Get index of predicted class
            pred_class = int(model.predict(df.values)[0])
            vals = vals[:, pred_class]
            
        # Create a mapping of feature names to their SHAP values
        feature_importance = dict(zip(feature_names, vals))
        
        # Sort by absolute value to get most impactful features
        sorted_features = sorted(feature_importance.items(), key=lambda x: abs(x[1]), reverse=True)
        
        # Return top K as a list of strings
        return [f"{feat} ({'increases' if val > 0 else 'decreases'} risk)" for feat, val in sorted_features[:top_k]]
    except Exception as e:
        print(f"SHAP error: {e}")
        return []

def get_model_explanations(features):
    """Returns top contributors for both Anemia and PCOS models."""
    explanations = {
        "anemia": get_top_contributors(anemia_model, features, ANEMIA_FEATURES),
        "pcos": get_top_contributors(pcos_model, features, PCOS_FEATURES)
    }
    return explanations