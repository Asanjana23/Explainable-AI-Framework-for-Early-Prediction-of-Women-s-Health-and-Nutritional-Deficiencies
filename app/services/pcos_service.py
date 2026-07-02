import joblib
import pandas as pd

model = joblib.load("model/pcos_prediction_model.pkl")

features = [
    "Skin_darkening","Hair_growth","Weight_gain",
    "Lifestyle_Risk","Cycle_Irregular","Fast_food",
    "Pimples","Weight","BMI","Hair_loss",
    "Waist","Hip","BMI_Category"
]

def predict_pcos(data):

    df = pd.DataFrame([data], columns=features)

    # Use .values to avoid UserWarning about feature names
    pred = model.predict(df.values)[0]
    prob = model.predict_proba(df.values)[0]

    return pred, prob, dict(df.iloc[0])