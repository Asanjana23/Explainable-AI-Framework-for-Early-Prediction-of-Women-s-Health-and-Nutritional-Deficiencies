import joblib
import pandas as pd

model = joblib.load("model/anemia_model.pkl")

features = ["Hb", "MCV", "MCH", "MCHC", "Gender"]

def predict_anemia(data):

    df = pd.DataFrame([data], columns=features)

    # Use .values to avoid UserWarning about feature names
    pred = model.predict(df.values)[0]
    prob = model.predict_proba(df.values)[0]

    return pred, prob, dict(df.iloc[0])