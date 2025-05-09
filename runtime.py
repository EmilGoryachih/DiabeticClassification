# -------------------- runtime.py --------------------
import joblib, numpy as np, pandas as pd

clf    = joblib.load("models/diabetes_xgb_calibrated.joblib")
THRESH = float(np.load("models/opt_threshold.npy")[0])

def predict_diabetes(record: dict) -> dict:
    """
    record:
    {
      "age": float, "bmi": float, "HbA1c_level": float, "blood_glucose_level": float,
      "smoking_history": str,     # e.g. "never", "current"
      "is_male": int, "hypertension": int, "heart_disease": int
    }
    """
    df = pd.DataFrame([record])
    p  = clf.predict_proba(df)[0, 1]
    return {
        "probability": round(float(p), 4),
        "prediction":  int(p > THRESH),
        "threshold":   round(float(THRESH), 4)
    }

# Example sanity-check
if __name__ == "__main__":
    healthy = {
        "age": 45, "bmi": 24.3, "HbA1c_level": 5.2, "blood_glucose_level": 110,
        "smoking_history": "never", "is_male": 1, "hypertension": 0, "heart_disease": 0
    }
    diabetic = {
        "age": 44, "bmi": 19.3, "HbA1c_level": 6.6, "blood_glucose_level": 200,
        "smoking_history": "never", "is_male": 0, "hypertension": 0, "heart_disease": 0
    }
    print("Healthy :", predict_diabetes(healthy))
    print("Diabetic:", predict_diabetes(diabetic))
