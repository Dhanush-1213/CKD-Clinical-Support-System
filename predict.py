import joblib
import pandas as pd
import numpy as np

MODELS_DIR = "models"

best_model = joblib.load(f"{MODELS_DIR}/best_model.joblib")
preprocessors = joblib.load(f"{MODELS_DIR}/preprocessors.joblib")
target_encoder = joblib.load(f"{MODELS_DIR}/target_encoder.joblib")


def transform_input(input_df: pd.DataFrame) -> pd.DataFrame:
    input_df = input_df.copy()
    input_df.columns = input_df.columns.str.strip().str.lower().str.replace(" ", "_")

    alias_map = {
        "bp": "blood_pressure",
        "sc": "serum_creatinine",
        "bu": "blood_urea",
        "hemo": "hemoglobin",
        "al": "albumin",
        "su": "sugar",
        "htn": "hypertension",
        "dm": "diabetes_mellitus"
    }

    input_df = input_df.rename(columns=alias_map)

    numeric_cols = preprocessors["numeric_cols"]
    categorical_cols = preprocessors["categorical_cols"]

    for col in numeric_cols + categorical_cols:
        if col not in input_df.columns:
            input_df[col] = np.nan

    input_df = input_df[numeric_cols + categorical_cols]

    for col in numeric_cols:
        input_df[col] = pd.to_numeric(input_df[col], errors="coerce")

    for col in categorical_cols:
        input_df[col] = input_df[col].astype(str).str.strip().str.lower()

    X_num = pd.DataFrame(
        preprocessors["num_imputer"].transform(input_df[numeric_cols]),
        columns=numeric_cols,
        index=input_df.index
    )

    X_cat = pd.DataFrame(
        preprocessors["cat_imputer"].transform(input_df[categorical_cols]),
        columns=categorical_cols,
        index=input_df.index
    )

    for col in categorical_cols:
        le = preprocessors["encoders"][col]
        fallback = le.classes_[0]
        X_cat[col] = X_cat[col].apply(lambda x: x if x in le.classes_ else fallback)
        X_cat[col] = le.transform(X_cat[col])

    X_processed = pd.concat([X_num, X_cat], axis=1)
    X_processed = X_processed[preprocessors["feature_columns"]]

    return X_processed


def get_ckd_stage(egfr: float) -> str:
    if pd.isna(egfr):
        return "Unknown"
    if egfr >= 90:
        return "Stage 1"
    elif egfr >= 60:
        return "Stage 2"
    elif egfr >= 45:
        return "Stage 3a"
    elif egfr >= 30:
        return "Stage 3b"
    elif egfr >= 15:
        return "Stage 4"
    return "Stage 5"


def predict_patient(input_df: pd.DataFrame) -> pd.DataFrame:
    processed_df = transform_input(input_df)

    probabilities = best_model.predict_proba(processed_df)
    predictions = best_model.predict(processed_df)
    predicted_labels = target_encoder.inverse_transform(predictions)

    results = []

    for i in range(len(input_df)):
        class_probs = probabilities[i]
        confidence = float(np.max(class_probs))

        egfr = pd.to_numeric(input_df.iloc[i].get("egfr", np.nan), errors="coerce")

        results.append({
            "predicted_class": predicted_labels[i],
            "confidence": round(confidence, 4),
            "prob_low": round(float(class_probs[0]), 4),
            "prob_moderate": round(float(class_probs[1]), 4),
            "prob_high": round(float(class_probs[2]), 4),
            "ckd_stage": get_ckd_stage(egfr),
            "recommendation": "Check clinical report"
        })

    return pd.DataFrame(results)