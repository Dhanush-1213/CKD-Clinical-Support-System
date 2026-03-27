import os
import joblib
import shap
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from predict import transform_input

MODELS_DIR = "models"
PLOTS_DIR = "shap_plots"

os.makedirs(PLOTS_DIR, exist_ok=True)

best_model = joblib.load(f"{MODELS_DIR}/best_model.joblib")
target_encoder = joblib.load(f"{MODELS_DIR}/target_encoder.joblib")


def explain_single_patient(input_df: pd.DataFrame, output_path=f"{PLOTS_DIR}/single_patient_shap.png"):
    processed_df = transform_input(input_df)

    explainer = shap.TreeExplainer(best_model)
    shap_values = explainer.shap_values(processed_df)

    predicted_class_idx = int(best_model.predict(processed_df)[0])
    predicted_class_name = target_encoder.inverse_transform([predicted_class_idx])[0]

    plt.figure(figsize=(10, 6))

    # multiclass shap output handling
    if isinstance(shap_values, list):
        shap_for_class = shap_values[predicted_class_idx][0]
    else:
        shap_array = shap_values
        if len(shap_array.shape) == 3:
            # shape: (n_samples, n_features, n_classes)
            shap_for_class = shap_array[0, :, predicted_class_idx]
        else:
            shap_for_class = shap_array[0]

    feature_names = processed_df.columns.tolist()
    feature_values = processed_df.iloc[0].values

    shap.plots._waterfall.waterfall_legacy(
        expected_value=explainer.expected_value[predicted_class_idx] if isinstance(explainer.expected_value, (list, np.ndarray)) else explainer.expected_value,
        shap_values=shap_for_class,
        features=feature_values,
        feature_names=feature_names,
        show=False
    )

    plt.title(f"SHAP Explanation - Predicted Class: {predicted_class_name}")
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches="tight")
    plt.close()

    return output_path, predicted_class_name