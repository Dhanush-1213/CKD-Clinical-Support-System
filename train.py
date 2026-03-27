import os
import re
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    ConfusionMatrixDisplay
)
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder, label_binarize
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

# =========================
# CONFIG
# =========================
DATA_PATH = "dataset/kidney_disease_dataset.csv"
MODELS_DIR = "models"
PLOTS_DIR = "shap_plots"

TARGET_COL_CANDIDATES = ["classification", "target", "label", "ckd"]

os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(PLOTS_DIR, exist_ok=True)


# =========================
# HELPERS
# =========================
def find_target_column(df: pd.DataFrame) -> str:
    for col in TARGET_COL_CANDIDATES:
        if col in df.columns:
            return col
    raise ValueError(f"Target column not found. Expected one of: {TARGET_COL_CANDIDATES}")


def clean_column_names(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = (
        df.columns.astype(str)
        .str.strip()
        .str.lower()
        .str.replace(r"\s+", "_", regex=True)
    )
    return df


def normalize_text(value):
    if pd.isna(value):
        return np.nan
    value = str(value)
    value = re.sub(r"\t", "", value)
    value = re.sub(r"\s+", " ", value)
    value = value.strip().lower()
    if value in ["", "nan", "none", "?"]:
        return np.nan
    return value


def clean_string_values(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for col in df.columns:
        if df[col].dtype == "object":
            df[col] = df[col].apply(normalize_text)
    return df


def convert_possible_numeric(df: pd.DataFrame, exclude_cols=None) -> pd.DataFrame:
    if exclude_cols is None:
        exclude_cols = []

    df = df.copy()

    for col in df.columns:
        if col in exclude_cols:
            continue

        if df[col].dtype == "object":
            converted = pd.to_numeric(df[col], errors="coerce")
            original_non_null = df[col].dropna()
            converted_non_null = converted.dropna()

            if len(original_non_null) > 0:
                ratio = len(converted_non_null) / len(original_non_null)
                if ratio >= 0.8:
                    df[col] = converted

    return df


def group_target(label: str) -> str:
    mapping = {
        "no_disease": "low",
        "low_risk": "low",
        "moderate_risk": "moderate",
        "high_risk": "high",
        "severe_disease": "high"
    }

    if label not in mapping:
        raise ValueError(f"Unexpected target label found: {label}")

    return mapping[label]


def build_grouped_target(y: pd.Series):
    y = y.apply(normalize_text)

    print("Unique cleaned target values:")
    print(sorted(y.dropna().unique()))

    if y.isna().any():
        raise ValueError("Target column contains missing values. Clean the dataset first.")

    grouped_y = y.apply(group_target)

    print("\nGrouped target values:")
    print(sorted(grouped_y.unique()))

    target_encoder = LabelEncoder()
    y_encoded = target_encoder.fit_transform(grouped_y)

    print("\nTarget class mapping:")
    for idx, label in enumerate(target_encoder.classes_):
        print(f"{label} -> {idx}")

    return pd.Series(y_encoded, index=y.index), target_encoder


def split_columns(X: pd.DataFrame):
    categorical_cols = X.select_dtypes(include=["object"]).columns.tolist()
    numeric_cols = X.select_dtypes(exclude=["object"]).columns.tolist()
    return numeric_cols, categorical_cols


def fit_preprocessors(X_train: pd.DataFrame, numeric_cols, categorical_cols):
    num_imputer = SimpleImputer(strategy="median")
    cat_imputer = SimpleImputer(strategy="most_frequent")

    X_train_num = pd.DataFrame(index=X_train.index)
    X_train_cat = pd.DataFrame(index=X_train.index)

    if numeric_cols:
        X_train_num = pd.DataFrame(
            num_imputer.fit_transform(X_train[numeric_cols]),
            columns=numeric_cols,
            index=X_train.index
        )

    if categorical_cols:
        X_train_cat = pd.DataFrame(
            cat_imputer.fit_transform(X_train[categorical_cols]),
            columns=categorical_cols,
            index=X_train.index
        )

    encoders = {}
    for col in categorical_cols:
        le = LabelEncoder()
        X_train_cat[col] = le.fit_transform(X_train_cat[col].astype(str))
        encoders[col] = le

    X_train_processed = pd.concat([X_train_num, X_train_cat], axis=1)
    final_columns = numeric_cols + categorical_cols
    X_train_processed = X_train_processed[final_columns]

    preprocessors = {
        "num_imputer": num_imputer,
        "cat_imputer": cat_imputer,
        "encoders": encoders,
        "numeric_cols": numeric_cols,
        "categorical_cols": categorical_cols,
        "feature_columns": X_train_processed.columns.tolist()
    }

    return X_train_processed, preprocessors


def transform_data(X: pd.DataFrame, preprocessors: dict) -> pd.DataFrame:
    numeric_cols = preprocessors["numeric_cols"]
    categorical_cols = preprocessors["categorical_cols"]

    X = X.copy()

    X_num = pd.DataFrame(index=X.index)
    X_cat = pd.DataFrame(index=X.index)

    if numeric_cols:
        X_num = pd.DataFrame(
            preprocessors["num_imputer"].transform(X[numeric_cols]),
            columns=numeric_cols,
            index=X.index
        )

    if categorical_cols:
        X_cat = pd.DataFrame(
            preprocessors["cat_imputer"].transform(X[categorical_cols]),
            columns=categorical_cols,
            index=X.index
        )

        for col in categorical_cols:
            le = preprocessors["encoders"][col]
            fallback = le.classes_[0]
            X_cat[col] = X_cat[col].astype(str).apply(
                lambda x: x if x in le.classes_ else fallback
            )
            X_cat[col] = le.transform(X_cat[col])

    X_processed = pd.concat([X_num, X_cat], axis=1)
    X_processed = X_processed[preprocessors["feature_columns"]]
    return X_processed


def compute_multiclass_roc_auc(y_true, y_prob, num_classes):
    try:
        y_true_bin = label_binarize(y_true, classes=list(range(num_classes)))
        return roc_auc_score(y_true_bin, y_prob, multi_class="ovr", average="macro")
    except Exception:
        return np.nan


def evaluate_model(model_name, model, X_test, y_test, num_classes):
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)

    metrics = {
        "model": model_name,
        "accuracy": accuracy_score(y_test, y_pred),
        "precision_macro": precision_score(y_test, y_pred, average="macro", zero_division=0),
        "recall_macro": recall_score(y_test, y_pred, average="macro", zero_division=0),
        "f1_macro": f1_score(y_test, y_pred, average="macro", zero_division=0),
        "roc_auc_ovr_macro": compute_multiclass_roc_auc(y_test, y_prob, num_classes)
    }

    cm = confusion_matrix(y_test, y_pred)
    return metrics, cm


def save_confusion_matrix(cm, model_name, class_names):
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    fig, ax = plt.subplots(figsize=(8, 6))
    disp.plot(ax=ax, xticks_rotation=45, cmap="Blues", colorbar=False)
    plt.title(f"Confusion Matrix - {model_name}")
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, f"confusion_matrix_{model_name.lower()}.png"))
    plt.close()


def save_feature_importance(model, feature_names, model_name):
    if not hasattr(model, "feature_importances_"):
        return

    importance_df = pd.DataFrame({
        "feature": feature_names,
        "importance": model.feature_importances_
    }).sort_values(by="importance", ascending=False)

    importance_df.to_csv(
        os.path.join(MODELS_DIR, f"feature_importance_{model_name.lower()}.csv"),
        index=False
    )


# =========================
# MAIN
# =========================
def main():
    print("Loading dataset...")
    df = pd.read_csv(DATA_PATH)

    df = clean_column_names(df)
    df = clean_string_values(df)

    target_col = find_target_column(df)
    print(f"Target column detected: {target_col}")

    print("\nOriginal target distribution:")
    print(df[target_col].value_counts())

    df = convert_possible_numeric(df, exclude_cols=[target_col])

    X = df.drop(columns=[target_col])
    y, target_encoder = build_grouped_target(df[target_col])

    grouped_target_names = target_encoder.inverse_transform(y)
    print("\nGrouped class distribution:")
    print(pd.Series(grouped_target_names).value_counts())

    numeric_cols, categorical_cols = split_columns(X)

    print(f"\nNumeric columns: {len(numeric_cols)}")
    print(f"Categorical columns: {len(categorical_cols)}")
    print(f"Number of grouped classes: {len(target_encoder.classes_)}")

    print("\nTrain-test split...")
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    print("Fitting preprocessors...")
    X_train_processed, preprocessors = fit_preprocessors(X_train, numeric_cols, categorical_cols)
    X_test_processed = transform_data(X_test, preprocessors)

    num_classes = len(target_encoder.classes_)

    print("Training XGBoost...")
    xgb_model = XGBClassifier(
        objective="multi:softprob",
        num_class=num_classes,
        n_estimators=300,
        max_depth=5,
        learning_rate=0.03,
        subsample=0.9,
        colsample_bytree=0.9,
        eval_metric="mlogloss",
        random_state=42
    )
    xgb_model.fit(X_train_processed, y_train)

    print("Training LightGBM...")
    lgbm_model = LGBMClassifier(
        objective="multiclass",
        num_class=num_classes,
        n_estimators=300,
        max_depth=5,
        learning_rate=0.03,
        random_state=42
    )
    lgbm_model.fit(X_train_processed, y_train)

    print("\nEvaluating models...")
    xgb_metrics, xgb_cm = evaluate_model("XGBoost", xgb_model, X_test_processed, y_test, num_classes)
    lgbm_metrics, lgbm_cm = evaluate_model("LightGBM", lgbm_model, X_test_processed, y_test, num_classes)

    results_df = pd.DataFrame([xgb_metrics, lgbm_metrics])

    print("\nModel Comparison:")
    print(results_df)

    save_confusion_matrix(xgb_cm, "XGBoost", target_encoder.classes_)
    save_confusion_matrix(lgbm_cm, "LightGBM", target_encoder.classes_)

    save_feature_importance(xgb_model, preprocessors["feature_columns"], "XGBoost")
    save_feature_importance(lgbm_model, preprocessors["feature_columns"], "LightGBM")

    best_model_name = results_df.sort_values(
        by=["f1_macro", "accuracy"],
        ascending=False
    ).iloc[0]["model"]

    best_model = xgb_model if best_model_name == "XGBoost" else lgbm_model

    print(f"\nBest model selected: {best_model_name}")

    print("Saving models and preprocessors...")
    joblib.dump(xgb_model, os.path.join(MODELS_DIR, "xgb_model.joblib"))
    joblib.dump(lgbm_model, os.path.join(MODELS_DIR, "lgbm_model.joblib"))
    joblib.dump(best_model, os.path.join(MODELS_DIR, "best_model.joblib"))
    joblib.dump(preprocessors, os.path.join(MODELS_DIR, "preprocessors.joblib"))
    joblib.dump(target_encoder, os.path.join(MODELS_DIR, "target_encoder.joblib"))
    results_df.to_csv(os.path.join(MODELS_DIR, "model_metrics.csv"), index=False)

    print("\nTraining complete.")
    print("Saved:")
    print("- models/xgb_model.joblib")
    print("- models/lgbm_model.joblib")
    print("- models/best_model.joblib")
    print("- models/preprocessors.joblib")
    print("- models/target_encoder.joblib")
    print("- models/model_metrics.csv")


if __name__ == "__main__":
    main()