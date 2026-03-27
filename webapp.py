import os
import pandas as pd
import streamlit as st
from fpdf import FPDF

from predict import predict_patient
from explain_single import explain_single_patient

st.set_page_config(page_title="CKD Clinical Support System", layout="wide")

os.makedirs("reports", exist_ok=True)
os.makedirs("shap_plots", exist_ok=True)


def generate_pdf_report(patient_data: dict, prediction_result: dict, pdf_path: str):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_auto_page_break(auto=True, margin=15)

    pdf.set_font("Arial", "B", 16)
    pdf.cell(200, 10, "CKD Clinical Support Report", ln=True, align="C")
    pdf.ln(10)

    pdf.set_font("Arial", "B", 12)
    pdf.cell(200, 10, "Patient Inputs", ln=True)
    pdf.set_font("Arial", "", 11)

    for key, value in patient_data.items():
        pdf.multi_cell(0, 8, f"{key}: {value}")

    pdf.ln(5)
    pdf.set_font("Arial", "B", 12)
    pdf.cell(200, 10, "Prediction Results", ln=True)
    pdf.set_font("Arial", "", 11)

    for key, value in prediction_result.items():
        pdf.multi_cell(0, 8, f"{key}: {value}")

    pdf.output(pdf_path)


st.title("CKD Clinical Support System")
st.write(
    "Predict grouped CKD risk level, estimate CKD stage using eGFR, "
    "generate recommendations, and view SHAP-based explanation."
)

tab1, tab2 = st.tabs(["Single Patient Prediction", "Bulk CSV Prediction"])

# =========================
# TAB 1 - SINGLE PATIENT
# =========================
with tab1:
    st.subheader("Single Patient Entry")

    col1, col2, col3 = st.columns(3)

    with col1:
        age = st.number_input("Age", min_value=1, max_value=120, value=45)
        blood_pressure = st.number_input("Blood Pressure", min_value=50, max_value=250, value=80)
        serum_creatinine = st.number_input("Serum Creatinine", min_value=0.0, value=1.2, step=0.1)
        blood_urea = st.number_input("Blood Urea", min_value=0.0, value=40.0, step=0.1)

    with col2:
        hemoglobin = st.number_input("Hemoglobin", min_value=0.0, value=12.0, step=0.1)
        albumin = st.selectbox("Albumin", [0, 1, 2, 3, 4, 5], index=1)
        sugar = st.selectbox("Sugar", [0, 1, 2, 3, 4, 5], index=0)
        egfr = st.number_input("eGFR", min_value=0.0, value=65.0, step=0.1)

    with col3:
        sodium = st.number_input("Sodium", min_value=0.0, value=135.0, step=0.1)
        potassium = st.number_input("Potassium", min_value=0.0, value=4.5, step=0.1)
        hypertension = st.selectbox("Hypertension", ["yes", "no"], index=1)
        diabetes_mellitus = st.selectbox("Diabetes Mellitus", ["yes", "no"], index=1)

    if st.button("Predict Single Patient", key="single_predict"):
        input_data = pd.DataFrame([{
            "age": age,
            "blood_pressure": blood_pressure,
            "serum_creatinine": serum_creatinine,
            "blood_urea": blood_urea,
            "hemoglobin": hemoglobin,
            "albumin": albumin,
            "sugar": sugar,
            "egfr": egfr,
            "sodium": sodium,
            "potassium": potassium,
            "hypertension": hypertension,
            "diabetes_mellitus": diabetes_mellitus
        }])

        try:
            result_df = predict_patient(input_data)
            result = result_df.iloc[0].to_dict()

            st.success("Prediction completed successfully.")

            c1, c2, c3 = st.columns(3)
            c1.metric("Predicted Risk Class", str(result["predicted_class"]).title())
            c2.metric("Model Confidence", result["confidence"])
            c3.metric("CKD Stage", result["ckd_stage"])

            st.write("### Class Probabilities")
            prob_df = pd.DataFrame({
                "Risk Class": ["Low", "Moderate", "High"],
                "Probability": [
                    result.get("prob_low", 0.0),
                    result.get("prob_moderate", 0.0),
                    result.get("prob_high", 0.0)
                ]
            })
            st.dataframe(prob_df, use_container_width=True)

            st.write("### Clinical Recommendation")
            st.info(result["recommendation"])

            shap_path, shap_class = explain_single_patient(input_data)
            st.write("### Local Explanation")
            st.caption(f"Explanation shown for predicted class: {str(shap_class).title()}")
            st.image(shap_path, use_container_width=True)

            pdf_path = "reports/single_patient_report.pdf"
            pdf_result = {
                "predicted_class": str(result["predicted_class"]).title(),
                "confidence": result["confidence"],
                "prob_low": result.get("prob_low", 0.0),
                "prob_moderate": result.get("prob_moderate", 0.0),
                "prob_high": result.get("prob_high", 0.0),
                "ckd_stage": result["ckd_stage"],
                "recommendation": result["recommendation"]
            }
            generate_pdf_report(input_data.iloc[0].to_dict(), pdf_result, pdf_path)

            with open(pdf_path, "rb") as f:
                st.download_button(
                    label="Download PDF Report",
                    data=f,
                    file_name="single_patient_report.pdf",
                    mime="application/pdf"
                )

        except Exception as e:
            st.error(f"Error: {e}")

# =========================
# TAB 2 - BULK CSV
# =========================
with tab2:
    st.subheader("Bulk CSV Upload")

    uploaded_file = st.file_uploader("Upload CSV file", type=["csv"], key="bulk_csv")

    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)

            st.write("### Uploaded Data Preview")
            st.dataframe(df.head(), use_container_width=True)

            st.write("### Uploaded Columns")
            st.write(list(df.columns))

            if st.button("Predict Bulk Data", key="bulk_predict"):
                result_df = predict_patient(df)
                final_df = pd.concat(
                    [df.reset_index(drop=True), result_df.reset_index(drop=True)],
                    axis=1
                )
                st.session_state["bulk_results"] = final_df

            if "bulk_results" in st.session_state:
                bulk_results = st.session_state["bulk_results"]

                st.success("Bulk prediction completed successfully.")

                st.write("## Patient-wise Prediction Results")

                for idx, row in bulk_results.iterrows():
                    with st.container():
                        st.markdown(f"### Patient {idx + 1}")

                        c1, c2, c3 = st.columns(3)
                        c1.metric("Predicted Risk Class", str(row["predicted_class"]).title())
                        c2.metric("Model Confidence", float(row["confidence"]))
                        c3.metric("CKD Stage", str(row["ckd_stage"]))

                        st.write("#### Class Probabilities")
                        prob_df = pd.DataFrame({
                            "Risk Class": ["Low", "Moderate", "High"],
                            "Probability": [
                                row.get("prob_low", 0.0),
                                row.get("prob_moderate", 0.0),
                                row.get("prob_high", 0.0)
                            ]
                        })
                        st.dataframe(prob_df, use_container_width=True)

                        st.write("#### Clinical Recommendation")
                        st.info(str(row["recommendation"]))

                        st.markdown("---")

                st.write("## Full Results Table")
                st.dataframe(bulk_results, use_container_width=True)

                csv_data = bulk_results.to_csv(index=False).encode("utf-8")
                st.download_button(
                    label="Download Results CSV",
                    data=csv_data,
                    file_name="bulk_predictions.csv",
                    mime="text/csv",
                    key="bulk_download"
                )

        except Exception as e:
            st.error(f"Bulk prediction error: {e}")