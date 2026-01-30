import streamlit as st
import pandas as pd
import joblib
from pathlib import Path

st.set_page_config(page_title="Churn Prediction", page_icon="📉", layout="centered")

MODEL_PATH = "decision_tree_Churn_model.joblib"

@st.cache_resource
def load_model(path: str):
    return joblib.load(path)

st.title("📉 Prediksi Churn (Decision Tree)")
st.caption("Input 1 pelanggan → model akan encode otomatis → keluarkan prediksi.")

if not Path(MODEL_PATH).exists():
    st.error(
        f"File model tidak ditemukan: {MODEL_PATH}\n\n"
        "Pastikan `decision_tree_Churn_model.joblib` satu folder dengan `app.py`."
    )
    st.stop()

model = load_model(MODEL_PATH)

with st.form("form"):
    st.subheader("Input Data Pelanggan")

    CustomerID = st.text_input("CustomerID", value="1001")

    col1, col2 = st.columns(2)

    with col1:
        Age = st.number_input("Age", min_value=0, max_value=120, value=30, step=1)
        Tenure = st.number_input("Tenure", min_value=0, value=12, step=1)
        Usage_Frequency = st.number_input("Usage Frequency", min_value=0, value=10, step=1)
        Support_Calls = st.number_input("Support Calls", min_value=0, value=1, step=1)
        Payment_Delay = st.number_input("Payment Delay", min_value=0, value=0, step=1)

    with col2:
        Gender = st.selectbox("Gender", ["Male", "Female", "Other"])
        Subscription_Type = st.selectbox("Subscription Type", ["Basic", "Standard", "Premium"])
        Contract_Length = st.selectbox("Contract Length", ["Monthly", "Quarterly", "Yearly"])
        Total_Spend = st.number_input("Total Spend", min_value=0.0, value=100.0, step=10.0)
        Last_Interaction = st.number_input("Last Interaction", min_value=0, value=7, step=1)

    submitted = st.form_submit_button("Predict")

if submitted:
    # Karena training kamu memasukkan CustomerID ke fitur, kita coba convert ke angka.
    # Kalau sebenarnya CustomerID di dataset kamu sudah numeric, ini aman.
    try:
        customer_id_num = float(CustomerID)
    except Exception:
        customer_id_num = 0.0

    X_input = pd.DataFrame([{
        "CustomerID": customer_id_num,
        "Age": Age,
        "Gender": Gender,
        "Tenure": Tenure,
        "Usage Frequency": Usage_Frequency,
        "Support Calls": Support_Calls,
        "Payment Delay": Payment_Delay,
        "Subscription Type": Subscription_Type,
        "Contract Length": Contract_Length,
        "Total Spend": Total_Spend,
        "Last Interaction": Last_Interaction,
    }])

    try:
        pred = model.predict(X_input)[0]
        proba = None
        if hasattr(model, "predict_proba"):
            proba = float(model.predict_proba(X_input)[0][1])  # asumsi churn=1
    except Exception as e:
        st.error("Gagal prediksi. Ada mismatch kolom / tipe data.")
        st.exception(e)
        st.stop()

    st.divider()
    st.subheader("Hasil Prediksi")

    churn_yes = int(pred) == 1 if str(pred).isdigit() else (str(pred).lower() in ["1", "true", "yes", "churn"])

    if churn_yes:
        st.error("🚨 Prediksi: **CHURN**")
    else:
        st.success("✅ Prediksi: **TIDAK CHURN**")

    if proba is not None:
        st.metric("Probabilitas Churn", f"{proba*100:.2f}%")

    with st.expander("Lihat input yang dikirim ke model"):
        st.dataframe(X_input, use_container_width=True)
