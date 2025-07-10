import streamlit as st
import pandas as pd
import joblib

# ✅ Must be first Streamlit command
st.set_page_config(page_title="Parkinson Disease Detector (Scikit-Learn)", layout="centered")

# Load model and scaler
@st.cache_resource
def load_model_scaler():
    model = joblib.load("model/model.pkl")
    scaler = joblib.load("model/scaler.pkl")
    return model, scaler

model, scaler = load_model_scaler()

st.title("🧠 Parkinson Disease Detector")
st.markdown("Upload a CSV file with 22 features (without `name` and `status`).")

uploaded_file = st.file_uploader("Upload CSV", type=["csv"])

if uploaded_file:
    try:
        input_df = pd.read_csv(uploaded_file)
        scaled = scaler.transform(input_df)
        preds = model.predict(scaled)
        probs = model.predict_proba(scaled)[:, 1]
        input_df["Prediction"] = ["Parkinson" if p == 1 else "Healthy" for p in preds]
        input_df["Confidence"] = (probs * 100).round(2)
        
        st.success("✅ Prediction done!")
        st.dataframe(input_df)

        # 🥧 Pie & 📊 Bar Charts
        count_data = input_df["Prediction"].value_counts()

        st.markdown("### 🥧 Prediction Distribution (Pie Chart)")
        st.plotly_chart({
            "data": [{
                "labels": count_data.index.tolist(),
                "values": count_data.values.tolist(),
                "type": "pie",
                "hole": 0.3
            }],
            "layout": {"margin": {"l": 0, "r": 0, "t": 30, "b": 0}}
        })

        st.markdown("### 📊 Prediction Count (Bar Graph)")
        st.bar_chart(count_data)

    except Exception as e:
        st.error(f"❌ Error: {e}")
