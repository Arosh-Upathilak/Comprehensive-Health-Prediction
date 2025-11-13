import streamlit as st
import pandas as pd
import joblib
from io import BytesIO
from sklearn.preprocessing import StandardScaler
import numpy as np

st.set_page_config(page_title="Health Predictor", page_icon="🏥", layout="wide")

# Custom CSS
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 10px;
        text-align: center;
        margin-bottom: 2rem;
        color: white;
    }
    .input-section {
        background: #f8f9fa;
        padding: 1.5rem;
        border-radius: 10px;
        margin: 1rem 0;
        border-left: 4px solid #667eea;
    }
    .result-box {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 15px;
        text-align: center;
        color: white;
        margin: 2rem 0;
    }
    .healthy {
        background: linear-gradient(135deg, #4CAF50 0%, #45a049 100%) !important;
    }
    .not-healthy {
        background: linear-gradient(135deg, #f44336 0%, #da190b 100%) !important;
    }
    .stButton > button {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        padding: 0.75rem 2rem;
        border-radius: 25px;
        font-weight: bold;
        transition: all 0.3s;
    }
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 5px 15px rgba(102, 126, 234, 0.4);
    }
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-header"><h1>🏥 Health Prediction System</h1><p>Advanced AI-powered health assessment tool</p></div>', unsafe_allow_html=True)

# --- Load model (cached so it only loads once) ---
@st.cache_resource
def load_model(path="best_model.pkl"):
    try:
        return joblib.load(path)
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

model = load_model("best_model.pkl")

# --- Inputs ---
col1, col2 = st.columns(2)

with col1:
    st.markdown('<div class="input-section"><h3>📋 Personal Information</h3></div>', unsafe_allow_html=True)
    name = st.text_input("👤 Full Name", value="")
    age = st.number_input("🎂 Age", min_value=0, max_value=120, value=25)
    bmi = st.number_input("⚖️ BMI", min_value=10.0, max_value=50.0, value=22.0, format="%.3f")
    
    st.markdown('<div class="input-section"><h3>🏠 Environment & Genetics</h3></div>', unsafe_allow_html=True)
    smoker = st.selectbox("🚬 Smoker", ["No", "Yes"])
    living_environment = st.selectbox("🌆 Living Environment", ["Urban", "Rural"])
    hereditary_conditions = st.selectbox("🧬 Hereditary Conditions", ["Stable", "Unstable"])
    illness = st.number_input("🤒 Illnesses last year", min_value=0, max_value=100, value=0)

with col2:
    st.markdown('<div class="input-section"><h3>💪 Lifestyle Habits</h3></div>', unsafe_allow_html=True)
    diet = st.selectbox("🥗 Follow specific diet?", ["No", "Yes"])
    activity = st.selectbox("🏃 Regular Physical Activity?", ["No", "Yes"])
    schedule = st.selectbox("😴 Regular sleeping schedule?", ["No", "Yes"])
    alcohol = st.selectbox("🍷 Regular alcohol consumption?", ["No", "Yes"])
    
    st.markdown('<div class="input-section"><h3>🧠 Mental & Social Health</h3></div>', unsafe_allow_html=True)
    interaction = st.selectbox("👥 Regular social interaction?", ["No", "Yes"])
    supplements = st.selectbox("💊 Take supplements?", ["No", "Yes"])
    management = st.selectbox("🧘 Active mental health management?", ["No", "Yes"])

# --- Encode categorical answers to numeric values (read-only) ---
def yesno_to_int(x):
    return 1 if str(x).strip().lower() == "yes" else 0

living_env_val = 1 if living_environment.lower() == "urban" else 0
hereditary_val = 1 if hereditary_conditions.lower() == "stable" else 0

smoker_val = yesno_to_int(smoker)
diet_val = yesno_to_int(diet)
activity_val = yesno_to_int(activity)
schedule_val = yesno_to_int(schedule)
alcohol_val = yesno_to_int(alcohol)
interaction_val = yesno_to_int(interaction)
supplements_val = yesno_to_int(supplements)
management_val = yesno_to_int(management)
illness_val = int(illness)
bmi_val = float(bmi)
age_val = int(age)

# --- Build DataFrame using a mapping (guarantees correct column→value alignment) ---
feature_columns = [
    "age",
    "bmi",
    "smoker",
    "living_environment_urban",
    "hereditary_stable",
    "diet",
    "physical_activity",
    "sleep_schedule",
    "alcohol",
    "social_interaction",
    "supplements",
    "mental_health_management",
    "illness_count_last_year",
]

row = {
    "age": age_val,
    "bmi": bmi_val,
    "smoker": smoker_val,
    "living_environment_urban": living_env_val,
    "hereditary_stable": hereditary_val,
    "diet": diet_val,
    "physical_activity": activity_val,
    "sleep_schedule": schedule_val,
    "alcohol": alcohol_val,
    "social_interaction": interaction_val,
    "supplements": supplements_val,
    "mental_health_management": management_val,
    "illness_count_last_year": illness_val,
}

# Create DataFrame in the exact order defined by feature_columns
X = pd.DataFrame([row], columns=feature_columns)

# Apply StandardScaler with reasonable defaults for health data
# Since we can't fit on single samples, we'll use approximate scaling
scaler = StandardScaler()
# Create a dummy dataset with reasonable ranges for health data to fit the scaler
dummy_data = pd.DataFrame({
    'age': [20, 30, 40, 50, 60, 70, 80],
    'bmi': [18, 22, 25, 28, 32, 35, 40],
    'smoker': [0, 1, 0, 1, 0, 1, 0],
    'living_environment_urban': [0, 1, 0, 1, 0, 1, 0],
    'hereditary_stable': [0, 1, 0, 1, 0, 1, 0],
    'diet': [0, 1, 0, 1, 0, 1, 0],
    'physical_activity': [0, 1, 0, 1, 0, 1, 0],
    'sleep_schedule': [0, 1, 0, 1, 0, 1, 0],
    'alcohol': [0, 1, 0, 1, 0, 1, 0],
    'social_interaction': [0, 1, 0, 1, 0, 1, 0],
    'supplements': [0, 1, 0, 1, 0, 1, 0],
    'mental_health_management': [0, 1, 0, 1, 0, 1, 0],
    'illness_count_last_year': [0, 1, 2, 3, 4, 5, 6]
})
scaler.fit(dummy_data)
X_scaled = scaler.transform(X)

# Helper: convert DataFrame to Excel bytes (try openpyxl first, fallback to xlsxwriter)
def df_to_excel_bytes(df: pd.DataFrame) -> bytes:
    buffer = BytesIO()
    try:
        with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
            df.to_excel(writer, index=False, sheet_name="Prediction")
    except Exception:
        with pd.ExcelWriter(buffer, engine="xlsxwriter") as writer:
            df.to_excel(writer, index=False, sheet_name="Prediction")
    buffer.seek(0)
    return buffer.getvalue()

# --- Validation before prediction ---
def validate_inputs() -> list:
    """Return list of missing/invalid field names (empty list = all good).
    This function only checks values and DOES NOT modify any inputs or defaults.
    """
    missing = []
    # require name to be filled
    if not str(name).strip():
        missing.append("Name")
    # Example: if you want to require non-default BMI or Age, add checks here
    # (currently we keep the initial defaults allowed and do not change them)
    return missing

# --- Prediction ---
st.markdown("<br>", unsafe_allow_html=True)
col1, col2, col3 = st.columns([1,2,1])
with col2:
    if st.button("🔮 Predict Health Status", use_container_width=True):
        # Step 1: validate (no changes to inputs)
        missing_fields = validate_inputs()
        if missing_fields:
            st.error(f"Please fill the required fields before predicting: {', '.join(missing_fields)}")
        elif model is None:
            st.error("Model not loaded. Check logs or file `best_model.pkl`.")
        else:
            try:
                pred = model.predict(X_scaled)
                result = pred[0] if hasattr(pred, "__iter__") else pred
                value = "Healthy" if result == 1 else "Not Healthy"

                # Build result dataframe: include original inputs + prediction
                result_df = pd.DataFrame([row], columns=feature_columns)
                result_df.insert(0, "Name", name)
                result_df["Prediction"] = value

                # Show result to user and provide download
                result_class = "healthy" if value == "Healthy" else "not-healthy"
                st.markdown(f'<div class="result-box {result_class}"><h2>🎯 Prediction Result</h2><h1>{value}</h1></div>', unsafe_allow_html=True)

                excel_bytes = df_to_excel_bytes(result_df)
                col1, col2, col3 = st.columns([1,2,1])
                with col2:
                    st.download_button(
                        label="📥 Download Detailed Report",
                        data=excel_bytes,
                        file_name="health_prediction_report.xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                        use_container_width=True
                    )

            except Exception as e:
                st.markdown('<div class="result-box not-healthy"><h2>❌ Prediction Error</h2><p>Unable to process your request</p></div>', unsafe_allow_html=True)
                with st.expander("🔧 Technical Details"):
                    st.error(f"Error: {e}")
                    st.write(f"Input shape: {X.shape}")
                    st.write(f"Scaled input shape: {X_scaled.shape}")
                    st.write(f"Feature values: {X.iloc[0].to_dict()}")
                    st.info("The model may expect different features. Please check the model configuration.")