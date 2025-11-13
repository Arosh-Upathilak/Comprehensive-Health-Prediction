import streamlit as st
import pandas as pd
import joblib
from io import BytesIO

st.set_page_config(page_title="Model Predictor", page_icon="🤖", layout="centered")

st.title("🔮 Machine Learning Predictor")
st.subheader("Fill in your details for a comprehensive health prediction")

# --- Load model (cached so it only loads once) ---
@st.cache_resource
def load_model(path="best_model.pkl"):
    try:
        return joblib.load(path)
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

model = load_model("best_model.pkl")

# --- Inputs (initial/default values preserved) ---
# NOTE: we do not modify these values anywhere — validation only reads them.
name = st.text_input("Name", value="")
age = st.number_input("Age", min_value=0, max_value=120, value=25)
bmi = st.number_input("BMI", min_value=10.0, max_value=50.0, value=22.0, format="%.1f")

smoker = st.selectbox("Smoker", ["Yes", "No"])
living_environment = st.selectbox("Living Environment", ["Urban", "Rural"])
hereditary_conditions = st.selectbox("Hereditary Conditions", ["Stable", "Unstable"])
diet = st.selectbox("Follow a specific diet?", ["Yes", "No"])
activity = st.selectbox("Regular Physical Activity?", ["Yes", "No"])
schedule = st.selectbox("Regular sleeping schedule?", ["Yes", "No"])
alcohol = st.selectbox("Regular alcohol consumption?", ["Yes", "No"])
interaction = st.selectbox("Regular social interaction?", ["Yes", "No"])
supplements = st.selectbox("Take supplements?", ["Yes", "No"])
management = st.selectbox("Active mental health management?", ["Yes", "No"])
illness = st.number_input("Number of illnesses last year", min_value=0, max_value=100, value=0)

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
    "living_environment_urban",
    "hereditary_stable",
    "smoker",
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
    "living_environment_urban": living_env_val,
    "hereditary_stable": hereditary_val,
    "smoker": smoker_val,
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
if st.button("Predict"):
    # Step 1: validate (no changes to inputs)
    missing_fields = validate_inputs()
    if missing_fields:
        st.error(f"Please fill the required fields before predicting: {', '.join(missing_fields)}")
    elif model is None:
        st.error("Model not loaded. Check logs or file `best_model.pkl`.")
    else:
        try:
            pred = model.predict(X)
            result = pred[0] if hasattr(pred, "__iter__") else pred
            value = "High Risk" if result == 1 else "Low Risk"

            # Build result dataframe: include original inputs + prediction
            result_df = X.copy()
            result_df.insert(0, "Name", name)
            result_df["Prediction"] = value

            # Show result to user and provide download
            st.markdown("---")
            st.success(f"Prediction Result: {value}")

            excel_bytes = df_to_excel_bytes(result_df)
            st.download_button(
                label="📥 Download result as Excel",
                data=excel_bytes,
                file_name="prediction_result.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            )

        except Exception as e:
            st.error(f"Error during prediction: {e}")
            st.info(
                "Common causes: the model expects a different number/order of features than provided. "
                "If you trained with different column names/order, update `feature_columns` and the `row` mapping to match."
            )
