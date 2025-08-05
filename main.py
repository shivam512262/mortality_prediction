# main.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import numpy as np
import pickle
import shap # shap is needed for the explainer object, even if not directly plotting
import matplotlib.pyplot as plt # matplotlib might be implicitly used by shap, keep for safety
from fastapi.middleware.cors import CORSMiddleware # NEW: Import CORSMiddleware

# Define the features that your model expects.
# This list is crucial and should match the features your model was trained on.
# It is inferred from your provided dummy data, excluding 'RecordID' and 'Time'.
MODEL_FEATURES = [
    'Age', 'Gender', 'Height', 'Weight', 'Urine', 'HR', 'Temp', 'NIDiasABP',
    'SysABP', 'DiasABP', 'pH', 'PaCO2', 'PaO2', 'Platelets', 'MAP', 'K',
    'Na', 'FiO2', 'GCS', 'ICUType'
]

# --- Global Model Loading ---
# Load models and imputer globally when the app starts.
# This is a critical optimization to avoid reloading large files on every request,
# significantly improving API performance.
IMPUTER = None
LGBM_MODEL = None
LGBM_SHAP_EXPLAINER = None
XGB_MODEL = None
XGB_SHAP_EXPLAINER = None

try:
    IMPUTER = pickle.load(open('imputer.pkl', 'rb'))
    LGBM_MODEL = pickle.load(open('lgbm_model.pkl', 'rb'))
    LGBM_SHAP_EXPLAINER = pickle.load(open('lgbm_shap_explainer.pkl', 'rb'))
    # Assuming you might also have XGBoost models based on your original code structure
    # If you only use LGBM, you can remove the XGBoost related lines.
    # We will try to load XGBoost models, but the app will still run if they are missing.
    try:
        XGB_MODEL = pickle.load(open('xgb_model.pkl', 'rb'))
        XGB_SHAP_EXPLAINER = pickle.load(open('xgb_shap_explainer.pkl', 'rb'))
        print("XGBoost models loaded successfully.")
    except FileNotFoundError:
        print("XGBoost models (xgb_model.pkl, xgb_shap_explainer.pkl) not found. XGBoost prediction will not be available.")

    print("Core models and imputer loaded successfully.")
except FileNotFoundError as e:
    print(f"Error loading essential pickle files: {e}. Please ensure 'imputer.pkl', 'lgbm_model.pkl', 'lgbm_shap_explainer.pkl' are present.")
    print("API will run, but prediction endpoints might fail if core models are missing.")
except Exception as e:
    print(f"An unexpected error occurred during model loading: {e}")

# Initialize FastAPI app
app = FastAPI(
    title="Mortality Prediction API",
    description="API for predicting patient mortality using LightGBM or XGBoost models.",
    version="1.0.0"
)

# --- CORS Configuration ---
# This is crucial for allowing your HTML frontend (served from file:// or a different domain)
# to make requests to your FastAPI backend.
origins = [
    "http://localhost",
    "http://localhost:8000",
    "http://127.0.0.1",
    "http://127.0.0.1:8000",
    "http://localhost:8001", # Allow requests from the CrewAI FastAPI
    "http://127.0.0.1:8001", # Allow requests from the CrewAI FastAPI
    "null", # Important for requests coming from a local file system (file://) in some browsers
    "*" # Temporarily allow all origins for debugging
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,       # Allows specified origins
    allow_credentials=True,      # Allows cookies to be included in cross-origin requests
    allow_methods=["*"],         # Allows all HTTP methods (GET, POST, PUT, DELETE, OPTIONS, etc.)
    allow_headers=["*"],         # Allows all headers
)

# --- Helper Function: extract_to_dict ---
# This function processes raw patient data into a DataFrame suitable for the model.
def extract_to_dict(df: pd.DataFrame, features: list) -> pd.DataFrame:
    """
    Extracts and aggregates patient parameters from a raw DataFrame into a dictionary
    of mean values for specified features, then converts it to a DataFrame.
    Handles missing parameters by setting their values to NaN.

    Args:
        df (pd.DataFrame): Raw patient data DataFrame with 'Parameter' and 'Value' columns.
        features (list): List of features (parameters) to extract.

    Returns:
        pd.DataFrame: A DataFrame with one row, containing the mean value for each feature.
    """
    df = df.replace(-1.0, np.nan) # Replace -1.0 with NaN for proper imputation
    patient_dict = {}
    for feat in features:
        # Check if the feature exists in the input DataFrame's 'Parameter' column
        if feat in df['Parameter'].values:
            # Calculate the mean value for the feature
            feature_mean = df[df['Parameter'] == feat]['Value'].mean()
            patient_dict[feat] = round(feature_mean, 2)
        else:
            # If a feature is not present in the raw data, set its value to NaN.
            # This allows the imputer to handle it during the transformation step.
            patient_dict[feat] = np.nan
    return pd.DataFrame(patient_dict, index=[0])

# --- Core Prediction Logic: predict_mortality_api ---
# This function encapsulates the model prediction and data processing logic.
def predict_mortality_api(raw_patient_df: pd.DataFrame, model_choice: str = 'lgbm'):
    """
    Predicts patient mortality based on raw patient data using a chosen model.

    Args:
        raw_patient_df (pd.DataFrame): DataFrame containing raw patient observations.
        model_choice (str): The choice of model ('lgbm' or 'xgb').

    Returns:
        dict: A dictionary containing the prediction outcome and probability of death.
    """
    # Ensure core models and imputer are loaded before proceeding
    if IMPUTER is None or LGBM_MODEL is None or LGBM_SHAP_EXPLAINER is None:
        raise HTTPException(status_code=500, detail="Essential model files (imputer, LGBM) not loaded on server startup. Please check server logs.")

    model = None
    explainer = None

    # Select the appropriate model and explainer based on user choice
    if model_choice == 'lgbm':
        model = LGBM_MODEL
        explainer = LGBM_SHAP_EXPLAINER
    elif model_choice == 'xgb':
        if XGB_MODEL is None or XGB_SHAP_EXPLAINER is None:
            raise HTTPException(status_code=400, detail="XGBoost models are not loaded. Please select LightGBM or ensure XGBoost models are available.")
        model = XGB_MODEL
        explainer = XGB_SHAP_EXPLAINER
    else:
        # Raise an error if an invalid model choice is provided
        raise HTTPException(status_code=400, detail="Invalid 'model_choice'. Must be 'lgbm' or 'xgb'.")

    # Process the raw patient data using the helper function
    # Ensure the resulting DataFrame has the correct model features in the correct order
    processed_df = extract_to_dict(raw_patient_df, MODEL_FEATURES)[MODEL_FEATURES]

    # Impute missing values using the pre-trained imputer
    imputed_data = IMPUTER.transform(processed_df)
    final_df = pd.DataFrame(imputed_data, columns=MODEL_FEATURES)

    # Make prediction and get probability
    prediction_class = model.predict(final_df)[0]
    prediction_proba = model.predict_proba(final_df)[0][1]
    outcome = "Predicted to Die" if prediction_class == 1 else "Predicted to Survive"

    return {
        "prediction": outcome,
        "probability_of_death": round(prediction_proba, 4), # Round probability for cleaner output
    }

# --- Pydantic Models for Request and Response ---
# These models define the expected structure of the JSON data sent to and received from the API.
class PatientRecord(BaseModel):
    """Defines the structure of a single patient observation."""
    Time: str
    Parameter: str
    Value: float

class PredictionRequest(BaseModel):
    """Defines the structure of the request body for mortality prediction."""
    patient_data: list[PatientRecord] # A list of patient observations
    model_choice: str = 'lgbm'      # The model to use, with 'lgbm' as default

class PredictionResponse(BaseModel):
    """Defines the structure of the response body for mortality prediction."""
    prediction: str
    probability_of_death: float

# --- API Endpoints ---
@app.post("/predict_mortality/", response_model=PredictionResponse)
async def get_mortality_prediction(request: PredictionRequest):
    """
    Endpoint to predict patient mortality.
    Receives raw patient data and a model choice, returns prediction and probability.
    """
    # Convert the incoming list of Pydantic models to a pandas DataFrame
    raw_patient_df = pd.DataFrame([p.dict() for p in request.patient_data])
    try:
        # Call the core prediction logic
        result = predict_mortality_api(raw_patient_df, request.model_choice)
        return result
    except HTTPException as e:
        # Re-raise HTTPExceptions directly
        raise e
    except Exception as e:
        # Catch any other unexpected errors and then return a 500 Internal Server Error
        raise HTTPException(status_code=500, detail=f"An internal server error occurred: {str(e)}")

@app.get("/")
async def root():
    """
    Root endpoint for a simple health check.
    Returns a message indicating the API is running.
    """
    return {"message": "Mortality Prediction API is running! Access /docs for API documentation."}

# This block is for running the app directly using 'python main.py'
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)
