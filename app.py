from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import joblib

# Load dataset
data = pd.read_csv("diabetes.csv")

# Define features and target variable
X = data.drop(columns="Outcome")
y = data["Outcome"]

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the RandomForestClassifier model
rf_model = RandomForestClassifier()
rf_model.fit(X_train, y_train)

# Save the trained model
joblib.dump(rf_model, "diabetes.pkl")

# Create FastAPI instance
app = FastAPI()

# Define the Pydantic model for input validation
class DiabetesInput(BaseModel):
    Pregnancies: int
    Glucose: int
    BloodPressure: int
    SkinThickness: int
    Insulin: int
    BMI: float
    DiabetesPedigreeFunction: float
    Age: int

# Load the saved model
rf_model = joblib.load("diabetes.pkl")

# Define the prediction endpoint
@app.post("/predict")
async def predict_diabetes(input_data: DiabetesInput):
    # Convert the input data to a pandas DataFrame
    input_df = pd.DataFrame([input_data.dict()])
    
    # Make prediction using the model
    prediction = rf_model.predict(input_df)
    
    # Determine the result based on the prediction
    result = "diabetes." if prediction[0] == 1 else "not diabetes."
    
    # Return the result
    return {"prediction": result}