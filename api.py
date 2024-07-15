# Import necessary libraries
import pickle
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from sklearn.preprocessing import MinMaxScaler
import uvicorn

# Initialize FastAPI app
app = FastAPI()

# Define a Pydantic model for input data validation
class HouseData(BaseModel):
    CRIM: float
    ZN: float
    INDUS: float
    CHAS: float
    NOX: float
    RM: float
    AGE: float
    DIS: float
    RAD: float
    TAX: float
    PTRATIO: float
    B: float
    LSTAT: float
    PRICE_PER_ROOM: float

# Load the trained model from the file
with open('linear_regression_model.pkl', 'rb') as f:
    model = pickle.load(f)

# Load the scaler from the file
with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

# Define a prediction endpoint
@app.post('/predict')
def predict(data: HouseData):
    try:
        # Convert input data to DataFrame
        input_data = pd.DataFrame([data.dict().values()], columns=data.dict().keys())

        # Debugging: Print input data and scaler information
        print("Input data columns: ", input_data.columns)
        print("Input data: ", input_data.head())
        print("Scaler data columns: ", scaler.data_max_)

        # Normalize the input data using the same scaler
        input_data = scaler.transform(input_data)
        
        # Make prediction using the trained model
        prediction = model.predict(input_data)
        
        # Return the prediction as a JSON response
        return {"predicted_price": prediction[0]}
        
    except Exception as e:
        # Log the error and return a JSON response with the error message
        print(f"Error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

# Run FastAPI with uvicorn
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
