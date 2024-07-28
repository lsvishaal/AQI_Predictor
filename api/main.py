from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import os
from datetime import datetime
from src.data_processing import load_data, clean_data, feature_engineering, filter_by_country, generate_predictions, save_predictions, get_existing_data

# Load the model
model = joblib.load('model/aqi_model.pkl')

# Define the request model
class PredictionRequest(BaseModel):
    country: str
    from_year: int
    to_year: int

# Create the FastAPI app
app = FastAPI()

# Define the prediction endpoint
@app.post("/predict/")
async def predict(request: PredictionRequest):
    try:
        data_file = 'data/AQI_Data_Latest.csv'
        
        # Check if the CSV file exists
        if not os.path.exists(data_file):
            raise HTTPException(status_code=404, detail="Data file not found")
        
        data = load_data(data_file)
        data = clean_data(data)
        data = feature_engineering(data)
        country_data = filter_by_country(data, request.country)

        if country_data.empty:
            raise HTTPException(status_code=404, detail="Country data not found")

        # Date validation
        if request.from_year < 2022 or request.to_year > 2030:
            raise HTTPException(status_code=400, detail="Cannot predict for the given date range")

        # Fetch existing data within the date range
        existing_data = get_existing_data(data, request.country, request.from_year, request.to_year)

        # Prepare final data combining existing and predicted
        predictions = []
        if not existing_data.empty:
            predictions.extend(existing_data.to_dict('records'))
        
        # Determine the months and years to predict
        existing_years_months = {(row['Year'], row['Month']) for _, row in existing_data.iterrows()}
        for year in range(request.from_year, request.to_year + 1):
            for month in range(1, 13):
                if (year, month) not in existing_years_months:
                    features = [[year, month]]
                    predicted_aqi = model.predict(features)[0]
                    predictions.append({
                        "country": request.country,
                        "year": year,
                        "month": month,
                        "datatype": "predicted",
                        "predicted_aqi": predicted_aqi
                    })

        output_file = f'country_data/{request.country}_predictions.csv'
        save_predictions(predictions, output_file)

        return {"predictions": predictions}
    
    except Exception as e:
        # Log the exception details
        print(f"Error: {e}")
        raise HTTPException(status_code=500, detail="Internal Server Error")

@app.get("/")
async def root():
    return {"message": "Welcome to the AQI Predictor API. Use /predict/ endpoint to get predictions."}
