import pandas as pd
import os

def load_data(file_path):
    return pd.read_csv(file_path)

def clean_data(df):
    df['Date'] = pd.to_datetime(df['Date'])
    df['AQI Value'] = pd.to_numeric(df['AQI Value'], errors='coerce')
    df.dropna(inplace=True)
    return df

def feature_engineering(df):
    df['Year'] = df['Date'].dt.year
    df['Month'] = df['Date'].dt.month
    return df

def filter_by_country(data, country):
    return data[data['Country'] == country]

def generate_predictions(model, country, start_year, end_year):
    predictions = []
    for year in range(start_year, end_year + 1):
        for month in range(1, 13):
            features = [[year, month]]
            predicted_aqi = model.predict(features)[0]
            predictions.append({
                "country": country,
                "year": year,
                "month": month,
                "datatype": "predicted",
                "predicted_aqi": predicted_aqi
            })
    return predictions

def get_existing_data(data, country, start_year, end_year):
    existing_data = data[(data['Country'] == country) & (data['Year'] >= start_year) & (data['Year'] <= end_year)]
    if not existing_data.empty:
        existing_data = existing_data.rename(columns={"AQI Value": "predicted_aqi"})
        existing_data['datatype'] = 'obtained'
    return existing_data

def save_predictions(predictions, output_file):
    df = pd.DataFrame(predictions)
    df.to_csv(output_file, index=False)
