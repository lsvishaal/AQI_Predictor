# src/model_training.py
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import joblib
import pandas as pd
from src.data_processing import load_data, clean_data, feature_engineering

def train_model(data_path):
    # Load and prepare data
    df = load_data(data_path)
    df = clean_data(df)
    df = feature_engineering(df)

    X = df[['Year', 'Month', 'Day']]  # Features
    y = df['AQI Value']  # Target

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train the model
    model = RandomForestRegressor()
    model.fit(X_train, y_train)

    # Evaluate the model
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    print(f'Mean Squared Error: {mse}')

    # Save the model
    joblib.dump(model, 'model/aqi_model.pkl')

if __name__ == "__main__":
    train_model('data/AQI_Data_Latest.csv')
