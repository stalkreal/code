# uber_regression.py
#experiment No:5
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
import numpy as np

def preprocess(df: pd.DataFrame):
    # Basic cleaning — dataset has columns like: fare_amount, pickup_datetime, pickup_longitude, ...
    df = df.copy()
    # Remove impossible fares and coords
    df = df[df["fare_amount"].between(0, 500)]
    # Drop extreme coordinates
    for col in ["pickup_longitude","pickup_latitude","dropoff_longitude","dropoff_latitude"]:
        df = df[df[col].between(-180, 180)]
    # Parse datetime
    df["pickup_datetime"] = pd.to_datetime(df["pickup_datetime"], errors="coerce", utc=True)
    df = df.dropna(subset=["pickup_datetime"])
    df["hour"] = df["pickup_datetime"].dt.hour
    df["dayofweek"] = df["pickup_datetime"].dt.dayofweek
    # Distance feature (haversine)
    def haversine(lat1, lon1, lat2, lon2):
        R = 6371.0
        lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        a = np.sin(dlat/2)**2 + np.cos(lat1)*np.cos(lat2)*np.sin(dlon/2)**2
        return 2*R*np.arcsin(np.sqrt(a))
    df["distance_km"] = haversine(
        df["pickup_latitude"], df["pickup_longitude"],
        df["dropoff_latitude"], df["dropoff_longitude"]
    )
    # Remove outliers by distance
    df = df[df["distance_km"].between(0, 100)]
    # Select features
    features = ["distance_km", "hour", "dayofweek", "passenger_count"]
    df = df.dropna(subset=["fare_amount"] + features)
    X = df[features]
    y = df["fare_amount"]
    return X, y

def evaluate(name, model, X_test, y_test):
    pred = model.predict(X_test)
    r2 = r2_score(y_test, pred)
    mse = mean_squared_error(y_test, pred)
    rmse = np.sqrt(mse)
    return {"model": name, "R2": r2, "RMSE": rmse}


if __name__ == "__main__":
    # Put your local path to the Kaggle CSV
    path = "uber.csv"  # e.g., 'uber_fares.csv'
    df = pd.read_csv(path)
    X, y = preprocess(df)

    # Simple correlation check
    print("Correlation with fare_amount (spearman):")
    corr = pd.concat([X, y], axis=1).corr(method="spearman")["fare_amount"].sort_values(ascending=False)
    print(corr)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    lr = LinearRegression()
    lr.fit(X_train, y_train)

    rf = RandomForestRegressor(n_estimators=300, max_depth=None, random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train)

    results = [
        evaluate("LinearRegression", lr, X_test, y_test),
        evaluate("RandomForest", rf, X_test, y_test)
    ]
    for r in results:
        print(r)
