import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from pathlib import Path

df = pd.read_csv("data/house_data.csv")
features = ["area", "bedrooms", "bathrooms", "age"]
X, y = df[features], df["price"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)
pred = model.predict(X_test)

print("MAE:", mean_absolute_error(y_test, pred))
print("R2 Score:", r2_score(y_test, pred))

Path("models").mkdir(exist_ok=True)
joblib.dump(model, "models/house_price_model.pkl")
print("Model saved to models/house_price_model.pkl")
