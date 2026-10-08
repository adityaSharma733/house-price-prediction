import sys, pandas as pd, joblib
model = joblib.load("models/house_price_model.pkl")
area, bedrooms, bathrooms, age = float(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4])
X = pd.DataFrame([{"area": area, "bedrooms": bedrooms, "bathrooms": bathrooms, "age": age}])
print(f"Predicted price: ₹{model.predict(X)[0]:,.0f}")
