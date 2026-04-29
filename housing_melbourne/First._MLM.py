import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.tree import DecisionTreeRegressor

melbourne_file_path = "dataset\\melb_data.csv"
melbourne_data = pd.read_csv(melbourne_file_path)
melbourne_data = melbourne_data.dropna(axis=0)

Y = melbourne_data.Price
melbourne_features = ["Rooms", "Bathroom", "Landsize", "Lattitude", "Longtitude"]
X = melbourne_data[melbourne_features]

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=1)

models = {
    'Decision Tree': DecisionTreeRegressor(max_depth=5, random_state=1),
    'Random Forest': RandomForestRegressor(n_estimators=100, random_state=1)
}

for name, model in models.items():
    model.fit(X_train, y_train)
    
    train_pred = model.predict(X_train)
    test_pred = model.predict(X_test)
    
    train_error = mean_absolute_error(y_train, train_pred)
    test_error = mean_absolute_error(y_test, test_pred)
    r2 = r2_score(y_test, test_pred)
    
    print(f"\n{'='*50}")
    print(f"MODEL: {name}")
    print(f"Training Error: ${train_error:,.2f}")
    print(f"Testing Error:  ${test_error:,.2f}")
    print(f"R² Score:       {r2:.3f}")
    
    if test_error > train_error * 1.2:
        print("WARNING: Model might be overfitting")

best_model = models['Random Forest']

Test_houses = pd.DataFrame({
    "Rooms": [3, 4, 6, 8, 10],
    "Bathroom": [1, 2, 3, 4, 5],
    "Landsize": [200, 300, 400, 600, 700],
    "Lattitude": [-37.80, -37.70, -37.60, -37.50, -37.25],
    "Longtitude": [144.9, 145.0, 144.8, 144.0, 144.5]
})

print(f"\n{'='*50}")
print("PREDICTIONS FOR NEW HOUSES:")
predictions = best_model.predict()
for i, price in enumerate(predictions):
    print(f"House {i+1}: ${price:,.2f}")

print(f"\n{'='*50}")
print("FEATURE IMPORTANCE:")
importances = pd.DataFrame({
    'feature': melbourne_features,
    'importance': best_model.feature_importances_
}).sort_values('importance', ascending=False)

for _, row in importances.iterrows():
    print(f"{row['feature']}: {row['importance']:.2%}")

print(f"\nEstimated price range for House 1: ${predictions[0]-test_error:,.0f} - ${predictions[0]+test_error:,.0f}")