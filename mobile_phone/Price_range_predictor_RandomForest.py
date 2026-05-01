import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Getting the data
df = pd.read_csv("dataset\\train.csv")

# Assigning the variables
y = df["price_range"]
feature_columns = [col for col in df.columns if col != "price_range"]
x = df[feature_columns]

# Splitting data for testing and training
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=1)

# Data overview
print(f"Training Sample Size : {x_train.shape[0]}")
print(f"Testing Sample Size : {x_test.shape[0]}")

# Model config
model = RandomForestClassifier(random_state=1,max_depth=5, n_estimators=100,min_samples_split=10,min_samples_leaf=4)
model.fit(x_train, y_train)

# Prediction
train_predict = model.predict(x_train)
test_predict = model.predict(x_test)

# Evaluation
train_acc = accuracy_score(y_train, train_predict)
test_acc = accuracy_score(y_test, test_predict)

# Model evaluation
print("\n" + "*" * 50)
print("Model Evaluation")
print("*" * 50)
print(f"Training Accuracy : {train_acc:.2%}")
print(f"Testing Accuracy : {test_acc:.2%}")

if train_acc * 0.9 > test_acc:
    print("Model may be overfitting")

# Classification Report
print("\n" + "*" * 50)
print("Classification Report")
print("*" * 50)
price_labels = ["0 : Budget", "1 : Mid", "2 : High", "3 : Premium"]
print(classification_report(y_test, test_predict, target_names=price_labels))

# Confusion matrix
print("\n" + "*" * 50)
print("Confusion Matrix")
print("*" * 50)
print(confusion_matrix(y_test, test_predict))

# Feature importance
print("\n" + "*" * 50)
print("Feature Importance")
print("*" * 50)
important_df = pd.DataFrame({
    "Features": feature_columns,
    "Importance": model.feature_importances_
}).sort_values("Importance", ascending=False)

for _, row in important_df.iterrows():
    print(f"{row['Features']:15s} : {row['Importance']:.2%}")

# Predict new phone
new_phone = pd.DataFrame({
    'battery_power': [2000],
    'blue': [1],
    'clock_speed': [3.0],
    'dual_sim': [1],
    'fc': [16],
    'four_g': [1],
    'int_memory': [128],
    'm_dep': [0.4],
    'mobile_wt': [150],
    'n_cores': [8],
    'pc': [20],
    'px_height': [1920],
    'px_width': [1080],
    'ram': [8192],
    'sc_h': [16],
    'sc_w': [8],
    'talk_time': [20],
    'three_g': [1],
    'touch_screen': [1],
    'wifi': [1]
})

prediction = model.predict(new_phone)[0]
print("\n" + "*" * 50)
print(f"Predicted Price Range: {prediction} ({price_labels[prediction]})")