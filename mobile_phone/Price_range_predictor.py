import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import warnings
warnings.filterwarnings('ignore')

df = pd.read_csv("dataset\\train.csv")

# Features and target
y = df['price_range']
feature_columns = [col for col in df.columns if col != 'price_range']
X = df[feature_columns]

# Split into train and test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

# Data overview
print(f"Training samples: {X_train.shape[0]}")
print(f"Testing samples: {X_test.shape[0]}")

# Train model
model = DecisionTreeClassifier(max_depth=5, random_state=1, min_samples_split=10, min_samples_leaf=4)
model.fit(X_train, y_train)

# Predictions
train_pred = model.predict(X_train)
test_pred = model.predict(X_test)

# Evaluation
train_acc = accuracy_score(y_train, train_pred)
test_acc = accuracy_score(y_test, test_pred)

print("\n" + "*" * 50)
print("MODEL EVALUATION")
print("*" * 50)
print(f"Training Accuracy: {train_acc:.2%}")
print(f"Testing Accuracy:  {test_acc:.2%}")

if test_acc < train_acc * 0.9:
    print("WARNING: Possible overfitting!")

# Classification report
print("\n" + "*" * 50)
print("CLASSIFICATION REPORT")
print("*" * 50)
price_labels = ['0: Budget', '1: Mid-range', '2: High-end', '3: Premium']
print(classification_report(y_test, test_pred, target_names=price_labels))

# Confusion matrix
print("=" * 60)
print("CONFUSION MATRIX")
print("=" * 60)
print(confusion_matrix(y_test, test_pred))

# Feature importance
print("\n" + "=" * 60)
print("FEATURE IMPORTANCE")
print("=" * 60)
importance_df = pd.DataFrame({
    'Feature': feature_columns,
    'Importance': model.feature_importances_
}).sort_values('Importance', ascending=False)

for _, row in importance_df.iterrows():
    if row['Importance'] > 0:
        print(f"  {row['Feature']:15s}: {row['Importance']:.2%}")

# Predict new phone
print("\n" + "=" * 60)
print("PREDICT NEW PHONE")
print("=" * 60)

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
print(f"Predicted Price Range: {prediction} ({price_labels[prediction]})")