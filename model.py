import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score
import joblib


# Load dataset
df = pd.read_csv("student_performance_dataset.csv")


# Features and Target
X = df.drop("Final_score", axis=1)
y = df["Final_score"]


# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
X, y, test_size=0.2, random_state=42
)


# Train model
model = LinearRegression()
model.fit(X_train, y_train)


# Evaluate
y_pred = model.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)


print("Model Trained!")
print("MAE:", mae)
print("R2 Score:", r2)


# Save model
joblib.dump(model, "student_performance_model.pkl")
print("Model saved as student_performance_model.pkl")