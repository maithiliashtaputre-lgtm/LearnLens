import pandas as pd
import matplotlib.pyplot as plt
import joblib

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score

# Load dataset
df = pd.read_csv("learnlens_synthetic_indian_student_dataset.csv")


# Select features

X = df[[
    "Study_Hours_Per_Day",
    "Physical_Activity_Hours_Per_Day",
    "Sleep_Hours_Per_Day",
    "Attendance_Percentage",
    "Previous_Exam_Score"
]]

# Select target
y = df["GPA"]

# Split data into training and testing data

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.25,
    random_state= 42
)

# Train Linear Regression model
from sklearn.linear_model import LinearRegression
model = LinearRegression()
model.fit(X_train, y_train)


# Make predicitions

y_pred = model.predict(X_test).ravel()

# Compare actual and predicted GPA

comparison = pd.DataFrame({
    "Actual CGPA": y_test,
    "Predicted CGPA": y_pred
})

print("\nActual vs Predicted CGPA:")
print(comparison.head(10))
# Evaluate model

from sklearn.metrics import mean_absolute_error, r2_score
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("\nModel Evaluation")
print("MAE:", mae)
print("R²:", r2)

# Save the trained model and its feature order
joblib.dump(
    {"model": model, "features": X},
    "learnlens_model.pkl"
)

print("\nModel saved as learnlens_model.pkl")

# Plot 1: Actual vs Predicted CGPA
plt.scatter(y_test, y_pred)
plt.xlabel("Actual CGPA")
plt.ylabel("Predicted CGPA")
plt.title("LearnLens: Actual vs Predicted CGPA")
plt.show()

# Plot 2: Residual analysis
residuals = y_test - y_pred

plt.scatter(y_pred, residuals)
plt.axhline(0)
plt.xlabel("Predicted CGPA")
plt.ylabel("Residual")
plt.title("LearnLens: Residual Plot")
plt.show()



