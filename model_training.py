import pandas as pd
import matplotlib.pyplot as plt
import joblib

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.ensemble import RandomForestRegressor


# Load dataset
df = pd.read_csv("learnlens_synthetic_indian_student_dataset.csv")


# Select features

features = [
    "Study_Hours_Per_Day",
    "Physical_Activity_Hours_Per_Day",
    "Sleep_Hours_Per_Day",
    "Attendance_Percentage",
    "Previous_Exam_Score"
]
X = df[features]

# Select target
y = df["CGPA"]

# Split data into training and testing data

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.25,
    random_state= 42
)

# Train Linear Regression model
model = LinearRegression()
model.fit(X_train, y_train)

# 5-Fold Cross-Validation
cv_scores = cross_val_score(
    LinearRegression(),
    X,
    y,
    cv=5,
    scoring="neg_mean_absolute_error"
)

cv_mae_scores = -cv_scores

print("\n5-Fold Cross-Validation")
print("-------------------------")
print("MAE for each fold:", cv_mae_scores.round(4))
print("Average MAE:", round(cv_mae_scores.mean(), 4))

# Train Random Forest model
rf_model = RandomForestRegressor(
    n_estimators=100,
    random_state=42
)


# Display model coefficients
print("\nModel Coefficients:")

for feature, coefficient in zip(features, model.coef_):
    print(f"{feature}: {coefficient:.4f}")

print("Intercept:", round(model.intercept_, 4))


# Make predicitions

y_pred = model.predict(X_test).ravel()

# Check prediction range
print("\nPrediction Range Check")
print("Minimum predicted CGPA:", round(y_pred.min(), 2))
print("Maximum predicted CGPA:", round(y_pred.max(), 2))

outside_range = ((y_pred < 0) | (y_pred > 10)).sum()
print("Predictions outside 0-10:", outside_range)

# Compare actual and predicted GPA

comparison = pd.DataFrame({
    "Actual CGPA": y_test,
    "Predicted CGPA": y_pred
})

print("\nActual vs Predicted CGPA:")
print(comparison.head(10))
# Evaluate model

# from sklearn.metrics import mean_absolute_error, r2_score
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# Mean Prediction Baseline
baseline_pred = [y_train.mean()] * len(y_test)

baseline_mae = mean_absolute_error(y_test, baseline_pred)
baseline_r2 = r2_score(y_test, baseline_pred)

print("\nMean Prediction Baseline")
print("-------------------------")
print("MAE:", round(baseline_mae, 4))
print("R²:", round(baseline_r2, 4))

print("\nModel Evaluation")
print("MAE:", mae)
print("R²:", r2)

rf_model.fit(X_train, y_train)

# Make predictions
rf_pred = rf_model.predict(X_test)

# Evaluate Random Forest
rf_mae = mean_absolute_error(y_test, rf_pred)
rf_r2 = r2_score(y_test, rf_pred)

print("\nModel Comparison")
print("-------------------------")
print("Linear Regression:")
print("MAE:", round(mae, 4))
print("R²:", round(r2, 4))

print("\nRandom Forest:")
print("MAE:", round(rf_mae, 4))
print("R²:", round(rf_r2, 4))


# Save the trained model and its feature order
joblib.dump(
    {"model": model, "features": features},
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

# Validate student inputs
def validate_student(student):

    limits = {
        "Study_Hours_Per_Day": (0, 24),
        "Physical_Activity_Hours_Per_Day": (0, 24),
        "Sleep_Hours_Per_Day": (0, 24),
        "Attendance_Percentage": (0, 100),
        "Previous_Exam_Score": (0, 100)
    }

    for feature, (minimum, maximum) in limits.items():

        value = student[feature].iloc[0]

        if not minimum <= value <= maximum:
            raise ValueError(
                f"{feature} must be between {minimum} and {maximum}."
            )

    return True

# Test a prediction for one example student

student = pd.DataFrame([{
    "Study_Hours_Per_Day": 4,
    "Physical_Activity_Hours_Per_Day": 1,
    "Sleep_Hours_Per_Day": 7,
    "Attendance_Percentage": 85,
    "Previous_Exam_Score": 75
}])

print("Features:", features)
print("Student columns:", student.columns.tolist())
print("Student data:")
print(student)

validate_student(student)
print("Student inputs passed validation!")

predicted_cgpa = model.predict(student[features])[0]

print("\nExample Student")
print(student.to_string(index=False))
print("Predicted CGPA:", round(predicted_cgpa, 2))



