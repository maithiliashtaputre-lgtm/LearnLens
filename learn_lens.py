import pandas as pd

# Load dataset
df = pd.read_csv("student_lifestyle_dataset.csv")
# print(df.head())
# print(df.columns)
# print(df.info())

# Encode stress levels
df["Stress_Level_Encoded"] = df["Stress_Level"].map({
    "Low":1,
    "Moderate":2,
    "High":3
})

# Select features

X = df[[
    "Study_Hours_Per_Day",
    "Extracurricular_Hours_Per_Day",
    "Sleep_Hours_Per_Day",
    "Social_Hours_Per_Day",
    "Physical_Activity_Hours_Per_Day",
    "Stress_Level_Encoded"
]]

# Select target
y = df["GPA"]

# Split data into training and testing data

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.25,
    random_state= 42
)
# print(X_train.shape)
# print(X_test.shape)
# print(y_train.shape)
# print(y_test.shape)

# Train Linear Regression model
from sklearn.linear_model import LinearRegression
model = LinearRegression()
model.fit(X_train, y_train)
# print(model.coef_)
# print(model.intercept_)

# Make predicitions

y_pred = model.predict(X_test).ravel()

# Compare actual and predicted GPA

comparison = pd.DataFrame({
    "Actual GPA": y_test,
    "Predicted GPA": y_pred
})

# Evaluate model

from sklearn.metrics import mean_absolute_error, r2_score
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("MAE:", mae)
print("R²:", r2)

# Residual analysis
import matplotlib.pyplot as plt

residuals = y_test - y_pred

plt.scatter(y_test, y_pred)
plt.xlabel("Actual GPA")
plt.ylabel("Predictd GPA")
plt.title("LearnLens: Actual vs Predicted GPA")
plt.show()

residuals = y_test - y_pred

plt.scatter(y_pred, residuals)
plt.axhline(0)
plt.xlabel("Predicted GPA")
plt.ylabel("Residual")
plt.title("LearnLens: Residual Plot")
plt.show()
