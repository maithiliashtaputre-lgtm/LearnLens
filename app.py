from fastapi.middleware.cors import CORSMiddleware
from fastapi import FastAPI
from pydantic import BaseModel, Field
import joblib
import pandas as pd

# Create the API
app = FastAPI(title="LearnLens API")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

# Load the trained model
saved_data = joblib.load("learnlens_model.pkl")

model = saved_data["model"]
features = saved_data["features"]


# Define the student's input data
class StudentData(BaseModel):
    Study_Hours_Per_Day: float = Field(ge=0, le=24)
    Physical_Activity_Hours_Per_Day: float = Field(ge=0, le=24)
    Sleep_Hours_Per_Day: float = Field(ge=0, le=24)
    Attendance_Percentage: float = Field(ge=0, le=100)
    Previous_Exam_Score: float = Field(ge=0, le=100)


# Home endpoint
@app.get("/")
def home():
    return {"message": "Welcome to LearnLens API!"}


# CGPA prediction endpoint
@app.post("/predict")
def predict_cgpa(student: StudentData):
    student_data = pd.DataFrame([student.model_dump()])
    student_data = student_data[features]

    prediction = model.predict(student_data)[0]

    return {
        "predicted_cgpa": round(float(prediction), 2)
    }