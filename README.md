# 📚 LearnLens

> A beginner machine learning project that predicts student GPA based on lifestyle and study-related factors.

## 🧠 About LearnLens

LearnLens is a machine learning project built to explore how different aspects of a student's daily lifestyle relate to their academic performance.

The current version of LearnLens uses a real student lifestyle dataset and a **Linear Regression** model to learn patterns between lifestyle factors and GPA.

The goal of this project is not just to make predictions, but to understand the complete machine learning workflow — from loading and preparing data to training, predicting, and evaluating a model.

---

## 🎯 Objective

The current objective of LearnLens is:

> **To train a machine learning model that can predict a student's GPA using study and lifestyle-related factors.**

The model currently uses the following features:

- 📖 Study Hours Per Day
- 🎯 Extracurricular Hours Per Day
- 😴 Sleep Hours Per Day
- 🧑‍🤝‍🧑 Social Hours Per Day
- 🏃 Physical Activity Hours Per Day

The target variable is:

- 🎓 **GPA**

---

## 📊 Dataset

LearnLens currently uses a real student lifestyle dataset containing:

- **2,000 student records**
- **8 columns**

The original dataset contains:

- `Student_ID`
- `Study_Hours_Per_Day`
- `Extracurricular_Hours_Per_Day`
- `Sleep_Hours_Per_Day`
- `Social_Hours_Per_Day`
- `Physical_Activity_Hours_Per_Day`
- `GPA`
- `Stress_Level`

### Features Used

For the current Linear Regression model:

```text
Study_Hours_Per_Day
Extracurricular_Hours_Per_Day
Sleep_Hours_Per_Day
Social_Hours_Per_Day
Physical_Activity_Hours_Per_Day