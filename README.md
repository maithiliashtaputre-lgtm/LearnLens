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

-  Study Hours Per Day
-  Physical Activity Hours Per Day
-  Sleep Hours Per Day
-  Attendance percentage
-  Previous Exam Score

The target variable is:

- 🎓 **CGPA**

---

## 📊 Dataset

LearnLens currently uses a **synthetic student academic performance dataset** containing 2,000 records. The dataset was created for developing and testing the initial machine learning model.

It includes five input features related to students' daily habits and academic performance:

| Feature | Description |
|---|---|
| Study Hours Per Day | Average hours spent studying daily |
| Physical Activity Hours Per Day | Average hours spent on physical activities daily |
| Sleep Hours Per Day | Average hours of sleep per day |
| Attendance Percentage | Percentage of classes attended |
| Previous Exam Score | Score obtained in a previous examination |

**Target variable:** CGPA, represented on a 0–10 scale.

### Data Source and Limitations

The current dataset is synthetically generated and does not contain verified records collected from real students. Its values are based on assumed relationships between the input features and CGPA, with variation added to simulate differences between students.

Therefore, model performance on this dataset should not be interpreted as evidence of accuracy on real-world student data.

Future development aims to evaluate LearnLens using suitable real-world student academic datasets, subject to data availability, quality, and privacy considerations.

### Features Used

For the current Linear Regression model:

```text
Study_Hours_Per_Day
Physical Activity_Hours_Per_Day
Sleep_Hours_Per_Day
Attendance_Percentage
Previous_Exam_Score