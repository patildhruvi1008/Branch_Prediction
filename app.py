from flask import Flask, render_template, request
import pandas as pd
import os
from sklearn.preprocessing import LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split

app = Flask(__name__)

# Load and preprocess the dataset
file_path = "branch_predictor_dataset_modified.csv"
df = pd.read_csv(file_path)

# Encode categorical variables
label_encoders = {}
categorical_cols = ['Category', 'Branch']
for col in categorical_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le

# Train the KNN model
def train_model(score_column):
    X = df[['Category', score_column, '10th Marks', '12th Marks']]
    y = df['Branch']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = KNeighborsClassifier(n_neighbors=5)
    model.fit(X_train, y_train)
    return model

@app.route("/", methods=["GET", "POST"])
def index():
    result = None
    error = None
    categories = label_encoders['Category'].classes_

    if request.method == "POST":
        exam_type = request.form.get("exam_type").upper()
        category = request.form.get("category")
        try:
            exam_score = float(request.form.get("exam_score"))
            tenth_marks = float(request.form.get("tenth_marks"))
            twelfth_marks = float(request.form.get("twelfth_marks"))
        except ValueError:
            error = "Please enter valid numeric values."
            return render_template("index.html", categories=categories, error=error)

        if exam_type not in ["JEE", "MHT-CET"]:
            error = "Invalid exam type selected."
        else:
            max_score = 300 if exam_type == "JEE" else 200
            score_column = "JEE Marks" if exam_type == "JEE" else "MHT-CET Marks"

            if not (0 <= exam_score <= max_score):
                error = f"{exam_type} score must be between 0 and {max_score}."
            elif not (0 <= tenth_marks <= 100) or not (0 <= twelfth_marks <= 100):
                error = "10th and 12th marks must be between 0 and 100."
            else:
                encoded_category = label_encoders['Category'].transform([category])[0]

                # Cutoff logic
                if exam_type == "MHT-CET":
                    if category.lower() == "general" and exam_score < 90:
                        error = "General category requires at least 90 in MHT-CET."
                    elif category.lower() != "general" and exam_score < 80:
                        error = "Reserved category requires at least 80 in MHT-CET."
                elif exam_type == "JEE":
                    if category.lower() == "general" and exam_score < 90:
                        error = "General category requires at least 90 in JEE."

                if not error:
                    model = train_model(score_column)
                    input_data = pd.DataFrame([[encoded_category, exam_score, tenth_marks, twelfth_marks]],
                                              columns=['Category', score_column, '10th Marks', '12th Marks'])
                    prediction = model.predict(input_data)[0]
                    result = label_encoders['Branch'].inverse_transform([prediction])[0]

    return render_template("index.html", categories=categories, result=result, error=error)
