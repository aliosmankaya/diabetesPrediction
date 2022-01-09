import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from joblib import load

# Load model
model = load("model/model.joblib")

# Data columns
columns = [
    'gender', 'polyuria', 'polydipsia', 'sudden_weight_loss',
    'weakness', 'polyphagia', 'genital_thrush', 'visual_blurring',
    'itching', 'irritability', 'delayed_healing', 'partial_paresis',
    'muscle_stiffness', 'alopecia', 'obesity',
]

# Predict function
def predict(columns, age, checks, model):

    checkList = []
    data = pd.read_csv("data/diabetes_data.csv", sep=";")

    scaler = MinMaxScaler()
    scaler.fit_transform(data.age.values.reshape(-1,1))
    age = scaler.transform(np.array([age]).reshape(-1,1))
    checkList.append(age[0][0])

    for i in columns:

        if i in checks:
            checkList.append(1)

        else:
            checkList.append(0)

    pred = model.predict(np.array([checkList,]))
    pred_proba = model.predict_proba(np.array([checkList,]))

    if pred[0] == 0:
        pred_text = f"You're ok\nAccuracy: {pred_proba[0][pred[0]]}"

    elif pred[0] == 1:
        pred_text = f"You have diabetes\nAccuracy: {pred_proba[0][pred[0]]}"

    return pred_text