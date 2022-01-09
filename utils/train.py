import pandas as pd
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from joblib import dump

# Read data
data = pd.read_csv("data/diabetes_data.csv", sep=";")

# Label Encoding for Gender
encoder = LabelEncoder()
data.gender = encoder.fit_transform(data.gender)

# Min-Max Scaling for Age
scaler = MinMaxScaler()
data.age = scaler.fit_transform(data.age.values.reshape(-1,1))

# Train-Test split
x = data.drop("class", axis=1)
y = data["class"]
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.1, random_state=44, shuffle=True)

# Modelling
model = RandomForestClassifier()
model.fit(x_train, y_train)

# Save model
dump(model, "model/model.joblib")

# Test Prediction
y_pred = model.predict(x_test)
print(f"Test Score: {accuracy_score(y_test, y_pred)}")