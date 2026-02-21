import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Load data
df = pd.read_csv("cardio_train.csv", sep=';')
df = df.drop(columns=["id"])

X = df.drop(columns=["cardio"])
y = df["cardio"]

_, X_test, _, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Load model
model = joblib.load("model.pkl")

preds = model.predict(X_test)
acc = accuracy_score(y_test, preds)

print(f"accuracy={acc:.4f}")