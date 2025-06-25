import os
import pandas as pd
from sklearn.model_selection import train_test_split

from src.data_loader import load_data
from src.preprocessing import preprocess_data
from src.model import train_model, evaluate_model, save_model
from src.utils import plot_confusion_matrix

# Setup
DATA_PATH = "data/telco_churn.csv"
MODEL_PATH = "models/churn_model.pkl"
os.makedirs("models", exist_ok=True)
os.makedirs("outputs/plots", exist_ok=True)

# Load Data
df = load_data(DATA_PATH)

# Preprocess Data
df, label_encoders = preprocess_data(df)

# Split Features & Labels
X = df.drop('Churn', axis=1)
y = df['Churn']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Model
model = train_model(X_train, y_train, model_type="logistic")

# Evaluate
accuracy, confusion = evaluate_model(model, X_test, y_test)
print(f"Accuracy: {accuracy:.4f}")
print("Confusion Matrix:\n", confusion)

# Save Model & Plot
save_model(model, MODEL_PATH)
plot_confusion_matrix(confusion)
