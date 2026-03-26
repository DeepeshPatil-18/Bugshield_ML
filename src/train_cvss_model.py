import pandas as pd
import numpy as np
import pickle

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score

print("Loading dataset...")

df = pd.read_csv("data/processed/nvd_training.csv")

print("Loading embeddings...")

X = np.load("data/processed/text_embeddings.npy")

# Target variable
y = df["cvss_score"]

# Train test split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

print("Training CVSS regression model...")

model = RandomForestRegressor(
    n_estimators=200,
    n_jobs=-1,
    random_state=42
)

model.fit(X_train, y_train)

# Predictions
preds = model.predict(X_test)

mae = mean_absolute_error(y_test, preds)
r2 = r2_score(y_test, preds)

print("\nMean Absolute Error:", mae)
print("R2 Score:", r2)

# Save model
with open("models/cvss_model.pkl", "wb") as f:
    pickle.dump(model, f)

print("\nCVSS model saved.")