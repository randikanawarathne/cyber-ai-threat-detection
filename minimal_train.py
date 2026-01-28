import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import joblib
import os

print("🚀 Starting minimal threat detection training...")

# Create directories
os.makedirs('models', exist_ok=True)
os.makedirs('data', exist_ok=True)

# Create synthetic data
np.random.seed(42)
X = np.random.randn(1000, 5)
y = np.random.choice([0, 1], 1000, p=[0.9, 0.1])

# Train model
model = RandomForestClassifier(n_estimators=50, random_state=42)
model.fit(X, y)

# Save model
joblib.dump(model, 'models/simple_detector.pkl')
print("✅ Model trained and saved!")

# Test
accuracy = model.score(X, y)
print(f"📊 Training accuracy: {accuracy:.2%}")
print(f"🔍 Threat detection rate: {y.mean():.2%}")