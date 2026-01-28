#!/usr/bin/env python3
"""
Simple test to verify everything works
"""
import os
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import joblib

print("🧪 Running Simple Cyber AI Test...")
print("=" * 50)

# 1. Create directories
os.makedirs('models', exist_ok=True)
os.makedirs('data', exist_ok=True)
print("✓ Created directories")

# 2. Create simple data
np.random.seed(42)
data = {
    'bytes_sent': np.random.randint(100, 10000, 100),
    'bytes_received': np.random.randint(100, 10000, 100),
    'duration': np.random.exponential(1, 100),
    'is_threat': np.random.choice([0, 1], 100, p=[0.9, 0.1])
}

df = pd.DataFrame(data)
df.to_csv('data/simple_test.csv', index=False)
print(f"✓ Created data: {len(df)} records, {df['is_threat'].sum()} threats")

# 3. Train simple model
X = df[['bytes_sent', 'bytes_received', 'duration']]
y = df['is_threat']

model = RandomForestClassifier(n_estimators=10, random_state=42)
model.fit(X, y)

# 4. Save model
joblib.dump(model, 'models/simple_model.pkl')
print("✓ Model trained and saved")

# 5. Test prediction
test_sample = [[5000, 100, 0.5]]  # High bytes sent, low received, short duration
prediction = model.predict(test_sample)
probability = model.predict_proba(test_sample)[0][1]

print(f"\n🔍 Test Prediction:")
print(f"   Sample: {test_sample[0]}")
print(f"   Prediction: {'THREAT' if prediction[0] == 1 else 'NORMAL'}")
print(f"   Confidence: {probability:.1%}")

print("\n" + "=" * 50)
print("✅ Simple test complete! Your setup is working.")
print("\nNow try running: python src/train_model.py")