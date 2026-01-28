import os
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import joblib

print("🚀 Minimal Cyber Threat Detection Training")
print("=" * 50)

# Create directories
os.makedirs('models', exist_ok=True)
os.makedirs('data', exist_ok=True)
os.makedirs('reports', exist_ok=True)

# Create sample data
np.random.seed(42)
n_samples = 500

# Normal traffic
normal_data = {
    'duration': np.random.exponential(2, n_samples),
    'bytes_up': np.random.randint(100, 5000, n_samples),
    'bytes_down': np.random.randint(1000, 10000, n_samples),
    'packet_count': np.random.randint(10, 100, n_samples),
}

# Threat traffic (5%)
threat_data = {
    'duration': np.random.exponential(0.1, int(n_samples * 0.05)),
    'bytes_up': np.random.randint(10000, 100000, int(n_samples * 0.05)),
    'bytes_down': np.random.randint(1, 100, int(n_samples * 0.05)),
    'packet_count': np.random.randint(1000, 10000, int(n_samples * 0.05)),
}

# Combine
df_normal = pd.DataFrame(normal_data)
df_normal['label'] = 0

df_threat = pd.DataFrame(threat_data)
df_threat['label'] = 1

df = pd.concat([df_normal, df_threat], ignore_index=True)
df = df.sample(frac=1, random_state=42).reset_index(drop=True)

df.to_csv('data/network_traffic.csv', index=False)
print(f"✓ Created dataset: {len(df)} samples")
print(f"   Threats: {df['label'].sum()} ({df['label'].mean():.1%})")

# Train model
X = df.drop('label', axis=1)
y = df['label']

model = RandomForestClassifier(n_estimators=50, random_state=42)
model.fit(X, y)

# Save model
joblib.dump(model, 'models/threat_detector.pkl')
print("✓ Model saved to models/threat_detector.pkl")

# Test
accuracy = model.score(X, y)
print(f"📊 Model accuracy: {accuracy:.1%}")

# Feature importance
print("\n🔍 Feature Importance:")
for name, importance in zip(X.columns, model.feature_importances_):
    print(f"   {name}: {importance:.3f}")

print("\n" + "=" * 50)
print("✅ Training complete!")
print("\nNow run detection:")
print("python src/detect_threats.py")