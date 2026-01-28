import numpy as np
import joblib
from datetime import datetime

print("🔍 Starting threat detection...")

try:
    model = joblib.load('models/simple_detector.pkl')
    print("✅ Model loaded successfully!")
except:
    print("⚠️ No model found. Please run training first.")
    exit(1)

# Simulate incoming logs
for i in range(10):
    # Generate random features
    features = np.random.randn(1, 5)
    
    # Predict
    prediction = model.predict(features)[0]
    probability = model.predict_proba(features)[0][1]
    
    # Generate alert if threat detected
    if prediction == 1:
        print(f"🚨 THREAT DETECTED! [Log #{i}]")
        print(f"   Time: {datetime.now().strftime('%H:%M:%S')}")
        print(f"   Confidence: {probability:.1%}")
        print(f"   Features: {features[0].round(2)}")
        print("-" * 40)

print("\n✅ Detection complete!")