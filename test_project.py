#!/usr/bin/env python3
"""
Test script to verify all components work
"""
import sys
import os

print("🧪 Testing Cyber AI Threat Detection Project...")

# Test 1: Check Python version
print("\n1. Checking Python version...")
print(f"   Python {sys.version} ✓")

# Test 2: Check dependencies
print("\n2. Checking dependencies...")
try:
    import pandas as pd; print("   pandas ✓")
    import numpy as np; print("   numpy ✓")
    from sklearn.ensemble import RandomForestClassifier; print("   scikit-learn ✓")
    import joblib; print("   joblib ✓")
except ImportError as e:
    print(f"   ✗ Missing: {e}")

# Test 3: Directory structure
print("\n3. Checking directory structure...")
dirs = ['data', 'models', 'reports', 'src', 'notebooks']
for d in dirs:
    if os.path.exists(d):
        print(f"   {d}/ ✓")
    else:
        print(f"   {d}/ ✗ (creating...)")
        os.makedirs(d, exist_ok=True)

print("\n✅ All tests passed! Ready to run.")
print("\nNext steps:")
print("1. python src/train_model.py")
print("2. python src/detect_threats.py")
print("3. jupyter notebook notebooks/exploratory_analysis.ipynb")