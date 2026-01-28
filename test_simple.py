#!/usr/bin/env python3
"""
Simple test without complex imports
"""
import os
import sys

print("🧪 Testing basic functionality...")

# Check Python
print(f"Python: {sys.version.split()[0]}")

# Check files
files_to_check = [
    "main.py",
    "src/detect_threats.py",
    "src/train_model.py"
]

for file in files_to_check:
    if os.path.exists(file):
        size = os.path.getsize(file)
        print(f"✅ {file} ({size:,} bytes)")
    else:
        print(f"❌ {file} (MISSING)")

# Create directories
for dir in ["data", "models", "reports"]:
    os.makedirs(dir, exist_ok=True)
    print(f"📁 Created/verified: {dir}/")

print("\n✅ Basic test complete!")
print("\nNow run: python main.py")