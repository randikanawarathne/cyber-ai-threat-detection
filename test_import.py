#!/usr/bin/env python3
"""
Test script to verify imports work
"""
import sys
import os

print("🧪 Testing imports...")

# Method 1: Direct import with path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

try:
    import detect_threats
    print("✅ Method 1: Direct import - SUCCESS")
except ImportError as e:
    print(f"❌ Method 1 failed: {e}")

# Method 2: Import as module
try:
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "detect_threats",
        os.path.join(os.path.dirname(__file__), "src", "detect_threats.py")
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    print("✅ Method 2: importlib - SUCCESS")
except Exception as e:
    print(f"❌ Method 2 failed: {e}")

# Method 3: Check file exists
detect_path = os.path.join(os.path.dirname(__file__), "src", "detect_threats.py")
if os.path.exists(detect_path):
    print(f"✅ File exists: {detect_path}")
    print(f"   File size: {os.path.getsize(detect_path)} bytes")
else:
    print(f"❌ File not found: {detect_path}")

# Current directory info
print(f"\n📁 Current directory: {os.getcwd()}")
print(f"📁 Directory contents:")
for item in os.listdir('.'):
    if os.path.isdir(item):
        print(f"   📁 {item}/")
    elif item.endswith('.py'):
        print(f"   📄 {item}")