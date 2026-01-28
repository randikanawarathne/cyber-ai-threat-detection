#!/usr/bin/env python3
"""
Fix import issues
"""
import os
import sys

# Add current directory to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

print("✅ Python path fixed!")
print(f"Current path: {sys.path[0]}")

# Now try to run main
exec(open("main.py").read())