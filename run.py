# save as run.py in project root
import subprocess
import sys

print("🚀 Starting Cyber Threat Detection...")
print("Running: python src/detect_threats.py")

# Run directly
result = subprocess.run([sys.executable, "src/detect_threats.py"])

if result.returncode == 0:
    print("✅ Execution successful!")
else:
    print(f"❌ Execution failed with code: {result.returncode}")
    