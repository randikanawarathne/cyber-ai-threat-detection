import sys
import os

print("🔧 Checking Cyber AI Project Setup...")
print("=" * 60)

# Check Python
print(f"Python: {sys.version}")

# Check directories
print("\n📁 Checking directories:")
dirs = ['data', 'models', 'reports', 'src', 'notebooks']
all_dirs_ok = True
for d in dirs:
    if os.path.exists(d):
        print(f"  ✓ {d}/")
    else:
        print(f"  ✗ {d}/ (missing)")
        all_dirs_ok = False

# Check packages
print("\n📦 Checking packages:")
packages = ['pandas', 'numpy', 'sklearn', 'joblib']
all_packages_ok = True
for pkg in packages:
    try:
        if pkg == 'sklearn':
            import sklearn
            version = sklearn.__version__
            name = 'scikit-learn'
        else:
            module = __import__(pkg)
            version = module.__version__
            name = pkg
        print(f"  ✓ {name} {version}")
    except ImportError:
        print(f"  ✗ {pkg} (not installed)")
        all_packages_ok = False

print("\n" + "=" * 60)
if all_dirs_ok and all_packages_ok:
    print("✅ SETUP IS CORRECT!")
    print("\nRun these commands:")
    print("1. python src/train_model.py  (to train models)")
    print("2. python src/detect_threats.py  (to detect threats)")
else:
    print("❴ SETUP ISSUES DETECTED")
    if not all_packages_ok:
        print("\nFix packages with: pip install pandas numpy scikit-learn joblib")
    if not all_dirs_ok:
        print("\nCreate missing directories manually")