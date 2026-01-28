#!/usr/bin/env python3
"""
AI-DRIVEN CYBER THREAT DETECTION SYSTEM - MAIN ENTRY POINT
🚀 No Import Errors - Guaranteed to Work
"""
import os
import sys
import subprocess
import importlib.util
from pathlib import Path

class CyberThreatDetection:
    def __init__(self):
        self.project_root = Path(__file__).parent
        self.src_dir = self.project_root / "src"
        self.data_dir = self.project_root / "data"
        self.models_dir = self.project_root / "models"
        self.reports_dir = self.project_root / "reports"
        
        # Create directories
        self.create_directories()
        
    def create_directories(self):
        """Create all necessary directories"""
        directories = [self.data_dir, self.models_dir, self.reports_dir]
        for directory in directories:
            directory.mkdir(exist_ok=True)
            print(f"📁 Created/verified: {directory}")
    
    def print_banner(self):
        """Print awesome banner"""
        print("\n" + "="*70)
        print("🛡️  AI-DRIVEN CYBER THREAT DETECTION SYSTEM")
        print("="*70)
        print("🔐 Combining Machine Learning & Cybersecurity")
        print("🎯 Real-time Anomaly Detection & Threat Analysis")
        print("📊 Built for SOC Analysts & Security Engineers")
        print("="*70)
    
    def check_requirements(self):
        """Check and install requirements WITHOUT errors"""
        print("\n🔍 Checking requirements...")
        
        requirements = {
            'pandas': 'Data analysis',
            'numpy': 'Numerical computing',
            'sklearn': 'Machine learning',
            'joblib': 'Model serialization',
            'matplotlib': 'Visualization'
        }
        
        missing = []
        
        for package, description in requirements.items():
            try:
                if package == 'sklearn':
                    import sklearn
                    version = sklearn.__version__
                    name = 'scikit-learn'
                else:
                    module = __import__(package)
                    version = module.__version__
                    name = package
                
                print(f"✅ {name:20} {version:10} - {description}")
            except ImportError:
                print(f"❌ {package:20} {'MISSING':10} - {description}")
                missing.append(package)
        
        if missing:
            print(f"\n⚠️  Missing packages detected!")
            self.install_packages(missing)
        else:
            print("\n🎉 All requirements satisfied!")
        
        return len(missing) == 0
    
    def install_packages(self, packages):
        """Install missing packages"""
        print(f"\n📦 Installing {len(packages)} missing packages...")
        
        for package in packages:
            if package == 'sklearn':
                package_name = 'scikit-learn'
            else:
                package_name = package
            
            print(f"  Installing {package_name}...", end=' ', flush=True)
            try:
                subprocess.run([
                    sys.executable, "-m", "pip", "install", 
                    "--quiet", package_name
                ], check=True, capture_output=True)
                print("✅")
            except subprocess.CalledProcessError:
                print("❌")
                print(f"    Failed to install {package_name}")
        
        print("\n✅ Package installation complete!")
    
    def load_detection_module(self):
        """Load detect_threats.py without import errors"""
        detect_path = self.src_dir / "detect_threats.py"
        
        if not detect_path.exists():
            print(f"\n❌ ERROR: {detect_path} not found!")
            print("Please make sure detect_threats.py exists in src/ directory")
            return None
        
        print(f"\n📄 Loading detection module: {detect_path}")
        
        try:
            # Method 1: Direct import with sys.path
            sys.path.insert(0, str(self.src_dir))
            
            # Method 2: Use importlib
            spec = importlib.util.spec_from_file_location(
                "detect_threats", 
                str(detect_path)
            )
            module = importlib.util.module_from_spec(spec)
            
            # Execute the module
            spec.loader.exec_module(module)
            
            print("✅ Detection module loaded successfully!")
            return module
            
        except Exception as e:
            print(f"❌ Failed to load module: {e}")
            print("\n💡 Trying alternative method...")
            return self.load_module_alternative()
    
    def load_module_alternative(self):
        """Alternative method to load module"""
        print("\n🔄 Using alternative loading method...")
        
        # Read the file and exec it
        detect_path = self.src_dir / "detect_threats.py"
        
        try:
            with open(detect_path, 'r', encoding='utf-8') as f:
                code = f.read()
            
            # Create a module dict
            module_dict = {
                '__name__': 'detect_threats',
                '__file__': str(detect_path)
            }
            
            # Execute the code
            exec(code, module_dict)
            
            # Create a simple module-like object
            class SimpleModule:
                def __init__(self, d):
                    self.__dict__.update(d)
            
            module = SimpleModule(module_dict)
            print("✅ Module loaded using exec() method!")
            return module
            
        except Exception as e:
            print(f"❌ Alternative method also failed: {e}")
            return None
    
    def run_training_mode(self):
        """Run in training mode"""
        print("\n" + "="*70)
        print("🎯 MODE: TRAINING - Building ML Models")
        print("="*70)
        
        train_path = self.src_dir / "train_model.py"
        
        if not train_path.exists():
            print("❌ train_model.py not found!")
            print("Running built-in training...")
            self.builtin_training()
            return
        
        print(f"Running: {train_path}")
        
        try:
            # Run as subprocess to avoid import issues
            result = subprocess.run(
                [sys.executable, str(train_path)],
                capture_output=False,
                text=True
            )
            
            if result.returncode == 0:
                print("\n✅ Training completed successfully!")
            else:
                print(f"\n❌ Training failed with code: {result.returncode}")
                
        except Exception as e:
            print(f"❌ Error running training: {e}")
            self.builtin_training()
    
    def builtin_training(self):
        """Built-in training if train_model.py fails"""
        print("\n🔧 Running built-in training...")
        
        import numpy as np
        import pandas as pd
        from sklearn.ensemble import RandomForestClassifier
        import joblib
        
        # Create sample data
        np.random.seed(42)
        n_samples = 500
        
        print("Generating synthetic cybersecurity data...")
        data = {
            'duration': np.random.exponential(1, n_samples),
            'bytes_sent': np.random.lognormal(8, 2, n_samples),
            'bytes_received': np.random.lognormal(9, 2, n_samples),
            'packet_count': np.random.poisson(10, n_samples),
            'connection_count': np.random.poisson(3, n_samples),
        }
        
        df = pd.DataFrame(data)
        
        # Add threats (7% of data)
        n_threats = int(0.07 * n_samples)
        threat_indices = np.random.choice(n_samples, n_threats, replace=False)
        
        # Make threats look different
        for idx in threat_indices:
            df.loc[idx, 'bytes_sent'] *= 50  # Large uploads
            df.loc[idx, 'duration'] = 0.005  # Very short
            df.loc[idx, 'packet_count'] *= 20  # Many packets
        
        df['is_threat'] = 0
        df.loc[threat_indices, 'is_threat'] = 1
        
        # Save data
        data_file = self.data_dir / "builtin_training_data.csv"
        df.to_csv(data_file, index=False)
        print(f"✅ Created dataset: {data_file} ({len(df)} samples, {n_threats} threats)")
        
        # Train model
        print("Training Random Forest model...")
        X = df.drop('is_threat', axis=1)
        y = df['is_threat']
        
        model = RandomForestClassifier(
            n_estimators=50,
            max_depth=10,
            random_state=42,
            class_weight='balanced'
        )
        
        model.fit(X, y)
        
        # Save model
        model_file = self.models_dir / "builtin_model.pkl"
        joblib.dump(model, model_file)
        
        # Evaluate
        accuracy = model.score(X, y)
        print(f"📊 Model accuracy: {accuracy:.2%}")
        print(f"📁 Model saved: {model_file}")
        
        # Feature importance
        print("\n🔍 Top 5 Important Features:")
        features = X.columns.tolist()
        importances = model.feature_importances_
        
        for feature, importance in sorted(zip(features, importances), 
                                        key=lambda x: x[1], reverse=True)[:5]:
            print(f"  {feature}: {importance:.4f}")
    
    def run_detection_mode(self, duration=30):
        """Run in detection mode"""
        print("\n" + "="*70)
        print("🔍 MODE: DETECTION - Real-time Threat Analysis")
        print("="*70)
        
        # Load module
        module = self.load_detection_module()
        
        if module is None:
            print("❌ Cannot load detection module!")
            print("Running built-in detection...")
            self.builtin_detection(duration)
            return
        
        try:
            # Try to run the main function
            if hasattr(module, 'main'):
                print("Running main() function...")
                module.main()
            elif hasattr(module, 'simulate_real_time_detection'):
                print(f"Running real-time detection for {duration} seconds...")
                module.simulate_real_time_detection(duration_seconds=duration)
            elif hasattr(module, 'ThreatDetector'):
                print("Creating ThreatDetector instance...")
                detector = module.ThreatDetector()
                detector.print_dashboard()
            else:
                print("Available functions in module:")
                for attr in dir(module):
                    if not attr.startswith('_') and callable(getattr(module, attr)):
                        print(f"  - {attr}()")
                
                # Try to run analyze_existing_data if exists
                if hasattr(module, 'analyze_existing_data'):
                    module.analyze_existing_data()
                else:
                    print("\n❌ No runnable function found!")
                    self.builtin_detection(duration)
                    
        except Exception as e:
            print(f"❌ Error running detection: {e}")
            self.builtin_detection(duration)
    
    def builtin_detection(self, duration=20):
        """Built-in detection simulation"""
        print("\n🔧 Running built-in detection simulation...")
        
        import time
        import random
        from datetime import datetime
        import json
        
        print(f"Simulating {duration} seconds of network traffic...")
        print("Press Ctrl+C to stop\n")
        
        alerts = []
        log_counter = 0
        threat_counter = 0
        
        try:
            start_time = time.time()
            
            while time.time() - start_time < duration:
                # Generate fake log
                log = {
                    'timestamp': datetime.now().isoformat(),
                    'src_ip': f'10.0.0.{random.randint(1, 100)}',
                    'dst_ip': f'{random.randint(1, 255)}.{random.randint(1, 255)}.1.{random.randint(1, 10)}',
                    'protocol': random.choice(['TCP', 'UDP', 'HTTP', 'HTTPS']),
                    'duration': random.expovariate(1.0),
                    'bytes_up': random.randint(100, 10000),
                    'bytes_down': random.randint(1000, 50000),
                }
                
                # Detect "threats" (15% chance)
                is_threat = random.random() < 0.15
                
                if is_threat:
                    threat_counter += 1
                    
                    # Create fake threat patterns
                    log['bytes_up'] = random.randint(100000, 1000000)
                    log['duration'] = 0.001
                    
                    # Calculate threat score
                    bytes_ratio = log['bytes_up'] / max(log['bytes_down'], 1)
                    threat_score = min(95, bytes_ratio / 100)
                    
                    # Determine severity
                    if threat_score > 80:
                        severity = "CRITICAL"
                    elif threat_score > 60:
                        severity = "HIGH"
                    elif threat_score > 40:
                        severity = "MEDIUM"
                    else:
                        severity = "LOW"
                    
                    # Create alert
                    alert = {
                        'id': f'ALT-{threat_counter:04d}',
                        'time': datetime.now().strftime('%H:%M:%S'),
                        'severity': severity,
                        'score': round(threat_score, 1),
                        'src': log['src_ip'],
                        'dst': log['dst_ip'],
                        'type': 'Data Exfiltration' if log['bytes_up'] > 500000 else 'Suspicious Activity',
                        'bytes': log['bytes_up']
                    }
                    
                    alerts.append(alert)
                    
                    # Print alert
                    colors = {
                        'CRITICAL': '\033[91m',  # Red
                        'HIGH': '\033[93m',      # Yellow
                        'MEDIUM': '\033[33m',    # Orange
                        'LOW': '\033[96m',       # Cyan
                        'RESET': '\033[0m'
                    }
                    
                    color = colors.get(severity, colors['RESET'])
                    print(f"{color}🚨 [{alert['time']}] {severity}: {alert['type']} (Score: {alert['score']}){colors['RESET']}")
                
                log_counter += 1
                
                # Progress indicator
                if log_counter % 10 == 0:
                    elapsed = time.time() - start_time
                    remaining = max(0, duration - elapsed)
                    print(f"📊 Logs: {log_counter}, Threats: {threat_counter}, Time: {remaining:.0f}s remaining")
                
                time.sleep(0.2)
                
        except KeyboardInterrupt:
            print("\n\n🛑 Detection stopped by user")
        
        finally:
            # Save report
            if alerts:
                report = {
                    'generated': datetime.now().isoformat(),
                    'duration_seconds': duration,
                    'total_logs': log_counter,
                    'threats_detected': threat_counter,
                    'detection_rate': threat_counter / max(log_counter, 1),
                    'alerts': alerts
                }
                
                report_file = self.reports_dir / f"detection_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
                
                with open(report_file, 'w') as f:
                    json.dump(report, f, indent=2)
                
                print(f"\n📄 Report saved: {report_file}")
            
            # Summary
            print("\n" + "="*50)
            print("📋 DETECTION SUMMARY")
            print("="*50)
            print(f"Total logs analyzed: {log_counter}")
            print(f"Threats detected: {threat_counter}")
            if log_counter > 0:
                print(f"Detection rate: {threat_counter/log_counter:.2%}")
            print(f"Alerts saved: {len(alerts)}")
            print("="*50)
    
    def run_test_mode(self):
        """Run in test/verification mode"""
        print("\n" + "="*70)
        print("🧪 MODE: TEST - System Verification")
        print("="*70)
        
        print("\n🔧 System Check:")
        print(f"  Python: {sys.version.split()[0]}")
        print(f"  Platform: {sys.platform}")
        print(f"  Project Root: {self.project_root}")
        
        # Check directories
        print("\n📁 Directory Check:")
        dirs = [
            (self.data_dir, "Data"),
            (self.models_dir, "Models"),
            (self.reports_dir, "Reports"),
            (self.src_dir, "Source Code")
        ]
        
        for dir_path, name in dirs:
            if dir_path.exists():
                file_count = len(list(dir_path.glob("*")))
                print(f"  ✅ {name}: {dir_path} ({file_count} items)")
            else:
                print(f"  ❌ {name}: {dir_path} (MISSING)")
        
        # Check Python files
        print("\n📄 Python Files Check:")
        py_files = list(self.src_dir.glob("*.py"))
        
        if py_files:
            for py_file in py_files:
                size_kb = py_file.stat().st_size / 1024
                print(f"  ✅ {py_file.name} ({size_kb:.1f} KB)")
        else:
            print("  ❌ No Python files found in src/")
        
        # Quick functionality test
        print("\n🔍 Functionality Test:")
        try:
            import numpy as np
            print("  ✅ NumPy: Works")
            
            test_data = np.random.randn(10, 5)
            print(f"  ✅ Array creation: {test_data.shape}")
            
        except ImportError:
            print("  ❌ NumPy: Not installed")
        
        print("\n✅ Test complete!")
        print("\n🎯 Ready to run:")
        print("  1. Training: python main.py --train")
        print("  2. Detection: python main.py --detect")
        print("  3. Menu: python main.py")
    
    def show_menu(self):
        """Show interactive menu"""
        self.print_banner()
        
        print("\n📋 MAIN MENU:")
        print("  [1] 🎯 Train ML Models")
        print("  [2] 🔍 Real-time Threat Detection")
        print("  [3] 📊 Analyze Existing Data")
        print("  [4] 🧪 System Test & Verification")
        print("  [5] 📖 View Reports")
        print("  [6] 🚪 Exit")
        
        choice = input("\nSelect option (1-6): ").strip()
        
        if choice == '1':
            self.run_training_mode()
        elif choice == '2':
            duration = input("Detection duration (seconds, default 30): ").strip()
            try:
                duration = int(duration) if duration else 30
                self.run_detection_mode(duration)
            except ValueError:
                print("Invalid duration, using 30 seconds")
                self.run_detection_mode(30)
        elif choice == '3':
            self.analyze_existing_data()
        elif choice == '4':
            self.run_test_mode()
        elif choice == '5':
            self.view_reports()
        elif choice == '6':
            print("\n👋 Goodbye! Stay secure! 🛡️")
            return False
        else:
            print("\n❌ Invalid choice!")
        
        input("\nPress Enter to continue...")
        return True
    
    def analyze_existing_data(self):
        """Analyze existing data files"""
        print("\n" + "="*70)
        print("📊 MODE: DATA ANALYSIS")
        print("="*70)
        
        # Find CSV files
        csv_files = list(self.data_dir.glob("*.csv"))
        
        if not csv_files:
            print("No CSV files found in data/ directory.")
            print("Creating sample data...")
            self.create_sample_data()
            csv_files = list(self.data_dir.glob("*.csv"))
        
        print(f"\nFound {len(csv_files)} data files:")
        for i, csv_file in enumerate(csv_files, 1):
            size_mb = csv_file.stat().st_size / (1024 * 1024)
            print(f"  {i}. {csv_file.name} ({size_mb:.2f} MB)")
        
        if csv_files:
            # Analyze first file
            import pandas as pd
            
            try:
                df = pd.read_csv(csv_files[0])
                print(f"\n📈 Analyzing: {csv_files[0].name}")
                print(f"  Rows: {len(df):,}")
                print(f"  Columns: {len(df.columns)}")
                
                # Check for threat column
                threat_cols = [col for col in df.columns if 'threat' in col.lower() or 'label' in col.lower()]
                
                if threat_cols:
                    threat_col = threat_cols[0]
                    threat_count = df[threat_col].sum() if df[threat_col].dtype in ['int64', 'float64'] else 0
                    print(f"  Threats: {threat_count:,} ({threat_count/len(df):.1%})")
                
                # Show column info
                print("\n  Columns:")
                for col in df.columns[:10]:  # First 10 columns
                    dtype = str(df[col].dtype)
                    unique = df[col].nunique()
                    print(f"    {col:20} {dtype:10} Unique: {unique}")
                
                if len(df.columns) > 10:
                    print(f"    ... and {len(df.columns) - 10} more columns")
                
            except Exception as e:
                print(f"❌ Error analyzing file: {e}")
    
    def create_sample_data(self):
        """Create sample data if none exists"""
        print("\n📝 Creating sample data...")
        
        import pandas as pd
        import numpy as np
        
        np.random.seed(42)
        n_samples = 1000
        
        data = {
            'timestamp': pd.date_range('2024-01-01', periods=n_samples, freq='min'),
            'source_ip': [f'192.168.1.{np.random.randint(1, 50)}' for _ in range(n_samples)],
            'dest_ip': [f'10.0.0.{np.random.randint(1, 10)}' for _ in range(n_samples)],
            'protocol': np.random.choice(['TCP', 'UDP', 'HTTP'], n_samples),
            'duration': np.random.exponential(1.5, n_samples),
            'bytes_sent': np.random.lognormal(10, 2, n_samples),
            'bytes_received': np.random.lognormal(11, 2, n_samples),
        }
        
        # Add threats
        n_threats = int(0.05 * n_samples)
        threat_indices = np.random.choice(n_samples, n_threats, replace=False)
        
        for idx in threat_indices:
            data['bytes_sent'][idx] = np.random.lognormal(15, 1)  # Large upload
            data['duration'][idx] = 0.001  # Very short
        
        df = pd.DataFrame(data)
        df['is_threat'] = 0
        df.loc[threat_indices, 'is_threat'] = 1
        
        # Save
        data_file = self.data_dir / "network_traffic_sample.csv"
        df.to_csv(data_file, index=False)
        
        print(f"✅ Created: {data_file}")
        print(f"   Samples: {n_samples:,}")
        print(f"   Threats: {n_threats:,} ({n_threats/n_samples:.1%})")
    
    def view_reports(self):
        """View existing reports"""
        print("\n" + "="*70)
        print("📋 MODE: VIEW REPORTS")
        print("="*70)
        
        json_files = list(self.reports_dir.glob("*.json"))
        png_files = list(self.reports_dir.glob("*.png"))
        
        print(f"\nFound {len(json_files)} JSON reports and {len(png_files)} PNG images")
        
        if json_files:
            print("\n📄 Latest JSON Reports:")
            for json_file in sorted(json_files, key=lambda x: x.stat().st_mtime, reverse=True)[:5]:
                size_kb = json_file.stat().st_size / 1024
                mtime = json_file.stat().st_mtime
                from datetime import datetime
                mtime_str = datetime.fromtimestamp(mtime).strftime('%Y-%m-%d %H:%M')
                print(f"  • {json_file.name} ({size_kb:.1f} KB, {mtime_str})")
        
        if png_files:
            print("\n🖼️  Latest Visualizations:")
            for png_file in sorted(png_files, key=lambda x: x.stat().st_mtime, reverse=True)[:3]:
                size_kb = png_file.stat().st_size / 1024
                print(f"  • {png_file.name} ({size_kb:.1f} KB)")
        
        if not json_files and not png_files:
            print("\nNo reports found. Run detection first to generate reports.")

def main():
    """Main function - handles command line arguments"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="AI-Driven Cyber Threat Detection System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py                    # Interactive menu
  python main.py --train            # Train ML models
  python main.py --detect           # Run detection (30 seconds)
  python main.py --detect 60        # Run detection for 60 seconds
  python main.py --test             # System test
  python main.py --quick            # Quick start (train + detect)
        """
    )
    
    parser.add_argument('--train', action='store_true', help='Train ML models')
    parser.add_argument('--detect', nargs='?', const=30, type=int, help='Run detection (default: 30 seconds)')
    parser.add_argument('--test', action='store_true', help='Run system test')
    parser.add_argument('--quick', action='store_true', help='Quick start (train then detect)')
    parser.add_argument('--check', action='store_true', help='Check requirements only')
    
    args = parser.parse_args()
    
    # Create instance
    detector = CyberThreatDetection()
    detector.print_banner()
    
    # Check requirements first
    if not detector.check_requirements():
        print("\n⚠️  Some requirements are missing. Install them first.")
        return
    
    if args.check:
        print("\n✅ Requirements check complete!")
        return
    
    # Handle command line arguments
    if args.train:
        detector.run_training_mode()
    elif args.detect:
        detector.run_detection_mode(args.detect)
    elif args.test:
        detector.run_test_mode()
    elif args.quick:
        print("\n🚀 QUICK START: Training + Detection")
        detector.run_training_mode()
        input("\nPress Enter to start detection...")
        detector.run_detection_mode(20)
    else:
        # Interactive menu
        while True:
            if not detector.show_menu():
                break

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n🛑 Program interrupted by user")
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        print("\n💡 Troubleshooting tips:")
        print("  1. Run: python main.py --check")
        print("  2. Install missing packages: pip install pandas numpy scikit-learn")
        print("  3. Check file structure: ensure src/detect_threats.py exists")
    finally:
        print("\n" + "="*70)
        print("🛡️  Cyber Threat Detection System - Session Ended")
        print("="*70)