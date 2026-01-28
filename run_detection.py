#!/usr/bin/env python3
"""
Simple runner for Cyber Threat Detection System
"""
import sys
import os

def setup_environment():
    """Setup Python path and check dependencies"""
    # Get the project root directory
    project_root = os.path.dirname(os.path.abspath(__file__))
    src_path = os.path.join(project_root, 'src')
    
    # Add to Python path
    if src_path not in sys.path:
        sys.path.insert(0, src_path)
    
    print(f"📁 Project root: {project_root}")
    print(f"📁 Source path: {src_path}")
    
    # Check if src/detect_threats.py exists
    detect_threats_path = os.path.join(src_path, 'detect_threats.py')
    if not os.path.exists(detect_threats_path):
        print(f"❌ File not found: {detect_threats_path}")
        return False
    
    return True

def import_detection_module():
    """Import the detection module"""
    try:
        # Method 1: Direct import
        import src.detect_threats as detect_threats
        print("✅ Imported using: import src.detect_threats")
        return detect_threats
    except ImportError:
        try:
            # Method 2: Import from module
            from src import detect_threats
            print("✅ Imported using: from src import detect_threats")
            return detect_threats
        except ImportError:
            try:
                # Method 3: Import as module
                import importlib.util
                spec = importlib.util.spec_from_file_location(
                    "detect_threats",
                    os.path.join(os.path.dirname(__file__), "src", "detect_threats.py")
                )
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)
                print("✅ Imported using importlib")
                return module
            except Exception as e:
                print(f"❌ All import methods failed: {e}")
                return None

def main():
    """Main function"""
    print("\n" + "="*60)
    print("🤖 CYBER THREAT DETECTION SYSTEM - RUNNER")
    print("="*60)
    
    # Setup environment
    if not setup_environment():
        print("\n❌ Setup failed. Please check:")
        print("   1. Are you in the project root directory?")
        print("   2. Does src/detect_threats.py exist?")
        print("   3. Are all requirements installed?")
        return
    
    # Import module
    print("\n🔧 Importing detection module...")
    detect_threats = import_detection_module()
    
    if detect_threats is None:
        print("\n❌ Could not import detection module.")
        print("\n📋 Try running directly instead:")
        print("   python src/detect_threats.py")
        return
    
    # Run the detection
    print("\n🚀 Starting detection system...")
    print("="*60)
    
    try:
        # Check if main function exists
        if hasattr(detect_threats, 'main'):
            detect_threats.main()
        elif hasattr(detect_threats, 'simulate_real_time_detection'):
            detect_threats.simulate_real_time_detection(duration_seconds=20)
        else:
            print("❌ No runnable function found in module")
            print("\nAvailable functions:")
            for attr in dir(detect_threats):
                if not attr.startswith('_'):
                    print(f"  - {attr}")
    except KeyboardInterrupt:
        print("\n\n🛑 Stopped by user")
    except Exception as e:
        print(f"\n❌ Error during execution: {e}")
        print("\n💡 Try running: python src/detect_threats.py")

if __name__ == "__main__":
    main()