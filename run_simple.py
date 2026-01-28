#!/usr/bin/env python3
"""
Direct runner - calls detect_threats.py as a subprocess
"""
import subprocess
import sys
import os

def run_directly():
    """Run detect_threats.py directly"""
    script_path = os.path.join('src', 'detect_threats.py')
    
    if not os.path.exists(script_path):
        print(f"❌ File not found: {script_path}")
        print("Current directory:", os.getcwd())
        return
    
    print(f"🚀 Running: {script_path}")
    print("="*60)
    
    # Run the script
    result = subprocess.run([sys.executable, script_path], 
                          capture_output=False,
                          text=True)
    
    if result.returncode != 0:
        print(f"\n❌ Script exited with code: {result.returncode}")

def run_with_menu():
    """Run with menu options"""
    print("\n📋 CYBER THREAT DETECTOR - MENU")
    print("="*40)
    print("1. Run real-time simulation")
    print("2. Analyze existing data")
    print("3. Generate sample data and analyze")
    print("4. Quick test")
    print("5. Exit")
    
    choice = input("\nSelect (1-5): ").strip()
    
    script_path = os.path.join('src', 'detect_threats.py')
    
    if choice == '1':
        subprocess.run([sys.executable, script_path, "--mode", "realtime"])
    elif choice == '2':
        subprocess.run([sys.executable, script_path, "--mode", "analyze"])
    elif choice == '3':
        subprocess.run([sys.executable, script_path, "--mode", "sample"])
    elif choice == '4':
        subprocess.run([sys.executable, script_path, "--mode", "test"])
    else:
        print("👋 Goodbye!")

if __name__ == "__main__":
    # Check if running from project root
    if not os.path.exists('src'):
        print("❌ Please run from project root directory")
        print("   Current directory should contain 'src/' folder")
        sys.exit(1)
    
    print("🔧 Cyber Threat Detection System - Direct Runner")
    
    # Option 1: Run directly with default mode
    # run_directly()
    
    # Option 2: Run with menu
    run_with_menu()