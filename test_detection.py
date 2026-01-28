#!/usr/bin/env python3
"""
Quick test of the threat detection system
"""
import sys
import os

# Ensure we're in the right directory
if not os.path.exists('src'):
    print("Please run from project root directory")
    sys.exit(1)

# Add src to Python path
sys.path.append('.')

try:
    from src.detect_threats import ThreatDetector
    
    print("🧪 Testing Threat Detection System...")
    print("=" * 50)
    
    # Initialize detector
    detector = ThreatDetector()
    
    # Test a suspicious log
    test_log = {
        'source_ip': '192.168.1.50',
        'destination_ip': '142.250.185.174',  # Google
        'protocol': 'TCP',
        'duration': 0.005,
        'src_bytes': 2500000,
        'dst_bytes': 50,
        'flag': 'ACK',
        'count': 300
    }
    
    print(f"\n📝 Test Log: {test_log}")
    print("\n🔍 Analyzing...")
    
    is_threat, alert = detector.analyze_single(test_log)
    
    if is_threat:
        print("\n✅ SUCCESS: Threat detected!")
        print(f"   Threat Score: {alert['threat_score']}")
        print(f"   Severity: {alert['severity']}")
    else:
        print("\n❌ No threat detected (unexpected for this test)")
    
    # Export report
    detector.export_threat_report('test_report.json')
    print(f"\n📄 Report saved to: reports/test_report.json")
    
    print("\n✅ Test complete!")
    
except ImportError as e:
    print(f"❌ Import error: {e}")
    print("\nMake sure packages are installed:")
    print("pip install pandas numpy scikit-learn joblib")
except Exception as e:
    print(f"❌ Error: {e}")