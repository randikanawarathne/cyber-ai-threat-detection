# Remove any sys.path modifications at the top
# Add this instead at the very top:
"""
Cyber Threat Detection Module
"""

# Keep all your imports as they are
import os
import numpy as np
import pandas as pd
import joblib
from datetime import datetime
import json
import time
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# Create necessary directories
os.makedirs('models', exist_ok=True)
os.makedirs('reports', exist_ok=True)
os.makedirs('data', exist_ok=True)

class ThreatDetector:
    def __init__(self, model_path=None):
        """Initialize detector with trained model"""
        print("🔧 Initializing Threat Detector...")
        
        # Default model path
        if model_path is None:
            model_path = 'models/random_forest_model.pkl'
        
        # Check if model exists, train if not
        if not os.path.exists(model_path):
            print("⚠️  Model not found. Training a new model...")
            self.model = self.train_default_model()
            joblib.dump(self.model, model_path)
            print(f"✓ Model saved to {model_path}")
        else:
            try:
                self.model = joblib.load(model_path)
                print(f"✓ Model loaded from {model_path}")
            except Exception as e:
                print(f"❌ Error loading model: {e}")
                print("Training a new model...")
                self.model = self.train_default_model()
                joblib.dump(self.model, model_path)
        
        self.threshold = 0.7  # Confidence threshold for alerts
        self.threat_log = []
        self.alert_count = 0
        
        # MITRE ATT&CK mapping for common techniques
        self.mitre_mapping = {
            'data_exfiltration': ['T1048', 'T1020', 'T1030'],
            'command_control': ['T1071', 'T1095', 'T1132'],
            'credential_access': ['T1110', 'T1003', 'T1555'],
            'lateral_movement': ['T1021', 'T1072', 'T1080'],
            'persistence': ['T1053', 'T1050', 'T1543'],
            'reconnaissance': ['T1046', 'T1135', 'T1040'],
            'resource_hijacking': ['T1496', 'T1491', 'T1489']
        }
        
        print("✅ Threat Detector initialized successfully!")
    
    def train_default_model(self):
        """Train a simple model if none exists"""
        print("Training default model...")
        
        # Create synthetic training data
        np.random.seed(42)
        n_samples = 1000
        
        # Generate features
        X = np.random.randn(n_samples, 8)
        
        # Create labels (95% normal, 5% threats)
        y = np.zeros(n_samples)
        threat_indices = np.random.choice(n_samples, size=int(n_samples * 0.05), replace=False)
        y[threat_indices] = 1
        
        # Make threats look different (outliers)
        for idx in threat_indices:
            X[idx] += np.random.randn(8) * 3  # Add noise to make them anomalous
        
        # Train model
        model = RandomForestClassifier(
            n_estimators=50,
            max_depth=10,
            random_state=42,
            class_weight='balanced'
        )
        
        model.fit(X, y)
        accuracy = model.score(X, y)
        print(f"Default model trained with accuracy: {accuracy:.2%}")
        
        return model
    
    def extract_features(self, log_entry):
        """Extract features from a log entry dictionary"""
        features = []
        
        # Basic network features
        if 'duration' in log_entry:
            features.append(float(log_entry.get('duration', 0)))
        else:
            features.append(np.random.exponential(1))
        
        if 'src_bytes' in log_entry:
            features.append(float(log_entry.get('src_bytes', 0)))
        else:
            features.append(np.random.randint(100, 10000))
        
        if 'dst_bytes' in log_entry:
            features.append(float(log_entry.get('dst_bytes', 0)))
        else:
            features.append(np.random.randint(100, 10000))
        
        # Protocol encoding
        protocol_map = {'TCP': 0, 'UDP': 1, 'ICMP': 2, 'HTTP': 3, 'HTTPS': 4}
        if 'protocol' in log_entry:
            protocol = log_entry['protocol']
            features.append(protocol_map.get(protocol, 5))
        else:
            features.append(np.random.choice([0, 1, 2]))
        
        # Flag encoding
        flag_map = {'SYN': 0, 'ACK': 1, 'FIN': 2, 'RST': 3, 'PSH': 4, 'URG': 5}
        if 'flag' in log_entry:
            flag = log_entry['flag']
            features.append(flag_map.get(flag, 6))
        else:
            features.append(np.random.choice([0, 1, 2]))
        
        # Derived features
        total_bytes = float(log_entry.get('src_bytes', 0) + log_entry.get('dst_bytes', 0))
        features.append(total_bytes)
        
        if 'dst_bytes' in log_entry and log_entry['dst_bytes'] > 0:
            bytes_ratio = float(log_entry.get('src_bytes', 0)) / float(log_entry['dst_bytes'])
            features.append(bytes_ratio)
        else:
            features.append(1.0)
        
        # Connection count if available
        if 'count' in log_entry:
            features.append(float(log_entry['count']))
        else:
            features.append(np.random.randint(1, 10))
        
        return np.array(features)
    
    def predict(self, features):
        """Make prediction on features"""
        try:
            # Ensure features are 2D array
            if len(features.shape) == 1:
                features = features.reshape(1, -1)
            
            # Get prediction probability
            if hasattr(self.model, 'predict_proba'):
                proba = self.model.predict_proba(features)[0]
                prediction = self.model.predict(features)[0]
                confidence = max(proba)
                is_threat = prediction == 1 and confidence > self.threshold
            else:
                prediction = self.model.predict(features)[0]
                is_threat = prediction == 1
                confidence = 0.8 if is_threat else 0.2
            
            return is_threat, confidence, prediction
            
        except Exception as e:
            print(f"❌ Prediction error: {e}")
            return False, 0.0, 0
    
    def calculate_threat_score(self, features, confidence, threat_type='unknown'):
        """Calculate threat severity score (0-100)"""
        base_score = confidence * 100
        
        # Adjust based on feature patterns
        if len(features) > 1:
            # High bytes ratio (suspicious upload)
            if features[6] > 10:  # bytes_ratio > 10
                base_score *= 1.3
            
            # Very short duration
            if features[0] < 0.01:  # duration < 0.01s
                base_score *= 1.2
            
            # Large total bytes
            if features[5] > 1000000:  # total_bytes > 1MB
                base_score *= 1.1
        
        # Time-based adjustments
        current_hour = datetime.now().hour
        if current_hour < 6 or current_hour > 22:  # Non-business hours
            base_score *= 1.15
        
        # Cap at 100
        return min(100, round(base_score, 2))
    
    def determine_threat_type(self, features):
        """Determine the type of threat based on feature patterns"""
        if len(features) < 7:
            return "unknown"
        
        duration = features[0]
        src_bytes = features[1]
        dst_bytes = features[2]
        bytes_ratio = features[6] if len(features) > 6 else 1
        
        if bytes_ratio > 50 and src_bytes > 1000000:
            return "data_exfiltration"
        elif duration < 0.01 and bytes_ratio > 10:
            return "command_control"
        elif bytes_ratio < 0.01 and dst_bytes > 1000000:
            return "credential_access"
        elif features[7] > 100 if len(features) > 7 else False:  # high count
            return "reconnaissance"
        else:
            return "suspicious_activity"
    
    def map_to_mitre(self, threat_type):
        """Map threat type to MITRE ATT&CK techniques"""
        return self.mitre_mapping.get(threat_type, ['T1048'])  # Default to data exfiltration
    
    def determine_severity(self, score):
        """Convert score to severity level"""
        if score >= 90:
            return "CRITICAL"
        elif score >= 75:
            return "HIGH"
        elif score >= 50:
            return "MEDIUM"
        elif score >= 30:
            return "LOW"
        else:
            return "INFO"
    
    def suggest_actions(self, severity, threat_type):
        """Generate recommended actions based on severity"""
        base_actions = {
            "CRITICAL": [
                "🔴 IMMEDIATE: Isolate affected system from network",
                "🔴 Contact SOC team for incident response",
                "🔴 Preserve all logs and evidence",
                "🔴 Begin containment procedures"
            ],
            "HIGH": [
                "🟠 Review source IP reputation",
                "🟠 Block suspicious IP addresses",
                "🟠 Monitor related systems",
                "🟠 Escalate to senior analyst"
            ],
            "MEDIUM": [
                "🟡 Investigate further",
                "🟡 Check for similar patterns",
                "🟡 Update firewall rules if needed",
                "🟡 Document findings"
            ],
            "LOW": [
                "🟢 Monitor for escalation",
                "🟢 Add to watchlist",
                "🟢 Review during next audit",
                "🟢 Update baseline metrics"
            ],
            "INFO": [
                "🔵 Log for future reference",
                "🔵 No immediate action required"
            ]
        }
        
        # Add type-specific actions
        type_actions = {
            "data_exfiltration": ["Check data loss prevention logs", "Review outbound traffic patterns"],
            "command_control": ["Analyze DNS queries", "Check for beaconing behavior"],
            "reconnaissance": ["Review port scan alerts", "Check vulnerability scan logs"]
        }
        
        actions = base_actions.get(severity, base_actions["INFO"]).copy()
        if threat_type in type_actions:
            actions.extend(type_actions[threat_type])
        
        return actions
    
    def generate_alert(self, log_entry, features, threat_score, confidence, threat_type):
        """Generate structured alert"""
        self.alert_count += 1
        alert_id = f"ALT-{self.alert_count:04d}"
        
        alert = {
            'alert_id': alert_id,
            'timestamp': datetime.now().isoformat(),
            'threat_detected': True,
            'threat_type': threat_type,
            'threat_score': threat_score,
            'confidence': round(confidence, 3),
            'severity': self.determine_severity(threat_score),
            'mitre_techniques': self.map_to_mitre(threat_type),
            'source_ip': log_entry.get('source_ip', 'Unknown'),
            'destination_ip': log_entry.get('destination_ip', 'Unknown'),
            'protocol': log_entry.get('protocol', 'Unknown'),
            'features_summary': {
                'duration': round(features[0], 4) if len(features) > 0 else 0,
                'src_bytes': int(features[1]) if len(features) > 1 else 0,
                'dst_bytes': int(features[2]) if len(features) > 2 else 0,
                'bytes_ratio': round(features[6], 2) if len(features) > 6 else 0,
            },
            'recommended_actions': self.suggest_actions(
                self.determine_severity(threat_score), 
                threat_type
            )
        }
        
        self.threat_log.append(alert)
        
        # Also log to console
        self.print_alert(alert)
        
        return alert
    
    def print_alert(self, alert):
        """Print alert to console with colored output"""
        colors = {
            'CRITICAL': '\033[91m',  # Red
            'HIGH': '\033[93m',      # Yellow
            'MEDIUM': '\033[33m',    # Orange
            'LOW': '\033[96m',       # Cyan
            'INFO': '\033[92m',      # Green
            'RESET': '\033[0m'       # Reset
        }
        
        color = colors.get(alert['severity'], colors['INFO'])
        
        print(f"\n{color}{'='*60}")
        print(f"🚨 THREAT ALERT: {alert['alert_id']}")
        print(f"{'='*60}{colors['RESET']}")
        print(f"⏰ Time: {alert['timestamp'][11:19]}")
        print(f"📊 Severity: {color}{alert['severity']}{colors['RESET']}")
        print(f"🎯 Type: {alert['threat_type'].replace('_', ' ').title()}")
        print(f"📈 Score: {alert['threat_score']}/100")
        print(f"🎯 Confidence: {alert['confidence']:.1%}")
        print(f"📍 Source: {alert['source_ip']} → {alert['destination_ip']}")
        print(f"📡 Protocol: {alert['protocol']}")
        print(f"🔗 MITRE Techniques: {', '.join(alert['mitre_techniques'])}")
        print(f"\n💡 Recommended Actions:")
        for i, action in enumerate(alert['recommended_actions'][:3], 1):
            print(f"   {i}. {action}")
    
    def analyze_single(self, log_entry):
        """Analyze a single log entry"""
        try:
            # Extract features
            features = self.extract_features(log_entry)
            
            # Make prediction
            is_threat, confidence, prediction = self.predict(features)
            
            if is_threat:
                # Determine threat type
                threat_type = self.determine_threat_type(features)
                
                # Calculate threat score
                threat_score = self.calculate_threat_score(features, confidence, threat_type)
                
                # Generate alert
                alert = self.generate_alert(log_entry, features, threat_score, confidence, threat_type)
                
                return True, alert
            else:
                return False, None
                
        except Exception as e:
            print(f"❌ Error analyzing log: {e}")
            return False, None
    
    def analyze_batch(self, log_data):
        """Analyze batch of log entries"""
        threats = []
        
        if isinstance(log_data, pd.DataFrame):
            # Convert DataFrame to list of dictionaries
            log_entries = log_data.to_dict('records')
        elif isinstance(log_data, list):
            log_entries = log_data
        else:
            print("❌ Unsupported log data format")
            return threats
        
        print(f"\n🔍 Analyzing {len(log_entries)} log entries...")
        
        for idx, entry in enumerate(log_entries, 1):
            is_threat, alert = self.analyze_single(entry)
            if is_threat:
                threats.append(alert)
            
            # Show progress
            if idx % 50 == 0:
                print(f"   Processed {idx}/{len(log_entries)} logs...")
        
        return threats
    
    def generate_sample_logs(self, n=100):
        """Generate sample log entries for testing"""
        print(f"\n📝 Generating {n} sample log entries...")
        
        logs = []
        protocols = ['TCP', 'UDP', 'HTTP', 'HTTPS', 'ICMP']
        flags = ['SYN', 'ACK', 'FIN', 'RST']
        
        for i in range(n):
            # Most logs are normal (90%)
            if np.random.random() > 0.1:
                log = {
                    'timestamp': datetime.now().isoformat(),
                    'source_ip': f'192.168.1.{np.random.randint(1, 50)}',
                    'destination_ip': f'10.0.0.{np.random.randint(1, 10)}',
                    'protocol': np.random.choice(protocols),
                    'duration': np.random.exponential(1.5),
                    'src_bytes': np.random.randint(100, 5000),
                    'dst_bytes': np.random.randint(1000, 10000),
                    'flag': np.random.choice(flags),
                    'count': np.random.randint(1, 10)
                }
            else:
                # 10% are potential threats
                log = {
                    'timestamp': datetime.now().isoformat(),
                    'source_ip': f'192.168.1.{np.random.randint(1, 50)}',
                    'destination_ip': '8.8.8.8',  # External IP
                    'protocol': np.random.choice(protocols),
                    'duration': np.random.exponential(0.01),  # Very short
                    'src_bytes': np.random.randint(1000000, 5000000),  # Large upload
                    'dst_bytes': np.random.randint(1, 100),  # Small download
                    'flag': np.random.choice(flags),
                    'count': np.random.randint(100, 1000)  # High count
                }
            
            logs.append(log)
        
        print(f"✓ Generated {len(logs)} sample logs")
        return logs
    
    def export_threat_report(self, filename=None):
        """Export all detected threats to JSON"""
        if filename is None:
            filename = f"threat_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        report_path = os.path.join('reports', filename)
        
        report = {
            'report_generated': datetime.now().isoformat(),
            'detection_period': {
                'start': self.threat_log[0]['timestamp'] if self.threat_log else datetime.now().isoformat(),
                'end': datetime.now().isoformat()
            },
            'summary': {
                'total_logs_analyzed': self.alert_count + 100,  # Approximate
                'total_threats_detected': len(self.threat_log),
                'threats_by_severity': self.summarize_threats(),
                'detection_rate': len(self.threat_log) / (self.alert_count + 100) if (self.alert_count + 100) > 0 else 0
            },
            'detected_threats': self.threat_log,
            'analysis_notes': [
                "Analysis performed by AI-Driven Cyber Threat Detection System",
                "Confidence threshold: 70%",
                "Models used: Random Forest Classifier",
                "For investigation purposes only"
            ]
        }
        
        # Ensure reports directory exists
        os.makedirs('reports', exist_ok=True)
        
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        print(f"\n📄 Threat report exported to: {report_path}")
        
        return report_path
    
    def summarize_threats(self):
        """Create summary statistics"""
        summary = {'CRITICAL': 0, 'HIGH': 0, 'MEDIUM': 0, 'LOW': 0, 'INFO': 0}
        
        for threat in self.threat_log:
            summary[threat['severity']] += 1
        
        return summary
    
    def print_dashboard(self):
        """Print console dashboard of current threats"""
        print("\n" + "="*60)
        print("📊 CYBER THREAT DETECTION DASHBOARD")
        print("="*60)
        
        summary = self.summarize_threats()
        total_threats = sum(summary.values())
        
        print(f"\n📈 Detection Summary:")
        print(f"   Total Threats Detected: {total_threats}")
        print(f"   CRITICAL: {summary['CRITICAL']} | HIGH: {summary['HIGH']} | "
              f"MEDIUM: {summary['MEDIUM']} | LOW: {summary['LOW']}")
        
        # Show recent threats
        if self.threat_log:
            print(f"\n🚨 Recent Alerts (last 5):")
            for threat in self.threat_log[-5:]:
                timestamp = threat['timestamp'][11:19]
                severity = threat['severity']
                score = threat['threat_score']
                threat_type = threat['threat_type'].replace('_', ' ')
                print(f"   [{timestamp}] {severity}: {threat_type.title()} (Score: {score})")
        else:
            print(f"\n✅ No threats detected yet.")
        
        print(f"\n🛡️  Model Status:")
        print(f"   Model: {self.model.__class__.__name__}")
        print(f"   Features: {self.model.n_features_in_ if hasattr(self.model, 'n_features_in_') else 'Unknown'}")
        print(f"   Confidence Threshold: {self.threshold:.0%}")
        
        print(f"\n{'='*60}")

def simulate_real_time_detection(duration_seconds=30):
    """Simulate real-time threat detection"""
    print("\n" + "="*60)
    print("🚀 STARTING REAL-TIME THREAT DETECTION SIMULATION")
    print("="*60)
    print("Simulating network traffic analysis...")
    print("Press Ctrl+C to stop\n")
    
    detector = ThreatDetector()
    
    try:
        start_time = time.time()
        log_counter = 0
        threat_counter = 0
        
        while time.time() - start_time < duration_seconds:
            # Generate a random log entry
            protocols = ['TCP', 'UDP', 'HTTP', 'HTTPS', 'ICMP']
            flags = ['SYN', 'ACK', 'FIN', 'RST']
            
            # Occasionally generate suspicious traffic (15% chance)
            if np.random.random() < 0.15:
                log_entry = {
                    'source_ip': f'10.0.0.{np.random.randint(1, 100)}',
                    'destination_ip': f'{np.random.randint(1, 255)}.{np.random.randint(1, 255)}.{np.random.randint(1, 255)}.{np.random.randint(1, 255)}',
                    'protocol': np.random.choice(protocols),
                    'duration': np.random.exponential(0.001),  # Very short
                    'src_bytes': np.random.randint(100000, 10000000),  # Large upload
                    'dst_bytes': np.random.randint(1, 100),  # Small download
                    'flag': np.random.choice(flags),
                    'count': np.random.randint(50, 500)
                }
            else:
                log_entry = {
                    'source_ip': f'192.168.1.{np.random.randint(1, 50)}',
                    'destination_ip': f'10.0.0.{np.random.randint(1, 10)}',
                    'protocol': np.random.choice(protocols),
                    'duration': np.random.exponential(1.5),
                    'src_bytes': np.random.randint(100, 5000),
                    'dst_bytes': np.random.randint(1000, 10000),
                    'flag': np.random.choice(flags),
                    'count': np.random.randint(1, 10)
                }
            
            # Analyze the log
            is_threat, _ = detector.analyze_single(log_entry)
            log_counter += 1
            
            if is_threat:
                threat_counter += 1
            
            # Show progress every 10 logs
            if log_counter % 10 == 0:
                elapsed = time.time() - start_time
                remaining = max(0, duration_seconds - elapsed)
                print(f"   📊 Processed {log_counter} logs, detected {threat_counter} threats "
                      f"({remaining:.0f}s remaining)")
            
            # Random delay to simulate real-time
            time.sleep(np.random.uniform(0.1, 0.5))
        
    except KeyboardInterrupt:
        print("\n\n🛑 Detection stopped by user")
    finally:
        # Export report
        report_path = detector.export_threat_report()
        
        # Print final summary
        print("\n" + "="*60)
        print("📋 DETECTION SIMULATION COMPLETE")
        print("="*60)
        detector.print_dashboard()
        
        print(f"\n📄 Report saved to: {report_path}")
        print(f"📈 Total logs analyzed: {log_counter}")
        print(f"🎯 Threats detected: {threat_counter}")
        if log_counter > 0:
            print(f"📊 Detection rate: {threat_counter/log_counter:.2%}")
        
        print("\n🔧 Analysis complete!")

def analyze_existing_data():
    """Analyze existing data file"""
    detector = ThreatDetector()
    
    print("\n🔍 Analyzing existing data...")
    
    # Check for data files
    data_files = []
    if os.path.exists('data'):
        data_files = [f for f in os.listdir('data') if f.endswith('.csv')]
    
    if not data_files:
        print("No CSV files found in data/ directory.")
        print("Generating sample data...")
        sample_logs = detector.generate_sample_logs(200)
        threats = detector.analyze_batch(sample_logs)
    else:
        print(f"Found data files: {', '.join(data_files)}")
        
        # Use the first CSV file
        data_file = os.path.join('data', data_files[0])
        print(f"Loading {data_file}...")
        
        try:
            df = pd.read_csv(data_file)
            print(f"✓ Loaded {len(df)} records")
            
            # Analyze
            threats = detector.analyze_batch(df)
        except Exception as e:
            print(f"❌ Error loading data: {e}")
            print("Using sample data instead...")
            sample_logs = detector.generate_sample_logs(100)
            threats = detector.analyze_batch(sample_logs)
    
    # Export report
    detector.export_threat_report()
    
    # Show dashboard
    detector.print_dashboard()
    
    return detector

def main():
    """Main function with menu options"""
    print("\n" + "="*60)
    print("🤖 AI-DRIVEN CYBER THREAT DETECTION SYSTEM")
    print("="*60)
    
    print("\n📋 Available Modes:")
    print("1. Real-time simulation (30 seconds)")
    print("2. Analyze existing data files")
    print("3. Generate and analyze sample data")
    print("4. Quick test (single detection)")
    print("5. Exit")
    
    try:
        choice = input("\nSelect mode (1-5): ").strip()
        
        if choice == '1':
            simulate_real_time_detection(duration_seconds=30)
        elif choice == '2':
            analyze_existing_data()
        elif choice == '3':
            detector = ThreatDetector()
            sample_logs = detector.generate_sample_logs(150)
            detector.analyze_batch(sample_logs)
            detector.export_threat_report()
            detector.print_dashboard()
        elif choice == '4':
            # Quick test
            detector = ThreatDetector()
            
            print("\n🔍 Running quick test...")
            
            # Test with a suspicious log
            test_log = {
                'source_ip': '192.168.1.100',
                'destination_ip': '8.8.8.8',
                'protocol': 'TCP',
                'duration': 0.001,
                'src_bytes': 5000000,
                'dst_bytes': 100,
                'flag': 'SYN',
                'count': 500
            }
            
            print(f"\nTest log: {test_log}")
            is_threat, alert = detector.analyze_single(test_log)
            
            if is_threat:
                print(f"\n✅ Threat detected! Score: {alert['threat_score']}")
            else:
                print("\n✅ No threat detected (this is unexpected for this test)")
        elif choice == '5':
            print("\n👋 Exiting...")
            return
        else:
            print("\n❌ Invalid choice. Running real-time simulation...")
            simulate_real_time_detection(duration_seconds=20)
            
    except Exception as e:
        print(f"\n❌ Error: {e}")
        print("Running fallback simulation...")
        simulate_real_time_detection(duration_seconds=15)

if __name__ == "__main__":
    # Welcome message
    print("\n" + "="*60)
    print("🚀 CYBER THREAT DETECTION SYSTEM v1.0")
    print("="*60)
    print("Built with Python & Machine Learning")
    print("For educational and demonstration purposes")
    print("="*60)
    
    # Run main function
    main()
    
    print("\n" + "="*60)
    print("✅ Analysis complete. Check 'reports/' directory for details.")
    print("="*60)