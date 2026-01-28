def create_directories():
    """Create necessary directories if they don't exist"""
    directories = ['models', 'reports', 'data']
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        print(f"✓ Created directory: {directory}")

# Call at the start of main functions
create_directories()
# Add at the top of detect_threats.py
import numpy as np
import pandas as pd
import joblib
from datetime import datetime
import json
import os  # Add this
import sys
sys.path.append('.')  # Allows importing from src

"""
Real-time threat detection and alerting system
"""
import numpy as np
import pandas as pd
import joblib
from datetime import datetime
import json

class ThreatDetector:
    def __init__(self, model_path=None):
    """ Initialize detector with trained model """

    if model_path is None:
        model_path = 'models/random_forest_model.pkl'
    
    if not os.path.exists(model_path):
        print(f"⚠️  Model not found at {model_path}. Training a new one...")
        self.model = self.train_default_model()
    else:
        self.model = joblib.load(model_path)
    
    self.threshold = 0.5
    self.threat_log = []
    
def train_default_model(self):
    """Train a simple model if none exists"""
    from sklearn.ensemble import RandomForestClassifier
    model = RandomForestClassifier(n_estimators=10, random_state=42)
    # Simple dummy training for demo
    X_dummy = np.random.randn(100, 10)
    y_dummy = np.random.randint(0, 2, 100)
    model.fit(X_dummy, y_dummy)
    
    os.makedirs('models', exist_ok=True)
    joblib.dump(model, 'models/random_forest_model.pkl')
    return model
    
    def predict(self, features):
        """Make prediction on single sample"""
        if hasattr(self.model, 'predict_proba'):
            proba = self.model.predict_proba([features])[0]
            prediction = self.model.predict([features])[0]
            confidence = max(proba)
            is_threat = prediction == 1
        else:  # For Isolation Forest
            prediction = self.model.predict([features])[0]
            is_threat = prediction == -1
            confidence = 0.8 if is_threat else 0.2
        
        return is_threat, confidence
    
    def calculate_threat_score(self, features, confidence, threat_type):
        """Calculate severity score (0-100)"""
        base_score = confidence * 100
        
        # Adjust based on feature patterns
        if 'bytes_ratio' in features:
            if features['bytes_ratio'] > 10:  # High outbound traffic
                base_score *= 1.3
        
        # Time-based adjustments
        current_hour = datetime.now().hour
        if current_hour < 6 or current_hour > 22:  # Non-business hours
            base_score *= 1.2
        
        return min(100, base_score)
    
    def generate_alert(self, features, threat_score, confidence):
        """Generate structured alert"""
        alert = {
            'timestamp': datetime.now().isoformat(),
            'threat_detected': True,
            'threat_score': round(threat_score, 2),
            'confidence': round(confidence, 3),
            'severity': self.determine_severity(threat_score),
            'mitre_techniques': self.map_to_mitre(features),
            'features': {k: round(float(v), 4) for k, v in features.items() 
                        if isinstance(v, (int, float, np.number))},
            'recommended_actions': self.suggest_actions(threat_score)
        }
        
        self.threat_log.append(alert)
        return alert
    
    def determine_severity(self, score):
        """Convert score to severity level"""
        if score >= 80:
            return 'CRITICAL'
        elif score >= 60:
            return 'HIGH'
        elif score >= 40:
            return 'MEDIUM'
        else:
            return 'LOW'
    
    def map_to_mitre(self, features):
        """Map detected threat to MITRE ATT&CK techniques"""
        techniques = []
        
        # Simple heuristic mapping (extend based on your features)
        if features.get('duration', 0) < 0.1:
            techniques.extend(['T1203', 'T1059'])  # Exploitation, Command Line
            
        if features.get('src_bytes', 0) > 1000000:
            techniques.append('T1048')  # Exfiltration
            
        return list(set(techniques))[:3]  # Return unique, max 3
    
    def suggest_actions(self, score):
        """Generate recommended actions based on severity"""
        if score >= 80:
            return [
                "Immediate isolation of affected system",
                "Notify SOC team immediately",
                "Begin incident response procedures"
            ]
        elif score >= 60:
            return [
                "Investigate source IP",
                "Review related logs",
                "Monitor for additional activity"
            ]
        else:
            return ["Log for further analysis", "Update baseline if needed"]
    
    def analyze_batch(self, log_data):
        """Analyze batch of log entries"""
        threats = []
        
        for idx, row in log_data.iterrows():
            features = row.to_dict()
            
            # Convert to numpy array for model (adjust based on your model)
            feature_vector = list(features.values())
            
            is_threat, confidence = self.predict(feature_vector)
            
            if is_threat:
                threat_score = self.calculate_threat_score(features, confidence, 'suspicious_activity')
                alert = self.generate_alert(features, threat_score, confidence)
                threats.append(alert)
        
        return threats
    
    def export_threat_report(self, filename='threat_report.json'):
        """Export all detected threats to JSON"""
        report = {
            'generated_at': datetime.now().isoformat(),
            'total_threats': len(self.threat_log),
            'threats_by_severity': self.summarize_threats(),
            'detected_threats': self.threat_log
        }
        
        with open(f'reports/{filename}', 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        return report
    
    def summarize_threats(self):
        """Create summary statistics"""
        summary = {'CRITICAL': 0, 'HIGH': 0, 'MEDIUM': 0, 'LOW': 0}
        
        for threat in self.threat_log:
            summary[threat['severity']] += 1
        
        return summary
    
    def print_dashboard(self):
        """Print console dashboard of current threats"""
        print("\n" + "="*60)
        print("CYBER THREAT DETECTION DASHBOARD")
        print("="*60)
        
        summary = self.summarize_threats()
        
        print(f"\n📊 Threat Summary:")
        print(f"   CRITICAL: {summary['CRITICAL']} | HIGH: {summary['HIGH']} | "
              f"MEDIUM: {summary['MEDIUM']} | LOW: {summary['LOW']}")
        print(f"   Total Threats Detected: {len(self.threat_log)}")
        
        if self.threat_log:
            print(f"\n🚨 Recent Threats:")
            for threat in self.threat_log[-5:]:  # Show last 5 threats
                print(f"   [{threat['timestamp'][11:19]}] "
                      f"{threat['severity']}: Score {threat['threat_score']} "
                      f"(Confidence: {threat['confidence']:.2f})")
        
        print(f"\n{'='*60}")

def simulate_real_time_detection():
    """Simulate real-time threat detection"""
    detector = ThreatDetector()
    
    print("🔍 Starting real-time threat detection simulation...")
    print("Press Ctrl+C to stop\n")
    
    try:
        while True:
            # Generate random network traffic features
            features = {
                'duration': np.random.exponential(1),
                'src_bytes': np.random.randint(0, 1000000),
                'dst_bytes': np.random.randint(0, 1000000),
                'protocol_type': np.random.choice([0, 1, 2]),
                'flag': np.random.choice([0, 1, 2, 3]),
                'wrong_fragment': np.random.randint(0, 5),
                'urgent': np.random.randint(0, 2),
                'count': np.random.randint(1, 100),
                'srv_count': np.random.randint(1, 50),
                'dst_host_srv_count': np.random.randint(1, 100)
            }
            
            # Add occasional anomalies
            if np.random.random() < 0.05:  # 5% chance of anomaly
                features['src_bytes'] = np.random.randint(1000000, 10000000)
                features['duration'] = 0.01
            
            # Convert to feature vector (in correct order for your model)
            feature_vector = list(features.values())
            
            # Detect threat
            is_threat, confidence = detector.predict(feature_vector)
            
            if is_threat:
                threat_score = detector.calculate_threat_score(features, confidence, 'suspicious_activity')
                alert = detector.generate_alert(features, threat_score, confidence)
                
                # Print alert
                print(f"\n⚠️  THREAT DETECTED! [{alert['severity']}]")
                print(f"   Score: {alert['threat_score']} | Confidence: {alert['confidence']:.3f}")
                print(f"   MITRE Techniques: {', '.join(alert['mitre_techniques']) if alert['mitre_techniques'] else 'None'}")
                print(f"   Recommended: {alert['recommended_actions'][0]}")
            
            # Update dashboard every 10 iterations
            if len(detector.threat_log) % 10 == 0 and len(detector.threat_log) > 0:
                detector.print_dashboard()
                
    except KeyboardInterrupt:
        print("\n\n🛑 Detection stopped by user")
        
        # Export final report
        report = detector.export_threat_report()
        print(f"\n📄 Threat report exported to reports/threat_report.json")
        
        # Print final summary
        detector.print_dashboard()

if __name__ == "__main__":
    # Run simulation
    simulate_real_time_detection()