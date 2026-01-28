"""
Train multiple ML models for threat detection
"""
import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

def create_directories():
    """Create necessary directories if they don't exist"""
    directories = ['models', 'reports', 'data']
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        print(f"✓ Created directory: {directory}")

class ThreatDetectionTrainer:
    def __init__(self):
        self.models = {
            'logistic_regression': LogisticRegression(max_iter=1000, class_weight='balanced'),
            'random_forest': RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced'),
            'isolation_forest': IsolationForest(contamination=0.1, random_state=42)
        }
        self.trained_models = {}
        
    def train_models(self, X_train, y_train, model_type='all'):
        """Train specified models"""
        if model_type == 'all':
            models_to_train = self.models.keys()
        else:
            models_to_train = [model_type]
        
        for model_name in models_to_train:
            print(f"\n{'='*50}")
            print(f"Training {model_name.replace('_', ' ').title()}")
            print(f"{'='*50}")
            
            model = self.models[model_name]
            
            if model_name == 'isolation_forest':
                # Isolation Forest is unsupervised, expects only X
                model.fit(X_train)
                y_pred = model.predict(X_train)
                # Convert to binary (1 for normal, -1 for anomaly)
                y_pred_binary = [0 if x == 1 else 1 for x in y_pred]
                print("Isolation Forest training completed")
                print(f"Detected anomalies: {sum(y_pred_binary)}/{len(y_pred_binary)}")
            else:
                # Supervised models
                model.fit(X_train, y_train)
                y_pred = model.predict(X_train)
                print(f"Training accuracy: {model.score(X_train, y_train):.4f}")
                
                # Cross-validation
                cv_scores = cross_val_score(model, X_train, y_train, cv=5)
                print(f"Cross-validation scores: {cv_scores}")
                print(f"Mean CV score: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
            
            self.trained_models[model_name] = model
            os.makedirs('models', exist_ok=True)
            joblib.dump(model, f'models/{model_name}_model.pkl')
            print(f"✓ Model saved to models/{model_name}_model.pkl")
        
        return self.trained_models
    
    def evaluate_models(self, X_test, y_test):
        """Evaluate all trained models on test set"""
        results = {}
        
        for model_name, model in self.trained_models.items():
            print(f"\n{'='*50}")
            print(f"Evaluating {model_name.replace('_', ' ').title()}")
            print(f"{'='*50}")
            
            if model_name == 'isolation_forest':
                y_pred = model.predict(X_test)
                y_pred_binary = [0 if x == 1 else 1 for x in y_pred]
                
                # For evaluation, we need to compare with ground truth
                from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
                
                acc = accuracy_score(y_test, y_pred_binary)
                precision = precision_score(y_test, y_pred_binary)
                recall = recall_score(y_test, y_pred_binary)
                f1 = f1_score(y_test, y_pred_binary)
                
                print(f"Accuracy: {acc:.4f}")
                print(f"Precision: {precision:.4f}")
                print(f"Recall: {recall:.4f}")
                print(f"F1-Score: {f1:.4f}")
                
                results[model_name] = {
                    'accuracy': acc,
                    'precision': precision,
                    'recall': recall,
                    'f1_score': f1
                }
                
                # Plot confusion matrix
                self.plot_confusion_matrix(y_test, y_pred_binary, model_name)
                
            else:
                y_pred = model.predict(X_test)
                y_pred_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else None
                
                print("Classification Report:")
                print(classification_report(y_test, y_pred))
                
                if y_pred_proba is not None:
                    roc_auc = roc_auc_score(y_test, y_pred_proba)
                    print(f"ROC-AUC Score: {roc_auc:.4f}")
                
                results[model_name] = classification_report(y_test, y_pred, output_dict=True)
                
                # Plot confusion matrix
                self.plot_confusion_matrix(y_test, y_pred, model_name)
        
        return results
    
    def plot_confusion_matrix(self, y_true, y_pred, model_name):
        """Plot confusion matrix for model evaluation"""
        os.makedirs('reports', exist_ok=True)
        cm = confusion_matrix(y_true, y_pred)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=['Normal', 'Threat'],
                    yticklabels=['Normal', 'Threat'])
        plt.title(f'Confusion Matrix - {model_name.replace("_", " ").title()}')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        plt.savefig(f'reports/{model_name}_confusion_matrix.png', dpi=150)
        plt.show()
    
    def feature_importance_analysis(self, model_name, feature_names):
        """Analyze and plot feature importance"""
        if model_name in self.trained_models and hasattr(self.trained_models[model_name], 'feature_importances_'):
            importances = self.trained_models[model_name].feature_importances_
            indices = np.argsort(importances)[::-1]
            
            plt.figure(figsize=(12, 8))
            plt.title(f'Feature Importances - {model_name.replace("_", " ").title()}')
            plt.bar(range(min(20, len(feature_names))), importances[indices[:20]])
            plt.xticks(range(min(20, len(feature_names))), 
                      [feature_names[i] for i in indices[:20]], rotation=45, ha='right')
            plt.tight_layout()
            os.makedirs('reports', exist_ok=True)
            plt.savefig(f'reports/{model_name}_feature_importance.png', dpi=150)
            plt.show()
            
            # Print top features
            print("\nTop 10 Most Important Features:")
            for i in range(min(10, len(feature_names))):
                print(f"{i+1}. {feature_names[indices[i]]}: {importances[indices[i]]:.4f}")

def create_sample_data():
    """Create sample data if none exists"""
    os.makedirs('data', exist_ok=True)
    
    # Create synthetic cybersecurity data
    np.random.seed(42)
    n_samples = 1000
    
    data = {
        'duration': np.random.exponential(1.5, n_samples),
        'protocol': np.random.choice([0, 1, 2], n_samples),
        'service': np.random.choice([0, 1, 2, 3], n_samples),
        'flag': np.random.choice([0, 1, 2], n_samples),
        'src_bytes': np.random.lognormal(10, 2, n_samples),
        'dst_bytes': np.random.lognormal(9, 2, n_samples),
        'count': np.random.poisson(5, n_samples),
        'srv_count': np.random.poisson(3, n_samples),
    }
    
    df = pd.DataFrame(data)
    
    # Add anomalies (5% of data)
    n_anomalies = int(0.05 * n_samples)
    anomaly_indices = np.random.choice(n_samples, n_anomalies, replace=False)
    
    # Make anomalies suspicious
    for idx in anomaly_indices:
        df.loc[idx, 'src_bytes'] *= 100  # Large upload
        df.loc[idx, 'duration'] = 0.01   # Very short
        df.loc[idx, 'count'] *= 10       # Many connections
    
    df['label'] = 0
    df.loc[anomaly_indices, 'label'] = 1
    
    df.to_csv('data/sample_logs.csv', index=False)
    print(f"✓ Created sample data: {len(df)} records ({n_anomalies} threats)")
    
    return df

def main():
    """Main training pipeline"""
    print("🚀 Starting Cyber Threat Detection Training...")
    print("=" * 60)
    
    # Create directories
    create_directories()
    
    # Check if data exists
    if not os.path.exists('data/sample_logs.csv'):
        print("No data found. Creating sample data...")
        df = create_sample_data()
    else:
        df = pd.read_csv('data/sample_logs.csv')
        print(f"✓ Loaded data: {len(df)} records")
    
    print(f"\n📊 Dataset Info:")
    print(f"   Shape: {df.shape}")
    print(f"   Threat rate: {df['label'].mean():.2%}")
    
    # Prepare data
    X = df.drop('label', axis=1)
    y = df['label']
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    
    print(f"\n🔧 Data Split:")
    print(f"   Training: {X_train.shape[0]} samples")
    print(f"   Testing:  {X_test.shape[0]} samples")
    print(f"   Train threat rate: {y_train.mean():.2%}")
    print(f"   Test threat rate:  {y_test.mean():.2%}")
    
    # Train models
    trainer = ThreatDetectionTrainer()
    trainer.train_models(X_train, y_train)
    
    # Evaluate models
    results = trainer.evaluate_models(X_test, y_test)
    
    # Feature importance
    if 'random_forest' in trainer.trained_models:
        trainer.feature_importance_analysis('random_forest', X.columns.tolist())
    
    print("\n" + "=" * 60)
    print("✅ Training complete!")
    print(f"   Models saved to: models/")
    print(f"   Reports saved to: reports/")
    print("=" * 60)
    
    return results

if __name__ == "__main__":
    results = main()