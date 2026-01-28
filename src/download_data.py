"""
Download and prepare real cybersecurity datasets
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import requests
import zipfile
import os

def download_unsw_nb15_sample():
    """Download sample UNSW-NB15 dataset"""
    print("📥 Downloading UNSW-NB15 sample dataset...")
    
    # Sample dataset URL (small version for testing)
    url = "https://raw.githubusercontent.com/UNSW-CE-CS-IDS-2018/UNSW-NB15/master/data/UNSW_NB15_training-set.csv"
    
    try:
        df = pd.read_csv(url)
        print(f"✓ Downloaded {len(df)} records")
        
        # Rename columns for consistency
        if 'label' in df.columns:
            df['label'] = df['label'].astype(int)
        
        # Save sample
        df_sample = df.sample(min(10000, len(df)), random_state=42)
        df_sample.to_csv('data/unsw_nb15_sample.csv', index=False)
        print(f"✓ Saved sample with {len(df_sample)} records")
        
        return df_sample
    except Exception as e:
        print(f"⚠️  Error downloading: {e}")
        print("Creating synthetic data instead...")
        return create_synthetic_data()

def create_synthetic_data():
    """Create realistic synthetic cybersecurity data"""
    np.random.seed(42)
    n_samples = 5000
    
    data = {
        'duration': np.random.exponential(1, n_samples),
        'protocol_type': np.random.choice([0, 1, 2, 3], n_samples),
        'service': np.random.choice([0, 1, 2, 3, 4], n_samples),
        'flag': np.random.choice([0, 1, 2, 3], n_samples),
        'src_bytes': np.random.lognormal(10, 2, n_samples),
        'dst_bytes': np.random.lognormal(9, 2, n_samples),
        'count': np.random.poisson(5, n_samples),
        'srv_count': np.random.poisson(3, n_samples),
        'dst_host_count': np.random.poisson(10, n_samples),
        'dst_host_srv_count': np.random.poisson(5, n_samples),
    }
    
    df = pd.DataFrame(data)
    
    # Create realistic anomalies (5% of data)
    n_anomalies = int(0.05 * n_samples)
    anomaly_indices = np.random.choice(n_samples, n_anomalies, replace=False)
    
    # Modify anomalies to be suspicious
    for idx in anomaly_indices:
        df.loc[idx, 'src_bytes'] *= 100  # Large outbound traffic
        df.loc[idx, 'duration'] = 0.01  # Very short connections
        df.loc[idx, 'count'] *= 10  # Many connections
    
    # Create labels (1 for anomaly, 0 for normal)
    df['label'] = 0
    df.loc[anomaly_indices, 'label'] = 1
    
    # Save
    df.to_csv('data/synthetic_security_logs.csv', index=False)
    print(f"✓ Created synthetic dataset with {n_samples} records ({n_anomalies} anomalies)")
    
    return df

def prepare_dataset(dataset_name='synthetic'):
    """Main function to prepare dataset"""
    if dataset_name == 'unsw':
        return download_unsw_nb15_sample()
    else:
        return create_synthetic_data()

if __name__ == "__main__":
    os.makedirs('data', exist_ok=True)
    df = prepare_dataset()
    print(f"\nDataset shape: {df.shape}")
    print(f"Anomaly rate: {df['label'].mean():.2%}")