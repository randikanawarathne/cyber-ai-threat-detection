# 🔐 AI-Driven Cyber Threat Detection System

An intelligent security analytics platform that uses machine learning to detect anomalous behavior and potential threats in network/system data.

## 🎯 Project Overview
This project implements a prototype Security Operations Center (SOC) tool that analyzes log data to identify suspicious activities using both supervised and unsupervised machine learning techniques.

## 🏗️ Architecture
Dataset → Preprocessing → Feature Engineering → ML Model → Threat Classification → Alert Generation


## 📊 Features
- **Log Analysis**: Processes network/system logs for anomaly detection
- **Multiple ML Models**: Implements Isolation Forest, Random Forest, and Logistic Regression
- **Real-time Simulation**: Can process streaming log data
- **Threat Scoring**: Assigns severity scores to detected threats
- **MITRE ATT&CK Mapping**: Maps detected anomalies to known techniques

## 🛠️ Technologies
- Python 3.8+
- Scikit-learn, Pandas, NumPy
- Matplotlib/Seaborn for visualization
- Optional: Flask/Dash for dashboard

## 📁 Project Structure
cyber-ai-threat-detection/
├── .gitignore
├── README.md
├── LICENSE
├── requirements.txt
├── main.py
├── setup.py
├── run.py
├── run.ps1
├── run.bat
├── src/
│   ├── __init__.py
│   ├── detect_threats.py
│   ├── train_model.py
│   └── preprocess.py
├── data/               (ignored in git)
│   └── sample_data.csv (optional sample)
├── models/             (ignored in git)
├── reports/            (ignored in git)
├── notebooks/
│   └── exploratory_analysis.ipynb
└── docs/
    └── API.md

## 🚀 Quick Start
1. Clone repository: `git clone https://github.com/randikanawarathne/cyber-ai-threat-detection.git`
2. Install dependencies: `pip install -r requirements.txt`
3. Run analysis: `python src/detect_threats.py`

## 📈 Results
Model performance metrics and detection examples are available in `reports/`

## 🔮 Future Enhancements
- Integration with SIEM tools (Splunk, Elastic)
- Real-time streaming with Apache Kafka
- Deep learning models (LSTM for sequential data)
- Cloud deployment (AWS/Azure security services)
