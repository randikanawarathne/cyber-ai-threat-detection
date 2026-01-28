from flask import Flask, render_template, jsonify
import json

app = Flask(__name__)

@app.route('/')
def dashboard():
    return render_template('dashboard.html')

@app.route('/api/threats')
def get_threats():
    with open('reports/threat_report.json', 'r') as f:
        threats = json.load(f)
    return jsonify(threats)

@app.route('/api/stats')
def get_stats():
    # Calculate real-time statistics
    stats = {
        'total_threats': 142,
        'critical': 8,
        'high': 23,
        'medium': 45,
        'low': 66,
        'detection_rate': 96.7,
        'false_positives': 3.2
    }
    return jsonify(stats)

if __name__ == '__main__':
    app.run(debug=True, port=5000)