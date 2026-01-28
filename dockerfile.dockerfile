FROM python:3.9-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy project files
COPY . .

# Create directories
RUN mkdir -p data models reports notebooks

# Download sample dataset
RUN python -c "
import pandas as pd
import numpy as np
np.random.seed(42)
data = pd.DataFrame({
    'duration': np.random.exponential(1, 1000),
    'src_bytes': np.random.randint(0, 1000000, 1000),
    'dst_bytes': np.random.randint(0, 1000000, 1000),
    'label': np.random.choice([0, 1], 1000, p=[0.95, 0.05])
})
data.to_csv('data/sample_logs.csv', index=False)
print('Sample data created')
"

# Expose port for dashboard
EXPOSE 5000

# Run application
CMD ["python", "src/detect_threats.py"]