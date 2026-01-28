from setuptools import setup, find_packages

setup(
    name="cyber-ai-threat-detection",
    version="1.0.0",
    packages=find_packages(),
    install_requires=[
        'pandas>=2.0.3',
        'numpy>=1.24.3',
        'scikit-learn>=1.3.0',
        'matplotlib>=3.7.2',
        'joblib>=1.3.1',
    ],
    entry_points={
        'console_scripts': [
            'threat-detector=src.detect_threats:main',
            'threat-train=src.train_model:main',
        ],
    },
)