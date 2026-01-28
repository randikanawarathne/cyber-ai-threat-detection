#!/usr/bin/env python3
"""
Main entry point for Cyber Threat Detection
"""
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from detect_threats import main as threat_main

if __name__ == "__main__":
    threat_main()