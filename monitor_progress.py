#!/usr/bin/env python3
"""
Monitor progress of musical feature extraction
"""

import os
import time
from pathlib import Path

def count_files():
    """Count files in musical_features directory"""
    feature_dir = Path("feature_data/musical_features")
    if feature_dir.exists():
        files = list(feature_dir.glob("*.json"))
        return len(files)
    return 0

def main():
    print("Monitoring musical feature extraction progress...")
    print("Press Ctrl+C to stop monitoring")
    
    try:
        while True:
            count = count_files()
            print(f"[{time.strftime('%H:%M:%S')}] Files processed: {count}/31")
            time.sleep(10)  # Check every 10 seconds
    except KeyboardInterrupt:
        print("\nMonitoring stopped.")

if __name__ == "__main__":
    main() 