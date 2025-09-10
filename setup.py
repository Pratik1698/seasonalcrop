#!/usr/bin/env python3
# setup.py - Quick setup script for the Crop Cycle Planner

import subprocess
import sys
import os

def install_requirements():
    """Install required packages"""
    try:
        print("Installing required packages...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        print("✅ All packages installed successfully!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Error installing packages: {e}")
        return False

def check_files():
    """Check if all required files are present"""
    required_files = [
        'app.py', 'requirements.txt', 'crop_recommendation_model.pkl',
        'yield_prediction_model.pkl', 'crop_encoder.pkl', 'crop_prices.json'
    ]

    missing_files = []
    for file in required_files:
        if not os.path.exists(file):
            missing_files.append(file)

    if missing_files:
        print(f"❌ Missing files: {missing_files}")
        return False
    else:
        print("✅ All required files are present!")
        return True

def run_app():
    """Launch the Streamlit app"""
    try:
        print("🚀 Launching the Crop Cycle Planner...")
        subprocess.run([sys.executable, "-m", "streamlit", "run", "app.py"])
    except KeyboardInterrupt:
        print("\n👋 Application stopped by user")
    except Exception as e:
        print(f"❌ Error running the app: {e}")

if __name__ == "__main__":
    print("🌾 AI-Based Crop Cycle Planner Setup")
    print("=" * 40)

    # Check files
    if not check_files():
        sys.exit(1)

    # Install requirements
    if not install_requirements():
        sys.exit(1)

    # Ask user if they want to run the app
    choice = input("\n🚀 Do you want to run the app now? (y/n): ").lower()
    if choice in ['y', 'yes']:
        run_app()
    else:
        print("\n📋 To run the app later, use: streamlit run app.py")
        print("🎯 Open http://localhost:8501 in your browser")
