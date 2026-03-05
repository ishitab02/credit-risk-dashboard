#!/bin/bash

# Credit Risk Dashboard - Setup Script
# Run this script to set up and launch the application

set -e

echo "=========================================="
echo "Credit Risk Analytics Dashboard Setup"
echo "=========================================="

# Check for Python
if ! command -v python3 &> /dev/null; then
    echo "Error: Python 3 is required but not installed."
    exit 1
fi

# Create virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    echo ""
    echo "1. Creating virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
echo ""
echo "2. Activating virtual environment..."
source venv/bin/activate

# Install dependencies
echo ""
echo "3. Installing dependencies..."
pip install -r requirements.txt --quiet

# Set up database
echo ""
echo "4. Setting up database..."
if [ ! -f "database/credit_risk.db" ]; then
    python src/db_setup.py
else
    echo "   Database already exists, skipping..."
fi

# Train model
echo ""
echo "5. Training model..."
if [ ! -f "models/default_scorer.pkl" ]; then
    python models/train_model.py
else
    echo "   Model already exists, skipping..."
fi

# Launch application
echo ""
echo "=========================================="
echo "Setup complete! Launching dashboard..."
echo "=========================================="
echo ""
streamlit run app.py
