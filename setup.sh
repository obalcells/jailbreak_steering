#!/bin/bash

# HuggingFace configuration
echo "Please enter your Hugging Face token (press Enter to skip):"
read -r token

# Check if the token variable is not empty
if [ -n "$token" ]; then
    echo "Storing HF_TOKEN in .env file..."
    echo "HF_TOKEN=$token" > .env
    
    echo "Installing Hugging Face CLI..."
    yes | pip install --upgrade huggingface_hub
    echo "Logging in to Hugging Face CLI..."
    huggingface-cli login --token $token
else
    echo "No token entered. Skipping..."
fi

# venv configuration
echo "Creating venv..."
python -m venv venv

echo "Activating venv..."
source venv/bin/activate

echo "Installing requirements..."

yes | pip install -r requirements.txt

echo "All set up!"