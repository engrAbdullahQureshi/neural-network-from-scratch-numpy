# Deep Learning Assignment 1 – MLP Analysis on Airbnb Dataset

## Project Overview
This project implements a multi-layer perceptron (MLP) for classifying Airbnb listings into `price_class`. It includes:

- **Part A** – Data preprocessing and exploratory data analysis (EDA)
- **Part B** – Training MLP using ReLU and Sigmoid activation functions
- **Part C** – Gradient-based feature attribution (analytical and implementation)
- **Part D** – Test evaluation and generalization analysis



## Project Structure

DL_Assignment1/
│
├── data/
│ ├── train.csv # Training dataset
│ ├── test.csv # Test dataset
│ ├── df_encoded.csv # Preprocessed & encoded dataset
│ └── scaler.pkl # Saved StandardScaler from Part A
│
├── notebooks/
│ ├── part_a.ipynb # EDA & preprocessing
│ ├── partb.ipynb # Part B, C, D implementation
│
├── venv/ # Python virtual environment
│
└── README.md # This file

# SET UP INSTRUCTION
git clone https://github.com/engrAbdullahQureshi/DL_Assignment1.git
cd DL_Assignment1

# Activate virtual environment

.\venv\Scripts\activate

(if you dont have venv install this please)
python -m venv venv
pip install --upgrade pip
pip install -r requirements.txt


# PDF REPORT OF MY ASSIGNMENT ALSO INCLUDED HERE NAMED DL_Assignment1