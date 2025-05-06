# Wine Quality Predictor

A simple Streamlit application that analyzes wine quality data and predicts whether a wine is "good" (quality ≥ 7) or "bad" (quality < 7) using a Random Forest classifier.

## Features

- Upload CSV files containing wine characteristics
- Basic data exploration (head, describe)
- Visualizations:
  - Quality distribution countplot
  - Citric acid vs quality barplot
  - Volatile acidity vs quality barplot
  - Correlation heatmap
- Binary classification (good/bad wine) prediction
- Accuracy score display

## Requirements

- Python 3.7+
- Required packages:
  - streamlit
  - pandas
  - scikit-learn
  - seaborn
  - matplotlib

## Installation

1. Clone this repository or download the Python file
2. Install the required packages:

```bash
pip install streamlit pandas scikit-learn seaborn matplotlib
```

## Usage

Run the application with:

```
streamlit run main.py
```

Then:

1. Upload a CSV file containing wine data with a 'quality' column
2. Explore the data visualizations
3. View the prediction accuracy score

## Expected Data Format

The application expects a CSV file with columns representing wine characteristics and a 'quality' column (numeric values).

Example structure:

```
fixed acidity,volatile acidity,citric acid,...,quality
7.4,0.7,0.0,...,5
7.8,0.88,0.0,...,5
...
```
## Limitations

- Basic visualization display (plots may overlap)
- Simple binary classification only
- No hyperparameter tuning
- Minimal error handling