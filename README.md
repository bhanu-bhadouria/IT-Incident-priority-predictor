# IT Incident Priority Predictor

An AI-powered system for predicting and prioritizing IT incidents using machine learning techniques.

## Overview

This project focuses on analyzing IT incident data and building predictive models to automatically assign priority levels to incoming incidents. By leveraging exploratory data analysis (EDA) and feature engineering, the system can help organizations optimize incident response workflows.

## Project Structure

- **EDA.py** - Exploratory Data Analysis script for understanding incident data patterns
- **feature_engineering.py** - Feature engineering pipeline for model preparation
- **model_training.py** - Model training, evaluation, and comparison
- **incidents_day1.csv** - Sample incident dataset
- **Visualizations**:
  - `category_vs_priority.png` - Distribution analysis across categories
  - `correlation_matrix.png` - Feature correlation heatmap
  - `numerical_distributions.png` - Distribution of numerical features
  - `target_distribution.png` - Priority distribution analysis
  - `time_patterns.png` - Temporal patterns in incidents
  - `confusion_matrix_dt.png` - Decision Tree confusion matrix
  - `confusion_matrix_rf.png` - Random Forest confusion matrix
  - `feature_importance_rf.png` - Random Forest feature importances
  - `roc_curve_comparison.png` - ROC curve comparison

## Getting Started

### Prerequisites

- Python 3.7+
- pandas
- scikit-learn
- matplotlib
- seaborn
- joblib

### Installation

```bash
pip install -r requirements.txt
```

### Usage

1. Run exploratory data analysis:
```bash
python EDA.py
```

2. Execute feature engineering:
```bash
python feature_engineering.py
```

3. Train and evaluate models:
```bash
python model_training.py
```

## Data

The primary dataset (`incidents_day1.csv`) contains incident records with various attributes including category, priority levels, and temporal information.

## Analysis & Insights

The project includes comprehensive visualizations and analysis of:
- Incident categories and their priority distributions
- Correlation patterns between features
- Numerical feature distributions
- Temporal patterns in incident occurrence

## License

This project is part of the IT incident management system.

## Contact

For questions or contributions, please reach out to the development team.
