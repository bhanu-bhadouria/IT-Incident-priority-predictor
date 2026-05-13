# IT Incident Priority Predictor

An AI-powered system for predicting and prioritizing IT incidents using machine learning techniques.

## Overview

This project focuses on analyzing IT incident data and building predictive models to automatically assign priority levels to incoming incidents. By leveraging exploratory data analysis (EDA) and feature engineering, the system can help organizations optimize incident response workflows.

## Project Structure

- **EDA.py** - Exploratory Data Analysis script for understanding incident data patterns
- **feature_engineering.py** - Feature engineering pipeline for model preparation
- **incidents_day1.csv** - Sample incident dataset
- **Visualizations**:
  - `category_vs_priority.png` - Distribution analysis across categories
  - `correlation_matrix.png` - Feature correlation heatmap
  - `numerical_distributions.png` - Distribution of numerical features
  - `target_distribution.png` - Priority distribution analysis
  - `time_patterns.png` - Temporal patterns in incidents

## Getting Started

### Prerequisites

- Python 3.7+
- pandas
- scikit-learn
- matplotlib
- seaborn

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

## Agentic AI Integration

This project leverages **Agentic AI** (autonomous agents) to streamline development and analysis workflows:

### Agent-Powered Components

#### 1. **Exploratory Data Analysis Agent**
- Automated statistical analysis of incident datasets
- Pattern detection and anomaly identification
- Generation of distribution analysis and correlation matrices
- Autonomous feature importance discovery

#### 2. **Feature Engineering Agent**
- Automated feature extraction from raw incident data
- Intelligent data transformation and preprocessing
- Feature scaling and normalization
- Dimensionality optimization suggestions

#### 3. **Data Visualization Agent**
- Autonomous generation of insightful visualizations
- Category vs. priority distribution analysis
- Temporal pattern analysis
- Correlation matrix generation
- Automated chart creation and optimization

#### 4. **Model Development Agent**
- Autonomous model selection and training
- Hyperparameter optimization
- Cross-validation and performance evaluation
- Model comparison and recommendation

#### 5. **Documentation & Git Management Agent**
- Automated README generation and updates
- Version control management
- Commit tracking and change documentation
- Repository organization

### Benefits of Agentic Approach

- **Efficiency**: Agents handle repetitive data processing tasks automatically
- **Consistency**: Standardized analysis and documentation workflows
- **Scalability**: Easy to extend agents for new incident sources
- **Accuracy**: Reduced human error in data processing and analysis
- **Speed**: Parallel processing of multiple analysis tasks

### How Agents Were Used

1. **Data Pipeline**: Agents automatically processed raw incident data
2. **Analysis**: Autonomous statistical and predictive analysis
3. **Visualization**: Agents created comprehensive visualizations without manual intervention
4. **Documentation**: Auto-generated project documentation and change logs
5. **Git Workflow**: Automated version control and commit management

This project demonstrates how **autonomous AI agents** can significantly accelerate ML project development while maintaining quality and consistency.

## Contact

For questions or contributions, please reach out to the development team.
