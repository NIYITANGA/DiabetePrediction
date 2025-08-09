# Diabetes Prediction Model

**Author:** TNT  
**Version:** 1.0  
**Dataset:** Pima Indian Diabetes Dataset  

## Overview

This project implements a comprehensive machine learning pipeline to predict diabetes risk using the Pima Indian Diabetes Dataset collected by the National Institute of Diabetes and Digestive and Kidney Diseases. The project includes multiple machine learning models, data visualization, and prediction utilities.

## Dataset Description

The dataset contains 768 records with 8 features and 1 target variable:

### Features:
- **Pregnancies**: Number of times pregnant
- **Glucose**: Plasma glucose concentration (mg/dL)
- **BloodPressure**: Diastolic blood pressure (mm Hg)
- **SkinThickness**: Triceps skin fold thickness (mm)
- **Insulin**: 2-Hour serum insulin (μU/mL)
- **BMI**: Body mass index (weight in kg/(height in m)^2)
- **Pedigree**: Diabetes pedigree function
- **Age**: Age in years

### Target:
- **Class**: 0 (No Diabetes) or 1 (Diabetes)

## Project Structure

```
DiabetePrediction/
├── data/
│   └── pima-diabetes.csv          # Dataset
├── scripts/
│   ├── 1.0-tnt-diabetes-prediction.py    # Main training script
│   └── 1.1-tnt-prediction-utility.py     # Prediction utility
├── notebooks/
│   └── 1.2-tnt-diabetes-analysis.ipynb   # Interactive analysis
├── output/
│   ├── models/                    # Trained models
│   └── plots/                     # Visualizations
├── results/                       # Model results and reports
└── README.md                      # This file
```

## Models Implemented

1. **Random Forest Classifier**
   - Ensemble method with multiple decision trees
   - Good for handling non-linear relationships
   - Provides feature importance

2. **Logistic Regression**
   - Linear model for binary classification
   - Interpretable coefficients
   - Fast training and prediction

3. **Support Vector Machine (SVM)**
   - Effective for high-dimensional data
   - Uses kernel trick for non-linear patterns
   - Good generalization capability

## Installation and Setup

1. **Clone the repository:**
   ```bash
   git clone https://github.com/NIYITANGA/DiabetePrediction.git
   cd DiabetePrediction
   ```

2. **Install required packages:**
   ```bash
   pip install -r requirements.txt
   ```

## Usage

### 1. Train Models

Run the main training script to build and evaluate all models:

```bash
python scripts/1.0-tnt-diabetes-prediction.py
```

This will:
- Load and preprocess the data
- Perform exploratory data analysis
- Train multiple models with hyperparameter tuning
- Evaluate model performance
- Save trained models and results
- Generate visualizations

### 2. Make Predictions

Use the prediction utility for new predictions:

```bash
python scripts/1.1-tnt-prediction-utility.py
```

Options available:
- Interactive single prediction
- Batch prediction from CSV file
- Example prediction demonstration

### 3. Interactive Analysis

Open the Jupyter notebook for detailed analysis:

```bash
jupyter notebook notebooks/1.2-tnt-diabetes-analysis.ipynb
```

## Model Performance

After training, the models are evaluated using:
- **Accuracy**: Overall correct predictions
- **ROC AUC**: Area under the ROC curve
- **Precision, Recall, F1-Score**: Detailed classification metrics
- **Confusion Matrix**: True/False positives and negatives

## Key Features

### Data Preprocessing
- Handles zero values (likely missing data)
- Feature scaling for algorithms that require it
- Train/test split with stratification

### Model Training
- Grid search for hyperparameter optimization
- Cross-validation for robust evaluation
- Multiple algorithms for comparison

### Visualization
- Feature distributions and correlations
- Model performance comparisons
- ROC curves and confusion matrices
- Feature importance analysis

### Prediction Utilities
- Single patient prediction
- Batch prediction capabilities
- Risk level assessment
- Consensus predictions from multiple models

## Results and Outputs

The pipeline generates:

1. **Trained Models** (`output/models/`):
   - `random_forest_model.pkl`
   - `logistic_regression_model.pkl`
   - `support_vector_machine_model.pkl`
   - `scaler.pkl`

2. **Visualizations** (`output/plots/`):
   - Target distribution
   - Feature correlations
   - Model comparisons
   - ROC curves

3. **Results** (`results/`):
   - JSON summary with metrics
   - Detailed classification reports
   - Timestamp-based file naming

## Risk Factors Analysis

Based on the analysis, key diabetes risk factors include:
- **Glucose levels**: Strongest predictor
- **BMI**: Important indicator of metabolic health
- **Age**: Risk increases with age
- **Pregnancies**: Relevant for women
- **Family history**: Genetic predisposition

## Recommendations

1. **Clinical Use**:
   - Use as a screening tool, not diagnostic
   - Combine with clinical judgment
   - Regular model updates with new data

2. **Prevention**:
   - Focus on glucose monitoring
   - Maintain healthy BMI
   - Age-specific screening protocols
   - Consider family history

## Technical Details

### Dependencies
- pandas: Data manipulation
- numpy: Numerical computations
- scikit-learn: Machine learning algorithms
- matplotlib/seaborn: Visualizations
- joblib: Model serialization

### Model Selection
- Grid search with 5-fold cross-validation
- ROC AUC as primary metric
- Stratified sampling to maintain class balance

### Data Quality
- 768 samples, 34.9% diabetes prevalence
- Some features have zero values (treated as missing)
- No explicit missing values in dataset

## Future Enhancements

1. **Model Improvements**:
   - Deep learning models
   - Ensemble methods
   - Feature engineering

2. **Deployment**:
   - Web application interface
   - REST API for predictions
   - Mobile app integration

3. **Data**:
   - Larger, more diverse datasets
   - Real-time data integration
   - Longitudinal studies

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## License

This project is for educational and research purposes. Please ensure compliance with data usage policies and medical regulations when adapting for clinical use.

## Contact

For questions or suggestions, please open an issue in the repository.

---

**Note**: This model is for educational purposes and should not be used as a substitute for professional medical advice, diagnosis, or treatment.
