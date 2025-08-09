"""
Diabetes Prediction Model
========================
Author: TNT
Version: 1.0
Dataset: Pima Indian Diabetes Dataset
Description: Machine learning model to predict diabetes risk using various health indicators
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, roc_auc_score, roc_curve
import joblib
import warnings
import os
from datetime import datetime

warnings.filterwarnings('ignore')

class DiabetesPredictionModel:
    """
    A comprehensive diabetes prediction model using multiple algorithms
    """
    
    def __init__(self, data_path):
        self.data_path = data_path
        self.data = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.scaler = StandardScaler()
        self.models = {}
        self.results = {}
        
    def load_data(self):
        """Load and display basic information about the dataset"""
        print("Loading Pima Indian Diabetes Dataset...")
        self.data = pd.read_csv(self.data_path)
        
        print(f"Dataset shape: {self.data.shape}")
        print(f"Dataset columns: {list(self.data.columns)}")
        print("\nFirst 5 rows:")
        print(self.data.head())
        
        print("\nDataset Info:")
        print(self.data.info())
        
        print("\nBasic Statistics:")
        print(self.data.describe())
        
        return self.data
    
    def exploratory_data_analysis(self):
        """Perform comprehensive EDA"""
        print("\n" + "="*50)
        print("EXPLORATORY DATA ANALYSIS")
        print("="*50)
        
        # Check for missing values
        print("\nMissing Values:")
        print(self.data.isnull().sum())
        
        # Check for zero values (which might be missing values in disguise)
        print("\nZero Values Analysis:")
        zero_cols = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
        for col in zero_cols:
            if col in self.data.columns:
                zero_count = (self.data[col] == 0).sum()
                print(f"{col}: {zero_count} zero values ({zero_count/len(self.data)*100:.2f}%)")
        
        # Target variable distribution
        print(f"\nTarget Variable Distribution:")
        print(self.data['Class'].value_counts())
        print(f"Diabetes prevalence: {self.data['Class'].mean()*100:.2f}%")
        
        # Create visualizations
        self.create_visualizations()
        
    def create_visualizations(self):
        """Create and save visualizations"""
        print("\nCreating visualizations...")
        
        # Create output directory if it doesn't exist
        os.makedirs('output/plots', exist_ok=True)
        
        # 1. Target distribution
        plt.figure(figsize=(8, 6))
        self.data['Class'].value_counts().plot(kind='bar')
        plt.title('Distribution of Diabetes Cases')
        plt.xlabel('Class (0: No Diabetes, 1: Diabetes)')
        plt.ylabel('Count')
        plt.xticks(rotation=0)
        plt.tight_layout()
        plt.savefig('output/plots/target_distribution.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # 2. Correlation heatmap
        plt.figure(figsize=(10, 8))
        correlation_matrix = self.data.corr()
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0)
        plt.title('Feature Correlation Heatmap')
        plt.tight_layout()
        plt.savefig('output/plots/correlation_heatmap.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # 3. Feature distributions by class
        fig, axes = plt.subplots(3, 3, figsize=(15, 12))
        features = self.data.columns[:-1]  # All except 'Class'
        
        for i, feature in enumerate(features):
            row = i // 3
            col = i % 3
            
            # Box plot for each feature by class
            self.data.boxplot(column=feature, by='Class', ax=axes[row, col])
            axes[row, col].set_title(f'{feature} by Diabetes Status')
            axes[row, col].set_xlabel('Class')
            
        plt.tight_layout()
        plt.savefig('output/plots/feature_distributions.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print("Visualizations saved to output/plots/")
    
    def preprocess_data(self):
        """Preprocess the data for modeling"""
        print("\n" + "="*50)
        print("DATA PREPROCESSING")
        print("="*50)
        
        # Handle zero values by replacing with median (for specific columns)
        zero_cols = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
        
        for col in zero_cols:
            if col in self.data.columns:
                # Replace zeros with NaN, then fill with median
                self.data[col] = self.data[col].replace(0, np.nan)
                median_val = self.data[col].median()
                self.data[col] = self.data[col].fillna(median_val)
                print(f"Replaced zeros in {col} with median value: {median_val:.2f}")
        
        # Separate features and target
        X = self.data.drop('Class', axis=1)
        y = self.data['Class']
        
        # Split the data
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Scale the features
        self.X_train_scaled = self.scaler.fit_transform(self.X_train)
        self.X_test_scaled = self.scaler.transform(self.X_test)
        
        print(f"Training set size: {self.X_train.shape}")
        print(f"Test set size: {self.X_test.shape}")
        print(f"Training set diabetes prevalence: {self.y_train.mean()*100:.2f}%")
        print(f"Test set diabetes prevalence: {self.y_test.mean()*100:.2f}%")
        
    def train_models(self):
        """Train multiple machine learning models"""
        print("\n" + "="*50)
        print("MODEL TRAINING")
        print("="*50)
        
        # Define models
        models_config = {
            'Random Forest': {
                'model': RandomForestClassifier(random_state=42),
                'params': {
                    'n_estimators': [100, 200],
                    'max_depth': [10, 20, None],
                    'min_samples_split': [2, 5],
                    'min_samples_leaf': [1, 2]
                },
                'use_scaled': False
            },
            'Logistic Regression': {
                'model': LogisticRegression(random_state=42, max_iter=1000),
                'params': {
                    'C': [0.1, 1, 10],
                    'penalty': ['l1', 'l2'],
                    'solver': ['liblinear']
                },
                'use_scaled': True
            },
            'Support Vector Machine': {
                'model': SVC(random_state=42, probability=True),
                'params': {
                    'C': [0.1, 1, 10],
                    'kernel': ['rbf', 'linear'],
                    'gamma': ['scale', 'auto']
                },
                'use_scaled': True
            }
        }
        
        # Train each model
        for name, config in models_config.items():
            print(f"\nTraining {name}...")
            
            # Choose scaled or unscaled data
            X_train_data = self.X_train_scaled if config['use_scaled'] else self.X_train
            X_test_data = self.X_test_scaled if config['use_scaled'] else self.X_test
            
            # Grid search for best parameters
            grid_search = GridSearchCV(
                config['model'], 
                config['params'], 
                cv=5, 
                scoring='roc_auc',
                n_jobs=-1
            )
            
            grid_search.fit(X_train_data, self.y_train)
            
            # Store the best model
            self.models[name] = grid_search.best_estimator_
            
            # Make predictions
            y_pred = grid_search.predict(X_test_data)
            y_pred_proba = grid_search.predict_proba(X_test_data)[:, 1]
            
            # Calculate metrics
            accuracy = accuracy_score(self.y_test, y_pred)
            roc_auc = roc_auc_score(self.y_test, y_pred_proba)
            
            # Store results
            self.results[name] = {
                'best_params': grid_search.best_params_,
                'accuracy': accuracy,
                'roc_auc': roc_auc,
                'y_pred': y_pred,
                'y_pred_proba': y_pred_proba,
                'classification_report': classification_report(self.y_test, y_pred),
                'confusion_matrix': confusion_matrix(self.y_test, y_pred)
            }
            
            print(f"Best parameters: {grid_search.best_params_}")
            print(f"Accuracy: {accuracy:.4f}")
            print(f"ROC AUC: {roc_auc:.4f}")
    
    def evaluate_models(self):
        """Evaluate and compare all models"""
        print("\n" + "="*50)
        print("MODEL EVALUATION")
        print("="*50)
        
        # Create comparison dataframe
        comparison_data = []
        for name, results in self.results.items():
            comparison_data.append({
                'Model': name,
                'Accuracy': results['accuracy'],
                'ROC AUC': results['roc_auc']
            })
        
        comparison_df = pd.DataFrame(comparison_data)
        comparison_df = comparison_df.sort_values('ROC AUC', ascending=False)
        
        print("Model Comparison:")
        print(comparison_df.to_string(index=False))
        
        # Find best model
        best_model_name = comparison_df.iloc[0]['Model']
        print(f"\nBest Model: {best_model_name}")
        
        # Detailed evaluation of best model
        print(f"\nDetailed Evaluation of {best_model_name}:")
        print("Classification Report:")
        print(self.results[best_model_name]['classification_report'])
        
        print("\nConfusion Matrix:")
        print(self.results[best_model_name]['confusion_matrix'])
        
        # Create evaluation plots
        self.create_evaluation_plots()
        
        return best_model_name
    
    def create_evaluation_plots(self):
        """Create evaluation plots"""
        print("\nCreating evaluation plots...")
        
        # ROC Curves
        plt.figure(figsize=(10, 8))
        
        for name, results in self.results.items():
            fpr, tpr, _ = roc_curve(self.y_test, results['y_pred_proba'])
            plt.plot(fpr, tpr, label=f"{name} (AUC = {results['roc_auc']:.3f})")
        
        plt.plot([0, 1], [0, 1], 'k--', label='Random')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curves Comparison')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig('output/plots/roc_curves.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # Model comparison bar plot
        comparison_data = []
        for name, results in self.results.items():
            comparison_data.append({
                'Model': name,
                'Accuracy': results['accuracy'],
                'ROC AUC': results['roc_auc']
            })
        
        comparison_df = pd.DataFrame(comparison_data)
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Accuracy comparison
        comparison_df.plot(x='Model', y='Accuracy', kind='bar', ax=ax1, color='skyblue')
        ax1.set_title('Model Accuracy Comparison')
        ax1.set_ylabel('Accuracy')
        ax1.tick_params(axis='x', rotation=45)
        
        # ROC AUC comparison
        comparison_df.plot(x='Model', y='ROC AUC', kind='bar', ax=ax2, color='lightcoral')
        ax2.set_title('Model ROC AUC Comparison')
        ax2.set_ylabel('ROC AUC')
        ax2.tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.savefig('output/plots/model_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print("Evaluation plots saved to output/plots/")
    
    def save_models_and_results(self, best_model_name):
        """Save trained models and results"""
        print("\n" + "="*50)
        print("SAVING MODELS AND RESULTS")
        print("="*50)
        
        # Create output directories
        os.makedirs('output/models', exist_ok=True)
        os.makedirs('results', exist_ok=True)
        
        # Save all models
        for name, model in self.models.items():
            model_filename = f"output/models/{name.lower().replace(' ', '_')}_model.pkl"
            joblib.dump(model, model_filename)
            print(f"Saved {name} model to {model_filename}")
        
        # Save scaler
        scaler_filename = "output/models/scaler.pkl"
        joblib.dump(self.scaler, scaler_filename)
        print(f"Saved scaler to {scaler_filename}")
        
        # Save results summary
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Create comprehensive results dictionary
        final_results = {
            'timestamp': timestamp,
            'dataset_info': {
                'shape': self.data.shape,
                'features': list(self.data.columns[:-1]),
                'target': 'Class',
                'diabetes_prevalence': f"{self.data['Class'].mean()*100:.2f}%"
            },
            'preprocessing': {
                'train_size': self.X_train.shape[0],
                'test_size': self.X_test.shape[0],
                'features_scaled': True
            },
            'model_results': {}
        }
        
        # Add model results
        for name, results in self.results.items():
            final_results['model_results'][name] = {
                'best_params': results['best_params'],
                'accuracy': float(results['accuracy']),
                'roc_auc': float(results['roc_auc']),
                'confusion_matrix': results['confusion_matrix'].tolist()
            }
        
        final_results['best_model'] = best_model_name
        
        # Save as JSON
        import json
        results_filename = f"results/diabetes_prediction_results_{timestamp}.json"
        with open(results_filename, 'w') as f:
            json.dump(final_results, f, indent=2)
        print(f"Saved results summary to {results_filename}")
        
        # Save detailed classification reports
        reports_filename = f"results/classification_reports_{timestamp}.txt"
        with open(reports_filename, 'w') as f:
            f.write("DIABETES PREDICTION MODEL - CLASSIFICATION REPORTS\n")
            f.write("="*60 + "\n\n")
            
            for name, results in self.results.items():
                f.write(f"{name} Model:\n")
                f.write("-" * 30 + "\n")
                f.write(f"Best Parameters: {results['best_params']}\n")
                f.write(f"Accuracy: {results['accuracy']:.4f}\n")
                f.write(f"ROC AUC: {results['roc_auc']:.4f}\n\n")
                f.write("Classification Report:\n")
                f.write(results['classification_report'])
                f.write("\n\nConfusion Matrix:\n")
                f.write(str(results['confusion_matrix']))
                f.write("\n\n" + "="*60 + "\n\n")
        
        print(f"Saved detailed reports to {reports_filename}")
        
        return final_results
    
    def run_complete_pipeline(self):
        """Run the complete machine learning pipeline"""
        print("DIABETES PREDICTION MODEL PIPELINE")
        print("="*60)
        print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        # Step 1: Load data
        self.load_data()
        
        # Step 2: EDA
        self.exploratory_data_analysis()
        
        # Step 3: Preprocessing
        self.preprocess_data()
        
        # Step 4: Train models
        self.train_models()
        
        # Step 5: Evaluate models
        best_model_name = self.evaluate_models()
        
        # Step 6: Save everything
        final_results = self.save_models_and_results(best_model_name)
        
        print("\n" + "="*60)
        print("PIPELINE COMPLETED SUCCESSFULLY!")
        print(f"Best Model: {best_model_name}")
        print(f"Best Model Accuracy: {final_results['model_results'][best_model_name]['accuracy']:.4f}")
        print(f"Best Model ROC AUC: {final_results['model_results'][best_model_name]['roc_auc']:.4f}")
        print(f"Completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("="*60)
        
        return final_results

def main():
    """Main function to run the diabetes prediction pipeline"""
    # Initialize the model
    model = DiabetesPredictionModel('data/pima-diabetes.csv')
    
    # Run the complete pipeline
    results = model.run_complete_pipeline()
    
    return results

if __name__ == "__main__":
    results = main()
