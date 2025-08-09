"""
Diabetes Prediction Utility
===========================
Author: TNT
Version: 1.1
Description: Utility script to make predictions using trained diabetes prediction models
"""

import pandas as pd
import numpy as np
import joblib
import json
import os
from datetime import datetime

class DiabetesPredictionUtility:
    """
    Utility class for making diabetes predictions using trained models
    """
    
    def __init__(self, models_dir='output/models'):
        self.models_dir = models_dir
        self.models = {}
        self.scaler = None
        self.feature_names = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 
                             'Insulin', 'BMI', 'Pedigree', 'Age']
        
    def load_models(self):
        """Load all trained models and scaler"""
        print("Loading trained models...")
        
        # Load scaler
        scaler_path = os.path.join(self.models_dir, 'scaler.pkl')
        if os.path.exists(scaler_path):
            self.scaler = joblib.load(scaler_path)
            print("✓ Scaler loaded successfully")
        else:
            raise FileNotFoundError(f"Scaler not found at {scaler_path}")
        
        # Load models
        model_files = {
            'Random Forest': 'random_forest_model.pkl',
            'Logistic Regression': 'logistic_regression_model.pkl',
            'Support Vector Machine': 'support_vector_machine_model.pkl'
        }
        
        for model_name, filename in model_files.items():
            model_path = os.path.join(self.models_dir, filename)
            if os.path.exists(model_path):
                self.models[model_name] = joblib.load(model_path)
                print(f"✓ {model_name} model loaded successfully")
            else:
                print(f"⚠ {model_name} model not found at {model_path}")
        
        if not self.models:
            raise FileNotFoundError("No models found in the models directory")
        
        print(f"Loaded {len(self.models)} models successfully")
    
    def validate_input(self, data):
        """Validate input data"""
        if isinstance(data, dict):
            # Single prediction
            missing_features = set(self.feature_names) - set(data.keys())
            if missing_features:
                raise ValueError(f"Missing features: {missing_features}")
            
            # Check for valid ranges
            validations = {
                'Pregnancies': (0, 20),
                'Glucose': (0, 300),
                'BloodPressure': (0, 200),
                'SkinThickness': (0, 100),
                'Insulin': (0, 1000),
                'BMI': (0, 100),
                'Pedigree': (0, 3),
                'Age': (0, 120)
            }
            
            for feature, (min_val, max_val) in validations.items():
                if not (min_val <= data[feature] <= max_val):
                    print(f"⚠ Warning: {feature} value {data[feature]} is outside typical range ({min_val}-{max_val})")
        
        elif isinstance(data, pd.DataFrame):
            # Multiple predictions
            missing_features = set(self.feature_names) - set(data.columns)
            if missing_features:
                raise ValueError(f"Missing features: {missing_features}")
        
        else:
            raise ValueError("Input data must be a dictionary or pandas DataFrame")
    
    def preprocess_data(self, data):
        """Preprocess input data"""
        if isinstance(data, dict):
            # Convert to DataFrame
            df = pd.DataFrame([data])
        else:
            df = data.copy()
        
        # Ensure correct column order
        df = df[self.feature_names]
        
        # Scale the data
        scaled_data = self.scaler.transform(df)
        
        return df, scaled_data
    
    def predict_single(self, patient_data, model_name=None):
        """Make prediction for a single patient"""
        self.validate_input(patient_data)
        original_data, scaled_data = self.preprocess_data(patient_data)
        
        if model_name and model_name in self.models:
            models_to_use = {model_name: self.models[model_name]}
        else:
            models_to_use = self.models
        
        results = {}
        
        for name, model in models_to_use.items():
            # Determine if model needs scaled data
            use_scaled = name in ['Logistic Regression', 'Support Vector Machine']
            input_data = scaled_data if use_scaled else original_data.values
            
            # Make prediction
            prediction = model.predict(input_data)[0]
            probability = model.predict_proba(input_data)[0]
            
            results[name] = {
                'prediction': int(prediction),
                'probability_no_diabetes': float(probability[0]),
                'probability_diabetes': float(probability[1]),
                'risk_level': self.get_risk_level(probability[1])
            }
        
        return results
    
    def predict_batch(self, data_file, output_file=None):
        """Make predictions for multiple patients from a CSV file"""
        print(f"Loading data from {data_file}...")
        df = pd.read_csv(data_file)
        
        self.validate_input(df)
        original_data, scaled_data = self.preprocess_data(df)
        
        results = []
        
        for idx in range(len(df)):
            patient_data = df.iloc[idx].to_dict()
            patient_results = self.predict_single(patient_data)
            
            # Add patient info
            result_row = {'patient_id': idx + 1}
            result_row.update(patient_data)
            
            # Add predictions from all models
            for model_name, pred_result in patient_results.items():
                result_row[f'{model_name}_prediction'] = pred_result['prediction']
                result_row[f'{model_name}_probability'] = pred_result['probability_diabetes']
                result_row[f'{model_name}_risk_level'] = pred_result['risk_level']
            
            results.append(result_row)
        
        results_df = pd.DataFrame(results)
        
        if output_file:
            results_df.to_csv(output_file, index=False)
            print(f"Results saved to {output_file}")
        
        return results_df
    
    def get_risk_level(self, probability):
        """Determine risk level based on probability"""
        if probability < 0.3:
            return "Low Risk"
        elif probability < 0.6:
            return "Moderate Risk"
        else:
            return "High Risk"
    
    def create_prediction_report(self, patient_data, output_file=None):
        """Create a detailed prediction report for a patient"""
        results = self.predict_single(patient_data)
        
        report = {
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'patient_data': patient_data,
            'predictions': results,
            'summary': {
                'consensus_prediction': self.get_consensus_prediction(results),
                'average_probability': np.mean([r['probability_diabetes'] for r in results.values()]),
                'recommendation': self.get_recommendation(results)
            }
        }
        
        if output_file:
            with open(output_file, 'w') as f:
                json.dump(report, f, indent=2)
            print(f"Detailed report saved to {output_file}")
        
        return report
    
    def get_consensus_prediction(self, results):
        """Get consensus prediction from all models"""
        predictions = [r['prediction'] for r in results.values()]
        return 1 if sum(predictions) > len(predictions) / 2 else 0
    
    def get_recommendation(self, results):
        """Generate recommendation based on predictions"""
        avg_prob = np.mean([r['probability_diabetes'] for r in results.values()])
        consensus = self.get_consensus_prediction(results)
        
        if consensus == 1:
            if avg_prob > 0.8:
                return "High risk of diabetes. Immediate medical consultation recommended."
            elif avg_prob > 0.6:
                return "Moderate to high risk of diabetes. Medical consultation recommended."
            else:
                return "Some risk of diabetes detected. Consider lifestyle changes and medical consultation."
        else:
            if avg_prob > 0.4:
                return "Low to moderate risk. Consider preventive measures and regular health checkups."
            else:
                return "Low risk of diabetes. Maintain healthy lifestyle."
    
    def interactive_prediction(self):
        """Interactive mode for single patient prediction"""
        print("\n" + "="*50)
        print("INTERACTIVE DIABETES PREDICTION")
        print("="*50)
        
        patient_data = {}
        
        prompts = {
            'Pregnancies': "Number of pregnancies (0-20): ",
            'Glucose': "Glucose level (mg/dL, typically 70-200): ",
            'BloodPressure': "Blood pressure (mmHg, typically 80-120): ",
            'SkinThickness': "Skin thickness (mm, typically 10-50): ",
            'Insulin': "Insulin level (μU/mL, typically 15-276): ",
            'BMI': "BMI (typically 18-40): ",
            'Pedigree': "Diabetes pedigree function (0-2.5): ",
            'Age': "Age (years, 21-120): "
        }
        
        for feature, prompt in prompts.items():
            while True:
                try:
                    value = float(input(prompt))
                    patient_data[feature] = value
                    break
                except ValueError:
                    print("Please enter a valid number.")
        
        print("\nMaking predictions...")
        results = self.predict_single(patient_data)
        
        print("\n" + "="*50)
        print("PREDICTION RESULTS")
        print("="*50)
        
        print(f"\nPatient Data:")
        for feature, value in patient_data.items():
            print(f"  {feature}: {value}")
        
        print(f"\nModel Predictions:")
        for model_name, result in results.items():
            prediction_text = "Diabetes" if result['prediction'] == 1 else "No Diabetes"
            print(f"\n{model_name}:")
            print(f"  Prediction: {prediction_text}")
            print(f"  Probability of Diabetes: {result['probability_diabetes']:.3f}")
            print(f"  Risk Level: {result['risk_level']}")
        
        # Consensus
        consensus = self.get_consensus_prediction(results)
        avg_prob = np.mean([r['probability_diabetes'] for r in results.values()])
        consensus_text = "Diabetes" if consensus == 1 else "No Diabetes"
        
        print(f"\nConsensus Prediction: {consensus_text}")
        print(f"Average Probability: {avg_prob:.3f}")
        print(f"Recommendation: {self.get_recommendation(results)}")
        
        return results

def main():
    """Main function for the prediction utility"""
    print("DIABETES PREDICTION UTILITY")
    print("="*40)
    
    # Initialize utility
    utility = DiabetesPredictionUtility()
    
    try:
        # Load models
        utility.load_models()
        
        # Interactive mode
        print("\nChoose an option:")
        print("1. Interactive single prediction")
        print("2. Batch prediction from CSV file")
        print("3. Example single prediction")
        
        choice = input("\nEnter your choice (1-3): ").strip()
        
        if choice == '1':
            utility.interactive_prediction()
        
        elif choice == '2':
            csv_file = input("Enter path to CSV file: ").strip()
            if os.path.exists(csv_file):
                output_file = f"results/batch_predictions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
                results = utility.predict_batch(csv_file, output_file)
                print(f"\nProcessed {len(results)} patients")
            else:
                print("File not found!")
        
        elif choice == '3':
            # Example prediction
            example_patient = {
                'Pregnancies': 6,
                'Glucose': 148,
                'BloodPressure': 72,
                'SkinThickness': 35,
                'Insulin': 0,
                'BMI': 33.6,
                'Pedigree': 0.627,
                'Age': 50
            }
            
            print("\nExample Patient Data:")
            for feature, value in example_patient.items():
                print(f"  {feature}: {value}")
            
            results = utility.predict_single(example_patient)
            
            print("\nPrediction Results:")
            for model_name, result in results.items():
                prediction_text = "Diabetes" if result['prediction'] == 1 else "No Diabetes"
                print(f"\n{model_name}:")
                print(f"  Prediction: {prediction_text}")
                print(f"  Probability: {result['probability_diabetes']:.3f}")
                print(f"  Risk Level: {result['risk_level']}")
        
        else:
            print("Invalid choice!")
    
    except Exception as e:
        print(f"Error: {e}")
        print("Make sure you have trained the models first by running 1.0-tnt-diabetes-prediction.py")

if __name__ == "__main__":
    main()
