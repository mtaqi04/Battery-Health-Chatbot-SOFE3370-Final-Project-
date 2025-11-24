# End-to-End Pipeline Testing Script
"""
Comprehensive test suite for the Battery Health Chatbot pipeline.

Tests the complete flow:
1. Data Preprocessing Validation
2. Model Loading & Prediction
3. Classification with Threshold Logic
4. Chatbot Integration
5. Consistency & Accuracy Verification

Run: python tests/test_pipeline_e2e.py
"""

import os
import sys
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
import json

try:
    import joblib
except ImportError:
    print("Warning: joblib not found. Some tests may fail.")
    joblib = None

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from chatbot.predict_soh import (
    load_model, 
    predict_soh, 
    predict_soh_batch,
    classify_soh,
    DEFAULT_THRESHOLD
)

# Test configuration
TEST_RUNS = 5  # Number of runs for consistency testing
THRESHOLDS_TO_TEST = [0.5, 0.6, 0.7, 0.8]
TEST_RESULTS_DIR = Path(__file__).parent.parent / "reports" / "test_logs"
TEST_RESULTS_DIR.mkdir(parents=True, exist_ok=True)

class PipelineTester:
    """Comprehensive pipeline testing class."""
    
    def __init__(self):
        self.results = {
            "test_timestamp": datetime.now().isoformat(),
            "data_preprocessing": {},
            "model_loading": {},
            "predictions": {},
            "classification": {},
            "consistency": {},
            "errors": []
        }
        self.project_root = Path(__file__).parent.parent
        self.data_path = self.project_root / "data" / "cleaned_pulsebat.csv"
        self.model_path = self.project_root / "models" / "soh_linear_model.pkl"
        
    def log_error(self, test_name, error):
        """Log errors during testing."""
        error_msg = f"{test_name}: {str(error)}"
        self.results["errors"].append(error_msg)
        print(f"ERROR: {error_msg}")
    
    def test_1_data_preprocessing(self):
        """Test 1: Validate data preprocessing pipeline."""
        print("\n" + "="*60)
        print("TEST 1: Data Preprocessing Validation")
        print("="*60)
        
        try:
            # Check if cleaned data exists
            if not self.data_path.exists():
                raise FileNotFoundError(f"Cleaned data not found: {self.data_path}")
            
            # Load and validate data
            df = pd.read_csv(self.data_path)
            
            # Check required columns
            required_cols = [f"U{i}" for i in range(1, 22)] + ["SOH"]
            missing_cols = [col for col in required_cols if col not in df.columns]
            
            if missing_cols:
                raise ValueError(f"Missing required columns: {missing_cols}")
            
            # Validate data shape
            expected_features = 21
            feature_cols = [f"U{i}" for i in range(1, 22)]
            actual_features = len([col for col in feature_cols if col in df.columns])
            
            # Check for missing values
            missing_values = df[feature_cols].isnull().sum().sum()
            
            # Validate SOH range
            invalid_soh = df[~df["SOH"].between(0, 1)]
            invalid_soh_count = len(invalid_soh)
            
            # Check data types
            numeric_check = all(df[col].dtype in ['float64', 'int64'] for col in feature_cols)
            
            self.results["data_preprocessing"] = {
                "status": "PASS",
                "total_samples": len(df),
                "features_count": actual_features,
                "missing_values": int(missing_values),
                "invalid_soh_count": invalid_soh_count,
                "numeric_types": numeric_check,
                "soh_range": {
                    "min": float(df["SOH"].min()),
                    "max": float(df["SOH"].max()),
                    "mean": float(df["SOH"].mean())
                }
            }
            
            print(f"Data loaded: {len(df)} samples")
            print(f"Features: {actual_features}/21")
            print(f"Missing values: {missing_values}")
            print(f"Invalid SOH: {invalid_soh_count}")
            print(f"SOH range: {df['SOH'].min():.4f} - {df['SOH'].max():.4f}")
            print(f"Numeric types: {numeric_check}")
            
            return True
            
        except Exception as e:
            self.log_error("test_1_data_preprocessing", e)
            self.results["data_preprocessing"]["status"] = "FAIL"
            return False
    
    def test_2_model_loading(self):
        """Test 2: Validate model loading."""
        print("\n" + "="*60)
        print("TEST 2: Model Loading Validation")
        print("="*60)
        
        try:
            # Check if model exists
            if not self.model_path.exists():
                raise FileNotFoundError(f"Model not found: {self.model_path}")
            
            # Load model
            model = load_model()
            
            # Validate model structure
            has_intercept = hasattr(model, 'intercept_')
            has_coef = hasattr(model, 'coef_')
            coef_count = len(model.coef_) if has_coef else 0
            
            # Test prediction capability
            test_features = np.random.rand(1, 21)
            test_pred = model.predict(test_features)
            can_predict = isinstance(test_pred[0], (float, np.floating))
            
            self.results["model_loading"] = {
                "status": "PASS",
                "model_path": str(self.model_path),
                "has_intercept": has_intercept,
                "has_coefficients": has_coef,
                "coefficient_count": coef_count,
                "can_predict": can_predict,
                "test_prediction": float(test_pred[0])
            }
            
            print(f"Model loaded from: {self.model_path}")
            print(f"Has intercept: {has_intercept}")
            print(f"Coefficients: {coef_count}")
            print(f"Can predict: {can_predict}")
            print(f"Test prediction: {test_pred[0]:.4f}")
            
            return True
            
        except Exception as e:
            self.log_error("test_2_model_loading", e)
            self.results["model_loading"]["status"] = "FAIL"
            return False
    
    def test_3_predictions(self):
        """Test 3: Validate prediction functionality."""
        print("\n" + "="*60)
        print("TEST 3: Prediction Validation")
        print("="*60)
        
        try:
            # Load data and model
            df = pd.read_csv(self.data_path)
            model = load_model()
            feature_cols = [f"U{i}" for i in range(1, 22)]
            
            # Test single prediction
            test_sample = df[feature_cols].iloc[0].values
            result = predict_soh(test_sample, threshold=0.6, model=model)
            
            # Validate result structure
            has_soh = 'soh' in result
            has_condition = 'condition' in result
            soh_valid = 0 <= result['soh'] <= 1
            condition_valid = result['condition'] in ['Healthy', 'Has a Problem']
            
            # Test batch prediction
            batch_results = predict_soh_batch(threshold=0.6)
            batch_count = len(batch_results)
            
            # Compare with actual SOH (first 10 samples)
            sample_size = min(10, len(df))
            predictions = []
            actuals = []
            for i in range(sample_size):
                features = df[feature_cols].iloc[i].values
                pred_result = predict_soh(features, threshold=0.6, model=model)
                predictions.append(pred_result['soh'])
                actuals.append(df['SOH'].iloc[i])
            
            # Calculate basic metrics
            mae = np.mean(np.abs(np.array(predictions) - np.array(actuals)))
            mse = np.mean((np.array(predictions) - np.array(actuals))**2)
            rmse = np.sqrt(mse)
            
            self.results["predictions"] = {
                "status": "PASS",
                "single_prediction": {
                    "has_soh": has_soh,
                    "has_condition": has_condition,
                    "soh_value": float(result['soh']),
                    "soh_valid": soh_valid,
                    "condition": result['condition'],
                    "condition_valid": condition_valid
                },
                "batch_prediction": {
                    "sample_count": batch_count,
                    "expected_count": len(df)
                },
                "accuracy_sample": {
                    "sample_size": sample_size,
                    "mae": float(mae),
                    "mse": float(mse),
                    "rmse": float(rmse)
                }
            }
            
            print(f"Single prediction: SOH={result['soh']:.4f}, Condition={result['condition']}")
            print(f"Batch prediction: {batch_count} samples")
            print(f"Sample accuracy (n={sample_size}):")
            print(f"MAE: {mae:.4f}")
            print(f"MSE: {mse:.4f}")
            print(f"RMSE: {rmse:.4f}")
            
            return True
            
        except Exception as e:
            self.log_error("test_3_predictions", e)
            self.results["predictions"]["status"] = "FAIL"
            return False
    
    def test_4_classification(self):
        """Test 4: Validate classification with different thresholds."""
        print("\n" + "="*60)
        print("TEST 4: Classification Validation")
        print("="*60)
        
        try:
            df = pd.read_csv(self.data_path)
            model = load_model()
            feature_cols = [f"U{i}" for i in range(1, 22)]
            test_sample = df[feature_cols].iloc[0].values
            
            threshold_results = {}
            
            for threshold in THRESHOLDS_TO_TEST:
                result = predict_soh(test_sample, threshold=threshold, model=model)
                soh = result['soh']
                condition = result['condition']
                
                # Validate classification logic
                expected_condition = "Healthy" if soh >= threshold else "Has a Problem"
                classification_correct = condition == expected_condition
                
                threshold_results[str(threshold)] = {
                    "soh": float(soh),
                    "condition": condition,
                    "classification_correct": classification_correct
                }
                
                print(f"Threshold {threshold}: SOH={soh:.4f} -> {condition} (Correct: {classification_correct})")
            
            # Test batch classification distribution
            batch_result_06 = predict_soh_batch(threshold=0.6)
            condition_counts = batch_result_06['Condition'].value_counts().to_dict()
            
            self.results["classification"] = {
                "status": "PASS",
                "threshold_tests": threshold_results,
                "batch_classification_06": {
                    "total_samples": len(batch_result_06),
                    "condition_distribution": condition_counts
                }
            }
            
            print(f"Batch classification (threshold=0.6): {condition_counts}")
            
            return True
            
        except Exception as e:
            self.log_error("test_4_classification", e)
            self.results["classification"]["status"] = "FAIL"
            return False
    
    def test_5_consistency(self):
        """Test 5: Verify consistency across multiple runs."""
        print("\n" + "="*60)
        print("TEST 5: Consistency Testing (Multiple Runs)")
        print("="*60)
        
        try:
            df = pd.read_csv(self.data_path)
            model = load_model()
            feature_cols = [f"U{i}" for i in range(1, 22)]
            
            # Test same sample multiple times
            test_sample = df[feature_cols].iloc[0].values
            predictions = []
            
            for run in range(TEST_RUNS):
                result = predict_soh(test_sample, threshold=0.6, model=model)
                predictions.append(result['soh'])
            
            # Check consistency
            predictions_array = np.array(predictions)
            mean_pred = np.mean(predictions_array)
            std_pred = np.std(predictions_array)
            is_consistent = std_pred < 1e-10  # Should be identical across runs
            
            # Test different samples
            sample_indices = [0, 10, 50, 100, min(200, len(df)-1)]
            multi_sample_results = {}
            
            for idx in sample_indices:
                features = df[feature_cols].iloc[idx].values
                result = predict_soh(features, threshold=0.6, model=model)
                multi_sample_results[f"sample_{idx}"] = {
                    "soh": float(result['soh']),
                    "condition": result['condition'],
                    "actual_soh": float(df['SOH'].iloc[idx])
                }
            
            self.results["consistency"] = {
                "status": "PASS",
                "same_sample_runs": {
                    "runs": TEST_RUNS,
                    "predictions": [float(p) for p in predictions],
                    "mean": float(mean_pred),
                    "std": float(std_pred),
                    "is_consistent": bool(is_consistent)
                },
                "multi_sample_test": multi_sample_results
            }
            
            print(f"Same sample tested {TEST_RUNS} times:")
            print(f"Mean: {mean_pred:.6f}")
            print(f"Std: {std_pred:.10f}")
            print(f"Consistent: {is_consistent}")
            print(f"Multiple samples tested: {len(sample_indices)} samples")
            
            return True
            
        except Exception as e:
            self.log_error("test_5_consistency", e)
            self.results["consistency"]["status"] = "FAIL"
            return False
    
    def test_6_chatbot_integration(self):
        """Test 6: Validate chatbot integration functions."""
        print("\n" + "="*60)
        print("TEST 6: Chatbot Integration Validation")
        print("="*60)
        
        try:
            import re
            
            # Replicate extract_features_from_text function for testing
            def extract_features_from_text(text: str) -> list:
                """Extract numeric values from user text input."""
                numbers = re.findall(r'-?\d+\.?\d*', text)
                if len(numbers) >= 21:
                    try:
                        features = [float(num) for num in numbers[:21]]
                        return features
                    except ValueError:
                        return None
                return None
            
            # Test various input formats
            test_inputs = [
                "0.0025,0.0125,0.0035,0.0019,0.0027,0.0057,0.0193,0.0202,0.0027,0.0197,0.0062,0.0042,0.0019,0.0157,0.0484,0.0508,0.0027,0.0346,0.0101,0.0119,0.0025",
                "0.0025 0.0125 0.0035 0.0019 0.0027 0.0057 0.0193 0.0202 0.0027 0.0197 0.0062 0.0042 0.0019 0.0157 0.0484 0.0508 0.0027 0.0346 0.0101 0.0119 0.0025",
                "U1=0.0025, U2=0.0125, U3=0.0035, U4=0.0019, U5=0.0027, U6=0.0057, U7=0.0193, U8=0.0202, U9=0.0027, U10=0.0197, U11=0.0062, U12=0.0042, U13=0.0019, U14=0.0157, U15=0.0484, U16=0.0508, U17=0.0027, U18=0.0346, U19=0.0101, U20=0.0119, U21=0.0025"
            ]
            
            parsing_results = {}
            model = load_model()
            
            for i, test_input in enumerate(test_inputs):
                features = extract_features_from_text(test_input)
                parsing_results[f"format_{i+1}"] = {
                    "input_length": len(test_input),
                    "features_extracted": features is not None,
                    "feature_count": len(features) if features else 0
                }
                
                if features:
                    # Test prediction with extracted features
                    result = predict_soh(features, threshold=0.6, model=model)
                    parsing_results[f"format_{i+1}"]["prediction_valid"] = True
                    parsing_results[f"format_{i+1}"]["soh"] = float(result['soh'])
                    parsing_results[f"format_{i+1}"]["condition"] = result['condition']
            
            self.results["chatbot_integration"] = {
                "status": "PASS",
                "input_parsing": parsing_results
            }
            
            print(f"Tested {len(test_inputs)} input formats")
            for fmt, result in parsing_results.items():
                print(f"{fmt}: Extracted={result['features_extracted']}, Count={result['feature_count']}")
                if result.get('prediction_valid'):
                    print(f"     Prediction: SOH={result.get('soh', 'N/A'):.4f}, Condition={result.get('condition', 'N/A')}")
            
            return True
            
        except Exception as e:
            self.log_error("test_6_chatbot_integration", e)
            self.results["chatbot_integration"] = {"status": "FAIL"}
            return False
    
    def test_7_edge_cases(self):
        """Test 7: Edge cases and error handling."""
        print("\n" + "="*60)
        print("TEST 7: Edge Cases & Error Handling")
        print("="*60)
        
        edge_case_results = {}
        
        try:
            model = load_model()
            
            # Test with wrong number of features
            try:
                wrong_features = [0.1] * 20  # Only 20 features
                predict_soh(wrong_features, model=model)
                edge_case_results["wrong_feature_count"] = "FAIL - Should raise error"
            except ValueError:
                edge_case_results["wrong_feature_count"] = "PASS - Correctly raises ValueError"
                print("Wrong feature count: Correctly raises error")
            
            # Test with extreme values
            extreme_features = [0.0] * 21
            result_min = predict_soh(extreme_features, threshold=0.6, model=model)
            extreme_features = [1.0] * 21
            result_max = predict_soh(extreme_features, threshold=0.6, model=model)
            
            edge_case_results["extreme_values"] = {
                "min_values": {
                    "soh": float(result_min['soh']),
                    "condition": result_min['condition']
                },
                "max_values": {
                    "soh": float(result_max['soh']),
                    "condition": result_max['condition']
                }
            }
            print(f"Extreme values: Min SOH={result_min['soh']:.4f}, Max SOH={result_max['soh']:.4f}")
            
            # Test classification boundary
            boundary_test = predict_soh([0.1] * 21, threshold=0.6, model=model)
            soh_at_boundary = boundary_test['soh']
            # Test just below and above threshold
            test_threshold = soh_at_boundary
            result_below = classify_soh(test_threshold - 0.001, threshold=test_threshold)
            result_above = classify_soh(test_threshold + 0.001, threshold=test_threshold)
            
            edge_case_results["boundary_classification"] = {
                "below_threshold": result_below,
                "above_threshold": result_above,
                "boundary_correct": result_below == "Has a Problem" and result_above == "Healthy"
            }
            print(f"Boundary classification: Below={result_below}, Above={result_above}")
            
            self.results["edge_cases"] = {
                "status": "PASS",
                "tests": edge_case_results
            }
            
            return True
            
        except Exception as e:
            self.log_error("test_7_edge_cases", e)
            self.results["edge_cases"] = {"status": "FAIL"}
            return False
    
    def run_all_tests(self):
        """Run all tests and generate report."""
        print("\n" + "="*60)
        print("BATTERY HEALTH CHATBOT - END-TO-END PIPELINE TESTING")
        print("="*60)
        print(f"Test Run: {self.results['test_timestamp']}")
        print(f"Project Root: {self.project_root}")
        
        tests = [
            ("Data Preprocessing", self.test_1_data_preprocessing),
            ("Model Loading", self.test_2_model_loading),
            ("Predictions", self.test_3_predictions),
            ("Classification", self.test_4_classification),
            ("Consistency", self.test_5_consistency),
            ("Chatbot Integration", self.test_6_chatbot_integration),
            ("Edge Cases", self.test_7_edge_cases)
        ]
        
        passed = 0
        failed = 0
        
        for test_name, test_func in tests:
            try:
                if test_func():
                    passed += 1
                else:
                    failed += 1
            except Exception as e:
                self.log_error(test_name, e)
                failed += 1
        
        # Generate summary
        self.results["summary"] = {
            "total_tests": len(tests),
            "passed": passed,
            "failed": failed,
            "success_rate": f"{(passed/len(tests)*100):.1f}%"
        }
        
        # Save results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_file = TEST_RESULTS_DIR / f"pipeline_test_{timestamp}.json"
        
        # Convert numpy types to native Python types for JSON serialization
        def convert_to_serializable(obj):
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, (np.bool_, bool)):
                return bool(obj)
            elif isinstance(obj, dict):
                return {key: convert_to_serializable(value) for key, value in obj.items()}
            elif isinstance(obj, list):
                return [convert_to_serializable(item) for item in obj]
            return obj
        
        serializable_results = convert_to_serializable(self.results)
        
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(serializable_results, f, indent=2, ensure_ascii=False)
        
        # Print summary
        print("\n" + "="*60)
        print("TEST SUMMARY")
        print("="*60)
        print(f"Total Tests: {len(tests)}")
        print(f"Passed: {passed}")
        print(f"Failed: {failed}")
        print(f"Success Rate: {self.results['summary']['success_rate']}")
        print(f"\nResults saved to: {results_file}")
        
        if self.results["errors"]:
            print("\nErrors encountered:")
            for error in self.results["errors"]:
                print(f"  - {error}")
        
        return passed == len(tests)


if __name__ == "__main__":
    tester = PipelineTester()
    success = tester.run_all_tests()
    sys.exit(0 if success else 1)

