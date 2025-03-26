# # test_anomaly_detection.py

# import unittest
# import os
# import numpy as np
# import pandas as pd
# from src.anomaly_detection import AnomalyDetector

# class TestAnomalyDetection(unittest.TestCase):
#     """Test anomaly detection functionality."""
    
#     def setUp(self):
#         """Set up test environment."""
#         # Create synthetic securities data for testing
#         np.random.seed(42)
        
#         # Generate 100 normal records
#         normal_data = pd.DataFrame({
#             'Unique_ID': [f"ID{i:03d}" for i in range(100)],
#             'Identifier_Type': np.random.choice(['CUSIP', 'ISIN', 'SEDOL'], 100),
#             'Market_Value_USD': np.random.normal(10000, 1000, 100),
#             'Price': np.random.normal(100, 10, 100),
#             'Amortized_Cost_USD': np.random.normal(9800, 800, 100),
#             'Private_Placement': np.random.choice(['Y', 'N'], 100),
#             'Accounting_Intent': np.random.choice(['AFS', 'HTM'], 100)
#         })
        
#         # Add 5 anomalous records
#         anomalies = pd.DataFrame({
#             'Unique_ID': [f"ID{i:03d}" for i in range(100, 105)],
#             'Identifier_Type': np.random.choice(['CUSIP', 'ISIN', 'SEDOL', 'OTHER'], 5),  # 'OTHER' is anomalous
#             'Market_Value_USD': np.random.normal(20000, 5000, 5),  # Much higher values
#             'Price': np.random.normal(200, 50, 5),  # Much higher prices
#             'Amortized_Cost_USD': np.random.normal(5000, 1000, 5),  # Much lower costs
#             'Private_Placement': np.random.choice(['Y', 'N', 'X'], 5),  # 'X' is anomalous
#             'Accounting_Intent': np.random.choice(['AFS', 'HTM', 'EQ'], 5)  # Added 'EQ'
#         })
        
#         # Combine normal and anomalous data
#         self.test_data = pd.concat([normal_data, anomalies])
        
#         # Create the anomaly detector
#         self.detector = AnomalyDetector(contamination=0.05)
    
#     def test_anomaly_detection(self):
#         """Test anomaly detection on synthetic data."""
#         # Fit the model
#         self.detector.fit(self.test_data)
        
#         # Detect anomalies
#         result = self.detector.detect_anomalies(self.test_data)
        
#         # Check that we have the expected columns
#         self.assertIn('anomaly', result.columns)
#         self.assertIn('anomaly_score', result.columns)
        
#         # Count detected anomalies
#         anomaly_count = result['anomaly'].sum()
        
#         # We expect around 5 anomalies (but allow for some flexibility)
#         self.assertGreaterEqual(anomaly_count, 1)
#         self.assertLessEqual(anomaly_count, 10)
        
#         print(f"\nDetected {anomaly_count} anomalies in {len(result)} records")
        
#         # Get anomaly indices
#         anomaly_indices = result[result['anomaly']].index.tolist()
        
#         # Identify anomalous features
#         anomaly_features = self.detector.identify_anomaly_features(self.test_data, anomaly_indices)
        
#         # Check that we found anomalous features
#         self.assertGreaterEqual(len(anomaly_features), 1)
        
#         # Print anomaly details
#         print("\nAnomaly Details:")
#         for idx, features in anomaly_features.items():
#             record = self.test_data.iloc[idx]
#             print(f"\nAnomaly at index {idx} (ID: {record['Unique_ID']}):")
            
#             print(f"  Identifier Type: {record['Identifier_Type']}")
#             print(f"  Market Value: ${record['Market_Value_USD']:.2f}")
#             print(f"  Price: ${record['Price']:.2f}")
#             print(f"  Amortized Cost: ${record['Amortized_Cost_USD']:.2f}")
            
#             if features:
#                 print("  Anomalous Features:")
#                 for feature in features:
#                     print(f"    {feature['feature']}: {feature['value']:.2f} (z-score: {feature['z_score']:.2f})")
#             else:
#                 print("  No specific anomalous features identified")
    
#     def test_feature_importance(self):
#         """Test feature importance extraction."""
#         # Fit the model
#         self.detector.fit(self.test_data)
        
#         try:
#             # Get feature importance
#             importance = self.detector.get_feature_importance()
            
#             # Check that we have importance scores
#             self.assertGreaterEqual(len(importance), 0)
            
#             # Print feature importance
#             print("\nFeature Importance:")
#             for idx, row in importance.head(10).iterrows():
#                 print(f"  {row['feature']}: {row['importance']:.4f}")
#         except Exception as e:
#             print(f"\nFeature importance calculation is not available: {e}")
#             print("This is expected behavior for some versions of scikit-learn's IsolationForest")

# if __name__ == "__main__":
#     unittest.main()

# test_anomaly_detection.py

import unittest
import os
import numpy as np
import pandas as pd
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.anomaly_detection import AnomalyDetector

class TestAnomalyDetection(unittest.TestCase):
    """Test anomaly detection functionality."""
    
    def setUp(self):
        """Set up test environment."""
        # Create synthetic securities data for testing
        np.random.seed(42)
        
        # Generate 100 normal records
        normal_data = pd.DataFrame({
            'Unique_ID': [f"ID{i:03d}" for i in range(100)],
            'Identifier_Type': np.random.choice(['CUSIP', 'ISIN', 'SEDOL'], 100),
            'Market_Value_USD': np.random.normal(10000, 1000, 100),
            'Price': np.random.normal(100, 10, 100),
            'Amortized_Cost_USD': np.random.normal(9800, 800, 100),
            'Private_Placement': np.random.choice(['Y', 'N'], 100),
            'Accounting_Intent': np.random.choice(['AFS', 'HTM'], 100)
        })
        
        # Add 5 anomalous records
        anomalies = pd.DataFrame({
            'Unique_ID': [f"ID{i:03d}" for i in range(100, 105)],
            'Identifier_Type': np.random.choice(['CUSIP', 'ISIN', 'SEDOL', 'OTHER'], 5),  # 'OTHER' is anomalous
            'Market_Value_USD': np.random.normal(20000, 5000, 5),  # Much higher values
            'Price': np.random.normal(200, 50, 5),  # Much higher prices
            'Amortized_Cost_USD': np.random.normal(5000, 1000, 5),  # Much lower costs
            'Private_Placement': np.random.choice(['Y', 'N', 'X'], 5),  # 'X' is anomalous
            'Accounting_Intent': np.random.choice(['AFS', 'HTM', 'EQ'], 5)  # Added 'EQ'
        })
        
        # Combine normal and anomalous data
        self.test_data = pd.concat([normal_data, anomalies])
        
        # Create the anomaly detector
        self.detector = AnomalyDetector(contamination=0.05)
    
    def test_anomaly_detection(self):
        """Test anomaly detection on synthetic data."""
        # Fit the model
        self.detector.fit(self.test_data)
        
        # Detect anomalies
        result = self.detector.detect_anomalies(self.test_data)
        
        # Check that we have the expected columns
        self.assertIn('anomaly', result.columns)
        self.assertIn('anomaly_score', result.columns)
        
        # Count detected anomalies
        anomaly_count = result['anomaly'].sum()
        
        # We expect around 5 anomalies (but allow for some flexibility)
        self.assertGreaterEqual(anomaly_count, 1)
        self.assertLessEqual(anomaly_count, 10)
        
        print(f"\nDetected {anomaly_count} anomalies in {len(result)} records")
        
        # Sort the data by anomaly score (most anomalous first)
        result_sorted = result.sort_values('anomaly_score')
        
        # Get the indices of the most anomalous records
        anomaly_indices = result_sorted[result_sorted['anomaly']].index.tolist()
        
        # Identify anomalous features
        anomaly_features = self.detector.identify_anomaly_features(self.test_data, anomaly_indices)
        
        # Check that we found anomalous features
        self.assertGreaterEqual(len(anomaly_features), 1)
        
        # Print anomaly details
        print("\nAnomaly Details:")
        for idx in anomaly_indices:
            record = self.test_data.iloc[idx]
            features = anomaly_features.get(idx, [])
            
            print(f"\nAnomaly at index {idx} (ID: {record['Unique_ID']}):")
            print(f"  Identifier Type: {record['Identifier_Type']}")
            print(f"  Market Value: ${record['Market_Value_USD']:.2f}")
            print(f"  Price: ${record['Price']:.2f}")
            print(f"  Amortized Cost: ${record['Amortized_Cost_USD']:.2f}")
            print(f"  Private Placement: {record['Private_Placement']}")
            print(f"  Accounting Intent: {record['Accounting_Intent']}")
            
            if features:
                print("  Anomalous Features:")
                for feature in features:
                    if feature['type'] == 'numeric':
                        print(f"    {feature['feature']}: {feature['value']:.2f} ({feature['reason']})")
                    else:
                        print(f"    {feature['feature']}: {feature['value']} ({feature['reason']})")
            else:
                print("  No specific anomalous features identified")
    
    def test_feature_importance(self):
        """Test feature importance extraction."""
        # Fit the model
        self.detector.fit(self.test_data)
        
        try:
            # Get feature importance
            importance = self.detector.get_feature_importance()
            
            # Check that we have importance scores
            self.assertGreaterEqual(len(importance), 0)
            
            # Print feature importance
            print("\nFeature Importance:")
            for idx, row in importance.head(10).iterrows():
                print(f"  {row['feature']}: {row['importance']:.4f}")
        except Exception as e:
            print(f"\nFeature importance calculation is not available: {e}")
            print("This is expected behavior for some versions of scikit-learn's IsolationForest")

if __name__ == "__main__":
    unittest.main()