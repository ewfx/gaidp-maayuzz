# test_remediation.py

import unittest
import os
import pandas as pd
import numpy as np
import json
import shutil
import re
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.document_processing import HuggingFaceProcessor
from src.rule_extraction import RuleExtractor
from src.rule_storage import RuleStorage
from src.rule_validation import RuleValidator
from src.anomaly_detection import AnomalyDetector
from src.remediation import RemediationGenerator

class TestRemediation(unittest.TestCase):
    """Test remediation suggestion functionality."""
    
    def setUp(self):
        """Set up test environment."""
        # Create temporary directories
        self.test_storage_dir = "test_rules"
        if os.path.exists(self.test_storage_dir):
            shutil.rmtree(self.test_storage_dir)
        os.makedirs(self.test_storage_dir)
        
        # Create the rule storage
        self.rule_storage = RuleStorage(storage_dir=self.test_storage_dir)
        
        # Extract and store rules
        self.document_processor = HuggingFaceProcessor()
        self.rule_extractor = RuleExtractor()
        
        # Sample regulatory text
        self.sample_text = """
SCHEDULE B—SECURITIES

Each BHC or IHC or SLHC should submit the two schedules comprising the FRY-14 Quarterly Securities data.

UNIQUE ID

A unique identifier must be included to identify each unique record. For a given security position, the same Unique ID should be used from one period to the next.

IDENTIFIER TYPE AND IDENTIFIER VALUE

Report individual security-level data for all securities. Generally, securities should always be reported with a public identifier, if available, such as CUSIP, ISIN, or SEDOL.

PRIVATE PLACEMENT

Please enter "Y" if the security is a private placement security or "N" if it is a publicly offered security.

ACCOUNTING INTENT

Indicate whether the security is available-for-sale (AFS) or held-to-maturity (HTM).
"""
        
        # Process the document and extract rules
        processed_doc = self.document_processor.process_text(self.sample_text)
        self.rules = self.rule_extractor.extract_rules(processed_doc)
        
        # Save rules to storage
        self.rules_file = self.rule_storage.save_rules(self.rules, "Schedule B Securities")
        
        # Create sample data
        self.sample_data = pd.DataFrame({
            'Unique_ID': ['ID001', 'ID002', 'ID003', 'ID004', 'ID005'],
            'Identifier_Type': ['CUSIP', 'ISIN', 'SEDOL', 'CUSIP', 'OTHER'],  # 'OTHER' is invalid
            'Identifier_Value': ['123456789', 'US123456789', '987654321', '123456ABC', ''],  # Last one is empty
            'Private_Placement': ['Y', 'N', 'Y', 'X', 'N'],  # 'X' is invalid
            'Accounting_Intent': ['AFS', 'HTM', 'AFS', 'HTM', 'HOLD']  # 'HOLD' is invalid
        })
        
        # Create and apply the rule validator
        self.validator = RuleValidator()
        self.validator.load_rules(self.rules_file)
        self.validation_results = self.validator.validate(self.sample_data)
        
        # Create anomaly detector and detect anomalies
        self.detector = AnomalyDetector(contamination=0.2)  # Set higher to get at least 1 anomaly
        self.detector.fit(self.sample_data)
        self.anomaly_results = self.detector.detect_anomalies(self.sample_data)
        self.anomaly_indices = self.anomaly_results[self.anomaly_results['anomaly']].index.tolist()
        self.anomaly_details = self.detector.identify_anomaly_features(self.sample_data, self.anomaly_indices)
        
        # Create the remediation generator
        self.remediation_generator = RemediationGenerator()
    
    def tearDown(self):
        """Clean up after tests."""
        # Remove temporary directories
        if os.path.exists(self.test_storage_dir):
            shutil.rmtree(self.test_storage_dir)
    
    def test_validation_remediation(self):
        """Test generating remediation suggestions for validation failures."""
        # Generate remediation suggestions
        remediation_results = self.remediation_generator.suggest_remediation(self.validation_results, self.rules)
        
        # Check that remediation suggestions were generated
        self.assertIn('remediation_suggestions', remediation_results.columns)
        
        # Get failed records
        failed_records = remediation_results[~remediation_results['validation_passed']]
        
        # Check that suggestions were generated for all failed records
        for _, record in failed_records.iterrows():
            self.assertGreater(len(record['remediation_suggestions']), 0)
        
        # Print remediation suggestions
        print("\nValidation Remediation Suggestions:")
        for idx, record in failed_records.iterrows():
            print(f"\nRecord {idx} (ID: {record['Unique_ID']}):")
            print(f"  Failed Rules: {', '.join(record['failed_rules'])}")
            
            for suggestion in record['remediation_suggestions']:
                print(f"  Suggestion for {suggestion.get('rule_name', 'unknown')}:")
                print(f"    Fields to update: {suggestion.get('fields_to_update', {})}")
                print(f"    Suggested values: {suggestion.get('suggested_values', {})}")
                print(f"    Explanation: {suggestion.get('explanation', '')}")
    
    def test_anomaly_remediation(self):
        """Test generating remediation suggestions for anomalies."""
        # Generate remediation suggestions
        remediation_results = self.remediation_generator.suggest_anomaly_remediation(self.anomaly_results, self.anomaly_details)
        
        # Check that remediation suggestions were generated
        self.assertIn('remediation_suggestions', remediation_results.columns)
        
        # Get anomalous records
        anomalous_records = remediation_results[remediation_results['anomaly']]
        
        # Print anomaly details and remediation suggestions
        print("\nAnomaly Remediation Suggestions:")
        for idx, record in anomalous_records.iterrows():
            print(f"\nAnomaly at index {idx} (ID: {record['Unique_ID']}):")
            print(f"  Anomaly Score: {record['anomaly_score']:.4f}")
            
            # Print anomaly details
            details = self.anomaly_details.get(idx, [])
            if details:
                print("  Anomalous Features:")
                for detail in details:
                    if detail['type'] == 'numeric':
                        print(f"    {detail['feature']}: {detail['value']:.2f} ({detail['reason']})")
                    else:
                        print(f"    {detail['feature']}: {detail['value']} ({detail['reason']})")
            
            # Print remediation suggestions
            suggestions = record['remediation_suggestions']
            if suggestions:
                for suggestion in suggestions:
                    print("  Remediation Suggestion:")
                    print(f"    Fields to update: {suggestion.get('fields_to_update', {})}")
                    print(f"    Suggested values: {suggestion.get('suggested_values', {})}")
                    print(f"    Explanation: {suggestion.get('explanation', '')}")
            else:
                print("  No remediation suggestions available")

if __name__ == "__main__":
    unittest.main()