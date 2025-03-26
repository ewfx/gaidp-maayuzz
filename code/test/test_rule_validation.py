# test_rule_validation.py

import unittest
import os
import pandas as pd
import json
import shutil
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.document_processing import HuggingFaceProcessor
from src.rule_extraction import RuleExtractor
from src.rule_storage import RuleStorage
from src.rule_validation import RuleValidator

class TestRuleValidation(unittest.TestCase):
    """Test rule validation functionality."""
    
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
        
        # Create the rule validator
        self.validator = RuleValidator()
    
    def tearDown(self):
        """Clean up after tests."""
        # Remove temporary directories
        if os.path.exists(self.test_storage_dir):
            shutil.rmtree(self.test_storage_dir)
    
    def test_load_rules(self):
        """Test loading rules from a file."""
        # Load rules
        self.validator.load_rules(self.rules_file)
        
        # Check that rules were loaded
        self.assertGreater(len(self.validator.rules), 0)
        
        print(f"\nLoaded {len(self.validator.rules)} rules")
    
    def test_validate(self):
        """Test validating data against rules."""
        # Load rules
        self.validator.load_rules(self.rules_file)
        
        # Validate the data
        validation_results = self.validator.validate(self.sample_data)
        
        # Check that validation results include the expected columns
        self.assertIn('validation_passed', validation_results.columns)
        self.assertIn('failed_rules', validation_results.columns)
        self.assertIn('validation_messages', validation_results.columns)
        
        # Check that at least one record failed validation
        failed_count = len(validation_results) - validation_results['validation_passed'].sum()
        self.assertGreater(failed_count, 0)
        
        print(f"\nValidation results: {failed_count} out of {len(validation_results)} records failed validation")
        
        # Print details of failed records
        failed_records = self.validator.get_failed_records(validation_results)
        if not failed_records.empty:
            print("\nFailed Records:")
            for idx, row in failed_records.iterrows():
                print(f"Record {idx} (ID: {row['Unique_ID']}):")
                print(f"  Failed Rules: {', '.join(row['failed_rules'])}")
                print(f"  Messages: {'; '.join(row['validation_messages'])}")
    
    def test_validation_report(self):
        """Test generating a validation report."""
        # Load rules
        self.validator.load_rules(self.rules_file)
        
        # Validate the data
        validation_results = self.validator.validate(self.sample_data)
        
        # Generate a validation report
        report = self.validator.generate_validation_report(validation_results)
        
        # Check that the report has the expected fields
        self.assertIn('total_records', report)
        self.assertIn('passed_records', report)
        self.assertIn('failed_records', report)
        self.assertIn('pass_rate', report)
        self.assertIn('rule_frequency', report)
        
        # Print the report
        print("\nValidation Report:")
        print(f"  Total Records: {report['total_records']}")
        print(f"  Passed Records: {report['passed_records']}")
        print(f"  Failed Records: {report['failed_records']}")
        print(f"  Pass Rate: {report['pass_rate']:.2%}")
        print("  Rule Failures:")
        for rule_id, count in report['rule_frequency'].items():
            print(f"    {rule_id}: {count}")
    
    def test_field_validation(self):
        """Test validating a specific field."""
        # Load rules
        self.validator.load_rules(self.rules_file)
        
        # Validate the Identifier_Type field
        validation_results = self.validator.validate_field('Identifier_Type', self.sample_data)
        
        # Check that validation results include the expected columns
        self.assertIn('validation_passed', validation_results.columns)
        
        # Print validation results for the field
        failed_count = len(validation_results) - validation_results['validation_passed'].sum()
        print(f"\nIdentifier_Type validation: {failed_count} out of {len(validation_results)} records failed validation")
        
        # Validate the Private_Placement field
        validation_results = self.validator.validate_field('Private_Placement', self.sample_data)
        
        # Print validation results for the field
        failed_count = len(validation_results) - validation_results['validation_passed'].sum()
        print(f"Private_Placement validation: {failed_count} out of {len(validation_results)} records failed validation")

if __name__ == "__main__":
    unittest.main()