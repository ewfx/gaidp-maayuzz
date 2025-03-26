# test_rule_extraction.py

import unittest
import os
import json
import pandas as pd
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.document_processing import HuggingFaceProcessor
from src.rule_extraction import RuleExtractor

class TestRuleExtraction(unittest.TestCase):
    """Test rule extraction functionality."""
    
    def setUp(self):
        """Set up test environment."""
        self.document_processor = HuggingFaceProcessor()
        self.rule_extractor = RuleExtractor()
        
        # Sample regulatory text for testing
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
    
    def test_rule_extraction(self):
        """Test extraction and formalization of rules."""
        # Process the document
        processed_doc = self.document_processor.process_text(self.sample_text)
        
        # Extract rules
        rules = self.rule_extractor.extract_rules(processed_doc)
        
        # Check that we extracted at least one rule
        self.assertGreater(len(rules), 0)
        
        # Print extracted rules for inspection
        print("\nExtracted Rules:")
        for rule in rules:
            print(f"Rule ID: {rule.get('id', 'Unknown')}")
            print(f"Name: {rule.get('name', 'Unknown')}")
            print(f"Type: {rule.get('type', 'Unknown')}")
            print(f"Fields: {', '.join(rule.get('fields', []))}")
            print(f"Source: {rule.get('source_section', 'Unknown')}")
            print()
            
            # Print the validation code
            print("Validation Code:")
            print(rule.get('validation_code', 'No code generated'))
            print("-" * 50)
    
    def test_rule_executable(self):
        """Test that the generated validation code is executable."""
        # Process the document
        processed_doc = self.document_processor.process_text(self.sample_text)
        
        # Extract rules
        rules = self.rule_extractor.extract_rules(processed_doc)
        
        if not rules:
            self.skipTest("No rules were extracted")
        
        # Try to execute the validation code
        
        # Create a sample row with all possible fields
        sample_row = pd.Series({
            "Unique_ID": "ID123",
            "Identifier_Type": "CUSIP",
            "Identifier_Value": "12345678",
            "Private_Placement": "N",
            "Accounting_Intent": "AFS",
            "Security_Description_1": "Corporate Bond",
            "Security_Description_2": "Financial",
            "Security_Description_3": "5-year",
            "Amortized_Cost_USD": 10000,
            "Market_Value_USD": 10500,
            "Current_Face_Value_USD": 10000,
            "Original_Face_Value_USD": 10000,
            "Allowance_for_Credit_Losses": 0,
            "Writeoffs": 0,
            "Price": 105.0,
            "Pricing_Date": "2023-01-01",
            "Book_Yield": "5.0",
            "Purchase_Date": "2022-01-01",
            "Currency": "USD"
        })
        
        success_count = 0
        fail_count = 0
        
        for rule in rules:
            validation_code = rule.get('validation_code')
            if not validation_code:
                continue
                
            print(f"\nTesting rule: {rule.get('id')}")
            print(f"Fields: {', '.join(rule.get('fields', []))}")
            
            try:
                # Create a namespace for the function
                namespace = {'pd': pd}
                
                # Execute the validation code to define the function
                exec(validation_code, namespace)
                
                # Get the validate_rule function
                validate_rule = namespace['validate_rule']
                
                # Call the function
                result = validate_rule(sample_row)
                
                # Check that result is boolean
                self.assertIsInstance(result, bool)
                
                print(f"Rule {rule.get('id')} execution successful, result: {result}")
                success_count += 1
            except Exception as e:
                print(f"Rule {rule.get('id')} execution failed: {e}")
                print(f"Validation code:\n{validation_code}")
                fail_count += 1
        
        print(f"\nRule execution summary: {success_count} succeeded, {fail_count} failed")
        
        # Test should pass if at least one rule executed successfully
        self.assertGreater(success_count, 0, "No rules executed successfully")

if __name__ == "__main__":
    unittest.main()