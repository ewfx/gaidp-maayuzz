# test_hf_processor.py

import os
import unittest
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.document_processing import HuggingFaceProcessor, load_document

class TestHuggingFaceProcessor(unittest.TestCase):
    """Test Hugging Face API-based document processing."""
    
    def setUp(self):
        """Set up test environment."""
        self.processor = HuggingFaceProcessor()
        
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
    
    def test_process_text(self):
        """Test processing of regulatory text."""
        processed = self.processor.process_text(self.sample_text)
        
        # Check that we have sections
        self.assertIn("sections", processed)
        self.assertIn("rule_sections", processed)
        
        # Print sections for inspection
        print("\nExtracted Sections:")
        for title, content in processed["sections"].items():
            print(f"Section: {title}")
            print(f"Content (first 100 chars): {content[:100].strip()}...")
        
        # Print rule sections for inspection
        print("\nRule Sections:")
        for title, content in processed["rule_sections"].items():
            print(f"Rule Section: {title}")
            print(f"Content (first 100 chars): {content[:100].strip()}...")
    
    def test_extract_rules(self):
        """Test extraction of rules from text."""
        rules = self.processor.extract_rules_from_text(self.sample_text)
        
        # Print rules for inspection
        print("\nExtracted Rules:")
        for rule in rules:
            print(f"Rule from section: {rule.get('section', 'Unknown')}")
            print(f"Fields: {', '.join(rule.get('fields', []))}")
            if 'type' in rule:
                print(f"Type: {rule['type']}")
            if 'requirement' in rule:
                print(f"Requirement: {rule['requirement'][:100].strip()}...")
            print()

if __name__ == "__main__":
    unittest.main()