# test_conversation.py

import unittest
import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.conversation import ConversationHandler

class TestConversation(unittest.TestCase):
    """Test the conversation handler functionality."""
    
    def setUp(self):
        """Set up the test environment."""
        self.handler = ConversationHandler()
    
    def test_basic_conversation(self):
        """Test basic conversation functionality."""
        # Test greeting
        response = self.handler.process_message("Hello")
        print(f"User: Hello\nAssistant: {response}")
        self.assertIsNotNone(response)
        self.assertNotEqual(response, "")
        
        # Test help request
        response = self.handler.process_message("What can you help me with?")
        print(f"\nUser: What can you help me with?\nAssistant: {response}")
        self.assertIsNotNone(response)
        self.assertNotEqual(response, "")
    
    def test_command_recognition(self):
        """Test command recognition."""
        # Test extract rules command
        response = self.handler.process_message("Please extract rules from the document")
        print(f"\nUser: Please extract rules from the document\nAssistant: {response}")
        self.assertIn("regulatory document", response)
        
        # Test validate data command
        response = self.handler.process_message("Validate my data against the rules")
        print(f"\nUser: Validate my data against the rules\nAssistant: {response}")
        self.assertIn("data", response)
        
        # Test anomaly detection command
        response = self.handler.process_message("Detect anomalies in my data")
        print(f"\nUser: Detect anomalies in my data\nAssistant: {response}")
        self.assertIn("anomalies", response)
        
        # Test remediation suggestion command
        response = self.handler.process_message("Suggest remediation for the issues")
        print(f"\nUser: Suggest remediation for the issues\nAssistant: {response}")
        self.assertIn("remediation", response)
    
    def test_rule_explanation(self):
        """Test rule explanation command."""
        response = self.handler.process_message("Explain rule 123 to me")
        print(f"\nUser: Explain rule 123 to me\nAssistant: {response}")
        self.assertIn("rule 123", response)
    
    def test_conversation_context(self):
        """Test conversation with context."""
        # Set up some context
        self.handler.current_document = "sample_document.pdf"
        self.handler.current_rules = [{"id": "rule_1", "name": "Test Rule"}]
        
        # Ask about rules
        response = self.handler.process_message("What rules have been extracted?")
        print(f"\nUser: What rules have been extracted?\nAssistant: {response}")
        self.assertIsNotNone(response)
        self.assertNotEqual(response, "")

if __name__ == "__main__":
    unittest.main()