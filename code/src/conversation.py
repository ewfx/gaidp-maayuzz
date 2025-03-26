import pandas as pd
import numpy as np
import json
import re
from typing import Dict, List, Any, Optional, Tuple
import requests
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class ConversationHandler:
    """Handle natural language interactions with the regulatory compliance system."""
    
    def __init__(self, model_name: str = "mistralai/Mistral-7B-Instruct-v0.2"):
        """
        Initialize the conversation handler.
        
        Args:
            model_name: The name of the Hugging Face model to use
        """
        self.model_name = model_name
        self.api_key = os.getenv("HUGGINGFACE_API_KEY")
        if not self.api_key:
            print("Warning: HUGGINGFACE_API_KEY not found in environment variables")
        
        self.api_url = f"https://api-inference.huggingface.co/models/{model_name}"
        self.headers = {"Authorization": f"Bearer {self.api_key}"}
        
        # Store conversation history
        self.history = []
        
        # Store current context
        self.current_document = None
        self.current_data = None
        self.current_rules = []
        self.validation_results = None
        self.anomaly_results = None
    
    def process_message(self, message: str) -> str:
        """
        Process a user message and generate a response.
        
        Args:
            message: The user's message
            
        Returns:
            The system's response
        """
        # Add user message to history
        self.history.append({"role": "user", "content": message})
        
        # Check for command intents
        command, args = self._extract_command(message)
        
        if command:
            # Handle specific commands
            response = self._handle_command(command, args)
        else:
            # Handle general conversation
            response = self._generate_conversation_response(message)
        
        # Add system response to history
        self.history.append({"role": "assistant", "content": response})
        
        return response
    
    def _extract_command(self, message: str) -> Tuple[Optional[str], Dict[str, Any]]:
        """
        Extract command intent from user message.
        
        Args:
            message: The user's message
            
        Returns:
            Tuple of (command, arguments) or (None, {}) if no command found
        """
        # Define command patterns
        command_patterns = {
            "extract_rules": [
                r"extract rules",
                r"identify rules",
                r"find rules",
                r"extract regulatory requirements"
            ],
            "validate_data": [
                r"validate data",
                r"check data",
                r"run validation",
                r"validate against rules"
            ],
            "detect_anomalies": [
                r"detect anomalies",
                r"find anomalies",
                r"identify outliers",
                r"check for anomalies"
            ],
            "suggest_remediation": [
                r"suggest remediation",
                r"how to fix",
                r"remediation suggestions",
                r"fix issues"
            ],
            "explain_rule": [
                r"explain rule",
                r"tell me about rule",
                r"what does rule .+ mean",
                r"explain validation rule"
            ],
            "show_summary": [
                r"show summary",
                r"summarize results",
                r"validation summary",
                r"give me a summary"
            ]
        }
        
        # Check for command matches
        for command, patterns in command_patterns.items():
            for pattern in patterns:
                match = re.search(pattern, message.lower())
                if match:
                    # Extract arguments based on the command type
                    args = {}
                    
                    if command == "explain_rule":
                        # Extract rule ID
                        rule_match = re.search(r"rule[_\s]*(\w+)", message.lower())
                        if rule_match:
                            args["rule_id"] = rule_match.group(1)
                    
                    return command, args
        
        return None, {}
    
    def _handle_command(self, command: str, args: Dict[str, Any]) -> str:
        """
        Handle a command intent.
        
        Args:
            command: The identified command
            args: Command arguments
            
        Returns:
            The system's response
        """
        if command == "extract_rules":
            return self._handle_extract_rules()
        elif command == "validate_data":
            return self._handle_validate_data()
        elif command == "detect_anomalies":
            return self._handle_detect_anomalies()
        elif command == "suggest_remediation":
            return self._handle_suggest_remediation()
        elif command == "explain_rule":
            return self._handle_explain_rule(args.get("rule_id"))
        elif command == "show_summary":
            return self._handle_show_summary()
        else:
            return "I'm not sure how to handle that command."
    
    def _handle_extract_rules(self) -> str:
        """
        Handle extract rules command.
        
        Returns:
            Response message
        """
        if not self.current_document:
            return "Please upload a regulatory document first."
        

        return "I'll extract rules from the regulatory document. This functionality will be implemented in the Streamlit app."
    
    def _handle_validate_data(self) -> str:
        """
        Handle validate data command.
        
        Returns:
            Response message
        """
        if not self.current_data:
            return "Please upload data to validate first."
        
        if not self.current_rules:
            return "Please extract rules from a regulatory document first."
        
        # For now, just return a placeholder message
        # In a real implementation, this would call the validation components
        return "I'll validate your data against the extracted rules. This functionality will be implemented in the Streamlit app."
    
    def _handle_detect_anomalies(self) -> str:
        """
        Handle detect anomalies command.
        
        Returns:
            Response message
        """
        if not self.current_data:
            return "Please upload data to analyze first."
        

        return "I'll detect anomalies in your data. This functionality will be implemented in the Streamlit app."
    
    def _handle_suggest_remediation(self) -> str:
        """
        Handle suggest remediation command.
        
        Returns:
            Response message
        """
        if not self.validation_results and not self.anomaly_results:
            return "Please validate your data or detect anomalies first."
        

        return "I'll suggest remediation actions for the issues in your data. This functionality will be implemented in the Streamlit app."
    
    def _handle_explain_rule(self, rule_id: Optional[str]) -> str:
        """
        Handle explain rule command.
        
        Args:
            rule_id: ID of the rule to explain
            
        Returns:
            Response message
        """
        if not rule_id:
            return "Please specify which rule you'd like me to explain."
        
        if not self.current_rules:
            return "Please extract rules from a regulatory document first."
        

        return f"I'll explain rule {rule_id} for you. This functionality will be implemented in the Streamlit app."
    
    def _handle_show_summary(self) -> str:
        """
        Handle show summary command.
        
        Returns:
            Response message
        """
        if not self.validation_results and not self.anomaly_results:
            return "There are no validation or anomaly detection results to summarize yet."

        return "I'll show you a summary of the validation and anomaly detection results. This functionality will be implemented in the Streamlit app."
    
    def _generate_conversation_response(self, message: str) -> str:
        """
        Generate a response for general conversation.
        
        Args:
            message: The user's message
            
        Returns:
            The system's response
        """
        if self.api_key:
            # Use LLM to generate response
            try:
                return self._generate_llm_response(message)
            except Exception as e:
                print(f"Error generating LLM response: {e}")
                return self._generate_fallback_response(message)
        else:
            return self._generate_fallback_response(message)
    
    def _generate_llm_response(self, message: str) -> str:
        """
        Generate a response using LLM.
        
        Args:
            message: The user's message
            
        Returns:
            The generated response
        """
        # Prepare the conversation context
        context = self._prepare_context()
        
        # Prepare the prompt for the model
        prompt = f"""
        You are an intelligent assistant for regulatory compliance in the financial sector. Your role is to help users extract regulatory requirements, validate data, detect anomalies, and suggest remediation actions.

        {context}
        
        User: {message}
        
        Assistant:
        """
        
        response = self._query_model(prompt)
        
        return response.strip()
    
    def _prepare_context(self) -> str:
        """
        Prepare context information for the model.
        
        Returns:
            Context information as a string
        """
        context = "Current context:\n"
        
        if self.current_document:
            context += "- A regulatory document has been uploaded.\n"
        else:
            context += "- No regulatory document has been uploaded yet.\n"
        
        if self.current_data is not None:
            context += f"- Data with {len(self.current_data)} records has been uploaded.\n"
        else:
            context += "- No data has been uploaded yet.\n"
        
        if self.current_rules:
            context += f"- {len(self.current_rules)} validation rules have been extracted.\n"
        else:
            context += "- No validation rules have been extracted yet.\n"
        
        if self.validation_results is not None:
            failed_count = len(self.validation_results) - self.validation_results['validation_passed'].sum()
            context += f"- Validation results: {failed_count} out of {len(self.validation_results)} records failed validation.\n"
        
        if self.anomaly_results is not None:
            anomaly_count = self.anomaly_results['anomaly'].sum()
            context += f"- Anomaly detection results: {anomaly_count} out of {len(self.anomaly_results)} records were flagged as anomalies.\n"
        
        return context
    
    def _generate_fallback_response(self, message: str) -> str:
        """
        Generate a fallback response when LLM is not available.
        
        Args:
            message: The user's message
            
        Returns:
            The fallback response
        """
        # Simple keyword-based response
        if "hello" in message.lower() or "hi" in message.lower():
            return "Hello! I'm your regulatory compliance assistant. How can I help you today?"
        
        if "help" in message.lower():
            return "I can help you with regulatory compliance. You can ask me to:\n- Extract rules from a regulatory document\n- Validate your data against rules\n- Detect anomalies in your data\n- Suggest remediation actions for issues\n- Explain specific rules"
        
        if "thank" in message.lower():
            return "You're welcome! Is there anything else I can help you with?"
        
        # Default response
        return "I understand you're asking about regulatory compliance. To get started, please upload a regulatory document and some data to analyze."
    
    def _query_model(self, prompt: str) -> str:
        """
        Query the Hugging Face model via the Inference API.
        
        Args:
            prompt: The text prompt to send to the model
            
        Returns:
            The model's response
        """
        payload = {
            "inputs": prompt,
            "parameters": {
                "max_new_tokens": 512,
                "temperature": 0.7,  
                "top_p": 0.9
            }
        }
        
        response = requests.post(self.api_url, headers=self.headers, json=payload)
        
        if response.status_code != 200:
            raise Exception(f"API request failed with status code {response.status_code}: {response.text}")
        
        return response.json()[0]["generated_text"].replace(prompt, "").strip()
