import pandas as pd
import numpy as np
import json
from typing import Dict, List, Any, Optional, Union
import requests
from dotenv import load_dotenv
import os
import re

# Load environment variables
load_dotenv()

class RemediationGenerator:
    """Generate remediation suggestions for validation issues."""
    
    def __init__(self, model_name: str = "mistralai/Mistral-7B-Instruct-v0.2"):
        """
        Initialize the remediation generator.
        
        Args:
            model_name: The name of the Hugging Face model to use
        """
        self.model_name = model_name
        self.api_key = os.getenv("HUGGINGFACE_API_KEY")
        if not self.api_key:
            print("Warning: HUGGINGFACE_API_KEY not found in environment variables")
        
        self.api_url = f"https://api-inference.huggingface.co/models/{model_name}"
        self.headers = {"Authorization": f"Bearer {self.api_key}"}
        
        # Common field constraints (to be used when LLM isn't available)
        self.field_constraints = {
            'Identifier_Type': {
                'allowed_values': ['CUSIP', 'ISIN', 'SEDOL', 'INTERNAL'],
                'default': 'INTERNAL'
            },
            'Private_Placement': {
                'allowed_values': ['Y', 'N'],
                'default': 'N'
            },
            'Accounting_Intent': {
                'allowed_values': ['AFS', 'HTM', 'EQ'],
                'default': 'AFS'
            }
        }
    
    def suggest_remediation(self, validation_results: pd.DataFrame, rules: List[Dict[str, Any]]) -> pd.DataFrame:
        """
        Generate remediation suggestions for validation failures.
        
        Args:
            validation_results: DataFrame with validation results
            rules: List of validation rules
            
        Returns:
            DataFrame with remediation suggestions
        """
        # Create a copy of the validation results
        result = validation_results.copy()
        
        # Add remediation suggestion column if it doesn't exist
        if 'remediation_suggestions' not in result.columns:
            result['remediation_suggestions'] = [[] for _ in range(len(result))]
        
        # Get failed records
        failed_records = result[~result['validation_passed']]
        
        print(f"Generating remediation suggestions for {len(failed_records)} records")
        
        # Process each failed record
        for idx, record in failed_records.iterrows():
            failed_rule_ids = record['failed_rules']
            
            # Get details of the failed rules
            failed_rules = [r for r in rules if r.get('id') in failed_rule_ids]
            
            # Generate remediation suggestions for each rule
            suggestions = []
            
            for rule in failed_rules:
                suggestion = self._generate_suggestion_for_rule(record, rule)
                if suggestion:
                    suggestions.append(suggestion)
            
            # Add suggestions to the result
            result.at[idx, 'remediation_suggestions'] = suggestions
        
        return result
    
    def _generate_suggestion_for_rule(self, record: pd.Series, rule: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Generate a remediation suggestion for a specific rule failure.
        
        Args:
            record: The record that failed validation
            rule: The validation rule that failed
            
        Returns:
            Remediation suggestion dictionary or None if no suggestion is available
        """
        rule_id = rule.get('id', 'unknown')
        rule_name = rule.get('name', 'Unknown Rule')
        rule_fields = rule.get('fields', [])
        
        # Generate suggestion based on the rule type and fields
        if self.api_key:
            # Use LLM to generate suggestion
            try:
                return self._generate_llm_suggestion(record, rule)
            except Exception as e:
                print(f"Error generating LLM suggestion for rule {rule_id}: {e}")
                # Fall back to rule-based suggestion
                return self._generate_rule_based_suggestion(record, rule)
        else:
            # Use rule-based suggestion
            return self._generate_rule_based_suggestion(record, rule)
    
    def _generate_llm_suggestion(self, record: pd.Series, rule: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Generate remediation suggestion using LLM.
        
        Args:
            record: The record that failed validation
            rule: The validation rule that failed
            
        Returns:
            Remediation suggestion dictionary or None if no suggestion is available
        """
        rule_id = rule.get('id', 'unknown')
        rule_name = rule.get('name', 'Unknown Rule')
        rule_fields = rule.get('fields', [])
        rule_description = rule.get('description', '')
        
        # Prepare prompt for the model
        prompt = f"""
        Task: Generate a remediation suggestion for a data validation issue.
        
        Rule that failed: {rule_name}
        Rule description: {rule_description}
        Fields involved: {', '.join(rule_fields)}
        
        Current record values:
        {record.to_string()}
        
        Provide a specific remediation suggestion to fix this issue. The suggestion should include:
        1. What fields need to be changed
        2. What the new values should be
        3. Why this change will resolve the issue
        
        Format your response as a concise JSON object with 'fields_to_update', 'suggested_values', and 'explanation' keys.
        """
        
        response = self._query_model(prompt)
        
        # Try to extract JSON from the response
        suggestion_data = self._extract_json_from_text(response)
        
        if suggestion_data:
            try:
                data = json.loads(suggestion_data)
                
                # Add rule information to the suggestion
                data['rule_id'] = rule_id
                data['rule_name'] = rule_name
                
                return data
            except json.JSONDecodeError:
                print(f"Error parsing JSON response for rule {rule_id}")
        
        return None
    
    def _generate_rule_based_suggestion(self, record: pd.Series, rule: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Generate remediation suggestion based on predefined rules.
        
        Args:
            record: The record that failed validation
            rule: The validation rule that failed
            
        Returns:
            Remediation suggestion dictionary or None if no suggestion is available
        """
        rule_id = rule.get('id', 'unknown')
        rule_name = rule.get('name', 'Unknown Rule')
        rule_fields = rule.get('fields', [])
        
        # Initialize suggestion
        suggestion = {
            'rule_id': rule_id,
            'rule_name': rule_name,
            'fields_to_update': {},
            'suggested_values': {},
            'explanation': f"Fix validation issue for {rule_name}"
        }
        
        # Check each field in the rule
        for field in rule_fields:
            # Skip fields that don't exist in the record
            if field not in record:
                continue
            
            current_value = record[field]
            
            # Check if this is a common field with known constraints
            if field in self.field_constraints:
                constraints = self.field_constraints[field]
                
                # If current value is not in allowed values, suggest the default value
                if current_value not in constraints['allowed_values']:
                    suggestion['fields_to_update'][field] = current_value
                    suggestion['suggested_values'][field] = constraints['default']
                    suggestion['explanation'] += f"\n- {field}: Change from '{current_value}' to '{constraints['default']}' to comply with allowed values {constraints['allowed_values']}"
            
            # Check for empty string or NaN values
            elif pd.isna(current_value) or (isinstance(current_value, str) and current_value.strip() == ''):
                suggestion['fields_to_update'][field] = current_value
                suggestion['suggested_values'][field] = f"[PROVIDE VALID {field.upper()}]"
                suggestion['explanation'] += f"\n- {field}: Provide a valid value instead of empty or missing data"
            
            # For unique_id field, ensure it's not duplicated
            elif field.lower() == 'unique_id' or field.lower() == 'unique_id':
                if pd.isna(current_value) or (isinstance(current_value, str) and current_value.strip() == ''):
                    suggestion['fields_to_update'][field] = current_value
                    suggestion['suggested_values'][field] = f"UNIQUE_ID_{pd.Timestamp.now().strftime('%Y%m%d%H%M%S')}"
                    suggestion['explanation'] += f"\n- {field}: Provide a valid unique identifier"
        
        # If no fields to update were found, suggest checking for missing fields
        if not suggestion['fields_to_update']:
            missing_fields = [f for f in rule_fields if f not in record]
            if missing_fields:
                suggestion['explanation'] += f"\n- Add missing fields: {', '.join(missing_fields)}"
                for field in missing_fields:
                    suggestion['fields_to_update'][field] = None
                    suggestion['suggested_values'][field] = f"[PROVIDE {field.upper()}]"
        
        # If there are fields to update, return the suggestion
        if suggestion['fields_to_update'] or missing_fields:
            return suggestion
        
        # Fall back to a generic suggestion
        suggestion['explanation'] = f"Please review validation issue for {rule_name}. The specific issue could not be automatically determined."
        return suggestion
    
    def suggest_anomaly_remediation(self, anomaly_results: pd.DataFrame, anomaly_details: Dict[int, List[Dict[str, Any]]]) -> pd.DataFrame:
        """
        Generate remediation suggestions for anomalies.
        
        Args:
            anomaly_results: DataFrame with anomaly detection results
            anomaly_details: Dictionary mapping anomaly indices to anomaly feature details
            
        Returns:
            DataFrame with remediation suggestions
        """
        # Create a copy of the anomaly results
        result = anomaly_results.copy()
        
        # Add remediation suggestion column if it doesn't exist
        if 'remediation_suggestions' not in result.columns:
            result['remediation_suggestions'] = [[] for _ in range(len(result))]
        
        # Get anomalous records
        anomalous_records = result[result['anomaly']]
        
        print(f"Generating remediation suggestions for {len(anomalous_records)} anomalous records")
        
        # Process each anomalous record
        for idx, record in anomalous_records.iterrows():
            # Get anomaly details
            details = anomaly_details.get(idx, [])
            
            # Generate remediation suggestion
            if self.api_key:
                suggestion = self._generate_llm_anomaly_suggestion(record, details)
                if not suggestion:
                    suggestion = self._generate_rule_based_anomaly_suggestion(record, details)
            else:
                suggestion = self._generate_rule_based_anomaly_suggestion(record, details)
            
            # Add suggestion to the result
            if suggestion:
                result.at[idx, 'remediation_suggestions'] = [suggestion]
        
        return result
    
    def _generate_llm_anomaly_suggestion(self, record: pd.Series, anomaly_details: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        """
        Generate remediation suggestion for anomaly using LLM.
        
        Args:
            record: The anomalous record
            anomaly_details: List of anomalous features and details
            
        Returns:
            Remediation suggestion dictionary or None if no suggestion is available
        """
        # Prepare details as string
        details_str = ""
        for detail in anomaly_details:
            if detail['type'] == 'numeric':
                details_str += f"- {detail['feature']}: {detail['value']:.2f} ({detail['reason']})\n"
            else:
                details_str += f"- {detail['feature']}: {detail['value']} ({detail['reason']})\n"
        
        # Prepare prompt for the model
        prompt = f"""
        Task: Generate a remediation suggestion for a data anomaly.
        
        Anomalous record:
        {record.to_string()}
        
        Anomaly details:
        {details_str}
        
        Provide a specific remediation suggestion to fix this anomaly. The suggestion should include:
        1. What fields need to be changed
        2. What the new values should be
        3. Why these changes will resolve the anomaly
        
        Format your response as a concise JSON object with 'fields_to_update', 'suggested_values', and 'explanation' keys.
        """
        
        response = self._query_model(prompt)
        
        # Try to extract JSON from the response
        suggestion_data = self._extract_json_from_text(response)
        
        if suggestion_data:
            try:
                data = json.loads(suggestion_data)
                
                # Add anomaly information to the suggestion
                data['type'] = 'anomaly'
                data['anomaly_score'] = float(record['anomaly_score'])
                
                return data
            except json.JSONDecodeError:
                print("Error parsing JSON response for anomaly suggestion")
        
        return None
    
    def _generate_rule_based_anomaly_suggestion(self, record: pd.Series, anomaly_details: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Generate remediation suggestion for anomaly based on predefined rules.
        
        Args:
            record: The anomalous record
            anomaly_details: List of anomalous features and details
            
        Returns:
            Remediation suggestion dictionary
        """
        # Initialize suggestion
        suggestion = {
            'type': 'anomaly',
            'anomaly_score': float(record['anomaly_score']),
            'fields_to_update': {},
            'suggested_values': {},
            'explanation': "Fix anomalous values that deviate significantly from normal patterns"
        }
        
        # Process each anomalous feature
        for detail in anomaly_details:
            feature = detail['feature']
            value = detail['value']
            
            if detail['type'] == 'numeric':
                # For numeric features, suggest a value closer to the mean
                # This is a simplistic approach - in practice, you might want to use more sophisticated methods
                suggestion['fields_to_update'][feature] = value
                
                # Check if there's a z-score available
                if 'z_score' in detail and abs(detail['z_score']) > 0:
                    # Calculate a suggested value that reduces the z-score
                    z_score = detail['z_score']
                    # Move toward the mean by reducing the z-score to at most ±2
                    direction = 1 if z_score < 0 else -1
                    adjustment_factor = max(0, abs(z_score) - 2) * direction
                    suggested_value = value - adjustment_factor * abs(value) * 0.2
                    
                    suggestion['suggested_values'][feature] = round(suggested_value, 2)
                    suggestion['explanation'] += f"\n- {feature}: The value {value} has an extreme z-score of {z_score:.2f}. Consider changing to {suggested_value:.2f} to make it less anomalous."
                else:
                    # Without z-score, suggest a more generic adjustment
                    suggested_value = value * 0.8 if value > 0 else value * 1.2
                    suggestion['suggested_values'][feature] = round(suggested_value, 2)
                    suggestion['explanation'] += f"\n- {feature}: The value {value} is unusual. Consider changing to {suggested_value:.2f} to make it less anomalous."
            
            elif detail['type'] == 'categorical':
                # For categorical features, suggest the most common value
                if 'most_common' in detail:
                    suggested_value = detail['most_common']
                    suggestion['fields_to_update'][feature] = value
                    suggestion['suggested_values'][feature] = suggested_value
                    suggestion['explanation'] += f"\n- {feature}: The value '{value}' is unusual. Consider changing to '{suggested_value}' which is more common."
                elif feature in self.field_constraints and value not in self.field_constraints[feature]['allowed_values']:
                    # If the value doesn't match known constraints, suggest the default value
                    suggested_value = self.field_constraints[feature]['default']
                    suggestion['fields_to_update'][feature] = value
                    suggestion['suggested_values'][feature] = suggested_value
                    suggestion['explanation'] += f"\n- {feature}: The value '{value}' is not allowed. Consider changing to '{suggested_value}' which is a valid value."
        
        return suggestion
    
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
                "max_new_tokens": 1024,
                "temperature": 0.1,  # Low temperature for more deterministic outputs
                "top_p": 0.95
            }
        }
        
        response = requests.post(self.api_url, headers=self.headers, json=payload)
        
        if response.status_code != 200:
            raise Exception(f"API request failed with status code {response.status_code}: {response.text}")
        
        return response.json()[0]["generated_text"].replace(prompt, "").strip()
    


    def _extract_json_from_text(self, text: str) -> Optional[str]:
        """
        Extract JSON from text response.
        
        Args:
            text: The text potentially containing JSON
            
        Returns:
            The extracted JSON string, or None if not found
        """
        # Try to find JSON array or object in text
        json_patterns = [
            r'\{.*\}',  # JSON object pattern
            r'\[.*\]',  # JSON array pattern
        ]
        
        for pattern in json_patterns:
            matches = re.search(pattern, text, re.DOTALL)
            if matches:
                json_str = matches.group(0)
                # Validate it's proper JSON
                try:
                    json.loads(json_str)
                    return json_str
                except:
                    continue
        
        return None