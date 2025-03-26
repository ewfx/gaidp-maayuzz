import re
import json
from typing import Dict, List, Any, Optional, Union
import pandas as pd
import requests
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()

class RuleExtractor:
    """Extract and formalize validation rules from regulatory text."""
    
    def __init__(self, model_name: str = "mistralai/Mistral-7B-Instruct-v0.2"):
        """
        Initialize the rule extractor.
        
        Args:
            model_name: The name of the Hugging Face model to use
        """
        self.model_name = model_name
        self.api_key = os.getenv("HUGGINGFACE_API_KEY")
        if not self.api_key:
            print("Warning: HUGGINGFACE_API_KEY not found in environment variables")
        
        self.api_url = f"https://api-inference.huggingface.co/models/{model_name}"
        self.headers = {"Authorization": f"Bearer {self.api_key}"}
        
        # Maps common rule types to standardized categories
        self.rule_type_mapping = {
            "format": "format_validation",
            "value": "value_validation",
            "range": "range_validation",
            "cross-field": "cross_field_validation",
            "required": "required_field",
            "allowed values": "allowed_values",
            "pattern": "pattern_validation"
        }
    
    def extract_rules(self, regulatory_text: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Extract validation rules from processed regulatory text.
        
        Args:
            regulatory_text: The processed regulatory document
            
        Returns:
            List of formalized rules
        """
        extracted_rules = []
        
        # Process each rule section
        for section_title, section_content in regulatory_text.get("rule_sections", {}).items():
            # Extract rules from this section
            section_rules = self._extract_rules_from_section(section_title, section_content)
            extracted_rules.extend(section_rules)
        
        # Formalize rules into executable format
        formalized_rules = self._formalize_rules(extracted_rules)
        
        return formalized_rules
    
    def _extract_rules_from_section(self, section_title: str, section_content: str) -> List[Dict[str, Any]]:
        """
        Extract rules from a section of regulatory text.
        
        Args:
            section_title: The title of the section
            section_content: The content of the section
            
        Returns:
            List of extracted rules
        """
        rules = []
        
        # If API key is available, use the LLM for better extraction
        if self.api_key:
            try:
                # Prepare prompt for the model
                prompt = f"""
                Task: Extract detailed financial regulatory validation rules from this text.
                
                Text from "{section_title}" section:
                {section_content[:1500]}  # Limit content to prevent token overflow
                Assume you have to generate a dataset with the schedule information in spotlight, generate rules which would validate against the dataset
                A sample Dataset column fields would look like this : UniqueId, IdentifierType, IdentifierValue, PrivatePlacement, Security_Description_1, Security_Description_2, Security_Description_3, AmortizedCost, MarketFaceValue and so on check other fields
                For each rule, identify:
                1. The specific fields it applies to
                2. The rule type (format validation, range validation, required field, allowed values, cross field, etc.)
                3. The exact validation requirement
                4. The condition or criteria that must be met
                
                Format your response as a JSON list, with each rule having 'fields', 'rule_type', 'requirement', and 'condition' keys.
                """
                
                # Call the model
                response = self._query_model(prompt)
                
                # Try to extract JSON from the response
                json_str = self._extract_json_from_text(response)
                if json_str:
                    extracted_rules = json.loads(json_str)
                    
                    # Process the extracted rules
                    if isinstance(extracted_rules, list):
                        for rule in extracted_rules:
                            rule["section"] = section_title
                            rule["text"] = section_content[:500]  # Store a snippet of the original text
                            rules.append(rule)
                    elif isinstance(extracted_rules, dict):
                        extracted_rules["section"] = section_title
                        extracted_rules["text"] = section_content[:500]
                        rules.append(extracted_rules)
            except Exception as e:
                print(f"Error extracting rules using LLM: {e}")
        
        # If can't extract rules using the LLM or no rules were found, fall back to regex-based extraction
        if not rules:
            rules = self._extract_rules_with_regex(section_title, section_content)
        
        return rules
    
    def _extract_rules_with_regex(self, section_title: str, section_content: str) -> List[Dict[str, Any]]:
        """
        Extract rules using regex patterns.
        
        Args:
            section_title: The title of the section
            section_content: The content of the section
            
        Returns:
            List of extracted rules
        """
        rules = []
        
        # Split content into paragraphs
        paragraphs = section_content.split('\n\n')
        
        for paragraph in paragraphs:
            if len(paragraph.strip()) < 10:
                continue
                
            # Look for rule indicators
            if any(word in paragraph.lower() for word in ["must", "shall", "required", "should", "report"]):
                # Try to identify fields mentioned
                fields = self._extract_fields(paragraph)
                
                # Determine rule type based on keywords
                rule_type = "validation"  # Default
                if "format" in paragraph.lower():
                    rule_type = "format_validation"
                elif "range" in paragraph.lower() or "between" in paragraph.lower():
                    rule_type = "range_validation"
                elif "required" in paragraph.lower():
                    rule_type = "required_field"
                elif "valid" in paragraph.lower() and "values" in paragraph.lower():
                    rule_type = "allowed_values"
                elif "cross" in paragraph.lower() and "field" in paragraph.lower():
                    rule_type = "cross_field_validation"
                
                # Extract condition or criteria
                condition = paragraph
                
                rule = {
                    "section": section_title,
                    "text": paragraph,
                    "fields": fields,
                    "rule_type": rule_type,
                    "requirement": paragraph,
                    "condition": condition
                }
                
                rules.append(rule)
        
        return rules
    
    def _extract_fields(self, text: str) -> List[str]:
        """
        Extract field names mentioned in rule text.
        
        Args:
            text: The rule text
            
        Returns:
            List of identified field names
        """
        # List of field names to look for
        field_names = [
            "Unique ID", "Identifier Type", "Identifier Value", 
            "Private Placement", "Security Description", 
            "Amortized Cost", "Market Value", "Current Face Value",
            "Original Face Value", "Allowance for Credit Losses",
            "Writeoffs", "Accounting Intent", "Price", 
            "Pricing Date", "Book Yield", "Purchase Date", "Currency"
        ]
        
        # Find exact matches
        exact_matches = [field for field in field_names if field in text]
        
        # If no exact matches, try to find partial matches
        if not exact_matches:
            for field in field_names:
                # Create a regex that matches the field name less strictly
                pattern = r'(?i)\b' + re.escape(field.lower()) + r'(?:s|es)?\b'
                if re.search(pattern, text.lower()):
                    exact_matches.append(field)
        
        return exact_matches
    
    def _formalize_rules(self, extracted_rules: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Formalize extracted rules into executable format.
        
        Args:
            extracted_rules: The rules extracted from regulatory text
            
        Returns:
            List of formalized rules ready for execution
        """
        formalized_rules = []
        rule_ids = set()  # Track rule IDs to avoid duplicates
        
        for rule in extracted_rules:
            # Convert rule to executable format
            formalized_rule = self._formalize_rule(rule)
            if formalized_rule:
                # Check for duplicate rule IDs
                if formalized_rule["id"] in rule_ids:
                    # Append a unique suffix
                    base_id = formalized_rule["id"]
                    counter = 1
                    while f"{base_id}_{counter}" in rule_ids:
                        counter += 1
                    formalized_rule["id"] = f"{base_id}_{counter}"
                
                rule_ids.add(formalized_rule["id"])
                formalized_rules.append(formalized_rule)
        
        # Sort rules by ID for consistent ordering
        formalized_rules.sort(key=lambda r: r["id"])
        
        return formalized_rules
    
    def _formalize_rule(self, rule: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Formalize a single rule into executable format.
        
        Args:
            rule: The extracted rule
            
        Returns:
            Formalized rule or None if rule couldn't be formalized
        """
        try:
            # Process field names to match DataFrame column names
            processed_fields = []
            for field in rule.get("fields", []):
                # Convert field name to likely column name (e.g., "Unique ID" -> "Unique_ID")
                processed_field = field.replace(" ", "_")
                processed_fields.append(processed_field)
            
            # Start with basic rule metadata
            formalized = {
                "id": f"rule_{len(processed_fields)}_{('_'.join(processed_fields) if processed_fields else 'general')}",
                "name": f"Validation for {', '.join(rule.get('fields', ['unknown']))}",
                "description": rule.get("requirement", ""),
                "source_section": rule.get("section", ""),
                "fields": processed_fields,
                "type": rule.get("rule_type", "validation")
            }
            
            # If we have the API key, use LLM to generate validation code
            if self.api_key:
                prompt = f"""
                Task: Generate a robust Python validation function for this financial regulatory rule.
                
                Rule details:
                - Fields: {', '.join(processed_fields)}
                - Type: {rule.get('rule_type', 'validation')}
                - Requirement: {rule.get('requirement', '')}
                - Condition: {rule.get('condition', '')}
                
                Create a Python function that validates a pandas Series ('row') against this rule.
                The function should:
                1. Take a pandas Series ('row') as input
                2. Return True if the row passes validation, False otherwise, but make sure to always return a bool value
                3. Include proper error handling for missing fields, different data types, etc.
                4. Use safe accessors (row.get() instead of row[]) to avoid KeyErrors
                5. DO NOT use pd.dtype() or other invalid pandas functions
                6. DO NOT use Series methods like .diff() on scalar values
                
                Example of a robust function:
                ```python
                def validate_rule(row):
                    # Check if Identifier_Type field exists and has valid values
                    if 'Identifier_Type' not in row:
                        return False
                        
                    if pd.isna(row['Identifier_Type']):
                        return False
                        
                    value = row['Identifier_Type']
                    if not isinstance(value, str):
                        return False
                        
                    if value not in ["CUSIP", "ISIN", "SEDOL", "INTERNAL"]:
                        return False
                        
                    return True
                ```
                
                Provide ONLY the function code, no explanation.
                """
                
                response = self._query_model(prompt)
                
                # Extract Python code from the response
                code_match = re.search(r'```python\s*(.+?)\s*```', response, re.DOTALL)
                if code_match:
                    validation_code = code_match.group(1).strip()
                else:
                    # If no code block markers, try to extract the function directly
                    code_match = re.search(r'def validate_rule\(.+?\):.+?return', response, re.DOTALL)
                    if code_match:
                        validation_code = code_match.group(0) + " True"
                    else:
                        validation_code = None
                
                if validation_code:
                    # Check for and fix common issues in the generated code
                    
                    # Fix: pd.dtype() is not a valid function
                    if "pd.dtype(" in validation_code:
                        validation_code = validation_code.replace("pd.dtype(", "isinstance(")
                        validation_code = validation_code.replace(") != 'object'", ", str)")
                    
                    # Fix: Attempting to use .diff() on a scalar value
                    if ".diff()" in validation_code:
                        validation_code = validation_code.replace("pd.notna(row['Unique_ID'].diff())", "False")
                    
                    # Check if the code looks valid
                    try:
                        compile(validation_code, "<string>", "exec")
                        formalized["validation_code"] = validation_code
                    except SyntaxError as e:
                        print(f"Syntax error in generated code: {e}")
                        # Fall back to basic validator
                        formalized["validation_code"] = self._generate_basic_validator(rule)
                else:
                    formalized["validation_code"] = self._generate_basic_validator(rule)
            else:
                formalized["validation_code"] = self._generate_basic_validator(rule)
            
            return formalized
        
        except Exception as e:
            print(f"Error formalizing rule: {e}")
            return None
    
    def _generate_basic_validator(self, rule: Dict[str, Any]) -> str:
        """
        Generate basic validation code for a rule.
        
        Args:
            rule: The rule to generate validation code for
            
        Returns:
            Python code for validation
        """
        fields = rule.get("fields", [])
        rule_type = rule.get("rule_type", "validation")
        
        if not fields:
            # Generic validator that always passes
            return "def validate_rule(row):\n    # No specific fields to validate\n    return True"
        
        # Convert rule field names to actual DataFrame column names (replace spaces with underscores)
        processed_fields = []
        for field in fields:
            # Convert field name to likely column name (e.g., "Unique ID" -> "Unique_ID")
            processed_field = field.replace(" ", "_")
            processed_fields.append(processed_field)
        
        # Generate different validation logic based on rule type
        if rule_type == "required_field":
            field_checks = []
            for field in processed_fields:
                field_checks.append(f"    if pd.isna(row.get('{field}', None)):\n        return False")
            
            return "def validate_rule(row):\n" + "\n".join(field_checks) + "\n    return True"
        
        elif rule_type == "format_validation":
            # Basic format validation for common fields
            if "Identifier_Type" in processed_fields:
                return """def validate_rule(row):
    if 'Identifier_Type' in row:
        if pd.isna(row['Identifier_Type']):
            return False
        if row['Identifier_Type'] not in ["CUSIP", "ISIN", "SEDOL", "INTERNAL"]:
            return False
    return True"""
            
            elif "Private_Placement" in processed_fields:
                return """def validate_rule(row):
    if 'Private_Placement' in row:
        if pd.isna(row['Private_Placement']):
            return False
        if row['Private_Placement'] not in ["Y", "N"]:
            return False
    return True"""
            
            elif "Accounting_Intent" in processed_fields:
                return """def validate_rule(row):
    if 'Accounting_Intent' in row:
        if pd.isna(row['Accounting_Intent']):
            return False
        if row['Accounting_Intent'] not in ["AFS", "HTM", "EQ"]:
            return False
    return True"""
            
            else:
                # Generic presence check for other fields
                field_checks = []
                for field in processed_fields:
                    field_checks.append(f"    if '{field}' in row and pd.isna(row['{field}']):\n        return False")
                
                return "def validate_rule(row):\n" + "\n".join(field_checks) + "\n    return True"
        
        elif rule_type == "allowed_values":
            field_checks = []
            
            for field in processed_fields:
                if field == "Identifier_Type":
                    field_checks.append("""    if 'Identifier_Type' in row:
        if pd.isna(row['Identifier_Type']):
            return False
        if row['Identifier_Type'] not in ["CUSIP", "ISIN", "SEDOL", "INTERNAL"]:
            return False""")
                
                elif field == "Private_Placement":
                    field_checks.append("""    if 'Private_Placement' in row:
        if pd.isna(row['Private_Placement']):
            return False
        if row['Private_Placement'] not in ["Y", "N"]:
            return False""")
                
                elif field == "Accounting_Intent":
                    field_checks.append("""    if 'Accounting_Intent' in row:
        if pd.isna(row['Accounting_Intent']):
            return False
        if row['Accounting_Intent'] not in ["AFS", "HTM", "EQ"]:
            return False""")
                
                else:
                    field_checks.append(f"    if '{field}' in row and pd.isna(row['{field}']):\n        return False")
            
            return "def validate_rule(row):\n" + "\n".join(field_checks) + "\n    return True"
        
        elif rule_type == "cross_field_validation":
            # Generic cross-field validation
            return """def validate_rule(row):
    # Cross-field validation would be implemented based on specific rules
    # For now, just check that required fields exist
    for field in [{}]:
        if field in row and pd.isna(row[field]):
            return False
    return True""".format(", ".join([f"'{field}'" for field in processed_fields]))
        
        else:
            # Generic validator that checks field presence
            field_checks = []
            for field in processed_fields:
                field_checks.append(f"    if '{field}' in row and pd.isna(row['{field}']):\n        return False")
            
            return "def validate_rule(row):\n" + "\n".join(field_checks) + "\n    return True"
    
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
                "temperature": 0.1, 
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
            r'\[\s*{.*}\s*\]',  
            r'{.*}',            
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