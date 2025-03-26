import os
import re
import json
import requests
from typing import Dict, List, Optional, Tuple, Any
import pandas as pd
from dotenv import load_dotenv

# Load environment variables (for API key)
load_dotenv()

class HuggingFaceProcessor:
    """
    Process regulatory documents using Hugging Face Inference API.
    """
    
    def __init__(self, model_name: str = "mistralai/Mistral-7B-Instruct-v0.2"):
        """
        Initialize the document processor.
        
        Args:
            model_name: The name of the Hugging Face model to use via API
        """
        self.model_name = model_name
        self.api_key = os.getenv("HUGGINGFACE_API_KEY")
        if not self.api_key:
            print("Warning: HUGGINGFACE_API_KEY not found in environment variables")
            print("You will need to set this key to use the Hugging Face Inference API")
        
        self.api_url = f"https://api-inference.huggingface.co/models/{model_name}"
        self.headers = {"Authorization": f"Bearer {self.api_key}"}
        
    def process_text(self, text: str) -> Dict[str, Any]:
        """
        Process extracted text from a regulatory document.
        
        Args:
            text: The text content of the document
            
        Returns:
            Structured representation of the document content
        """
        # Split text into sections based on headings
        sections = self._split_into_sections(text)
        
        # Identify sections containing rules
        rule_sections = self._identify_rule_sections(sections)
        
        # Extract tables from text
        tables = self._extract_tables(text)
        
        return {
            "sections": sections,
            "rule_sections": rule_sections,
            "tables": tables,
            "full_text": text
        }
    
    def _split_into_sections(self, text: str) -> Dict[str, str]:
        """
        Split document text into logical sections based on headings.
        
        Args:
            text: The document text
            
        Returns:
            Dictionary mapping section titles to section content
        """
        # Simple regex-based section splitting
        section_pattern = re.compile(r'^([A-Z][A-Za-z\s\-]+)$', re.MULTILINE)
        
        # Find all potential section headings
        matches = list(section_pattern.finditer(text))
        
        sections = {}
        
        # Extract each section's content
        for i, match in enumerate(matches):
            section_title = match.group(1).strip()
            start_pos = match.end()
            
            # If this isn't the last section, set end position to start of next section
            if i < len(matches) - 1:
                end_pos = matches[i + 1].start()
            else:
                end_pos = len(text)
            
            section_content = text[start_pos:end_pos].strip()
            sections[section_title] = section_content
        
        return sections
    
    def _identify_rule_sections(self, sections: Dict[str, str]) -> Dict[str, Any]:
        """
        Identify sections that contain regulatory rules using HF API.
        
        Args:
            sections: Dictionary of section titles to content
            
        Returns:
            Dictionary containing only sections with rules
        """
        rule_sections = {}
        
        # Fallback to keyword-based approach if API is not available
        if not self.api_key:
            rule_keywords = [
                "must", "required", "shall", "should", "may not", 
                "report", "exclude", "include", "allowable", "permitted"
            ]
            
            for title, content in sections.items():
                if any(keyword in content.lower() for keyword in rule_keywords):
                    rule_sections[title] = content
            
            return rule_sections
        
        # Use HF Inference API for better identification
        try:
            for title, content in sections.items():
                if len(content) < 10:  # Skip very short sections
                    continue
                    
                # Prepare prompt for model
                prompt = f"""
                Task: Determine if the following text contains financial regulatory rules or requirements.
                
                Text: 
                {content[:500]}  # Limit content length to prevent token limits
                
                Does this text contain specific rules, requirements, or validation criteria that financial institutions must follow? Respond with YES or NO.
                """
                
                # Query the model
                response = self._query_model(prompt)
                
                # Check if response indicates rules
                if "YES" in response.upper() or any(keyword in content.lower() for keyword in ["must", "required", "shall"]):
                    rule_sections[title] = content
                    
        except Exception as e:
            print(f"Error using Hugging Face API for rule section identification: {e}")
            # Fall back to keyword-based approach
            for title, content in sections.items():
                if any(keyword in content.lower() for keyword in ["must", "required", "shall", "should", "report"]):
                    rule_sections[title] = content
        
        return rule_sections
    
    def _extract_tables(self, text: str) -> List[pd.DataFrame]:
        """
        Extract tables from document text.
        
        Args:
            text: The document text
            
        Returns:
            List of extracted tables as pandas DataFrames
        """
        
        # For now, return an empty list as tabular extraction requires
        # more complex logic or OCR capabilities
        return []
    
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
                "max_new_tokens": 256,
                "temperature": 0.1,  
                "top_p": 0.95
            }
        }
        
        response = requests.post(self.api_url, headers=self.headers, json=payload)
        
        if response.status_code != 200:
            raise Exception(f"API request failed with status code {response.status_code}: {response.text}")
        
        return response.json()[0]["generated_text"].replace(prompt, "").strip()
    
    def extract_rules_from_text(self, text: str) -> List[Dict[str, Any]]:
        """
        Extract rules from document text using HF API.
        
        Args:
            text: The document text
            
        Returns:
            List of extracted rules with their details
        """
        # Process the document to get structured content
        processed_doc = self.process_text(text)
        
        # If API key is not available
        if not self.api_key:
            return self._extract_rules_simple(processed_doc)
        
        # Extract rules using HF API
        rules = []
        
        for section_title, section_content in processed_doc["rule_sections"].items():
            # Skip very long sections to avoid token limits
            if len(section_content) > 1500:
                # Split into smaller chunks for processing
                paragraphs = section_content.split('\n\n')
                for i, paragraph in enumerate(paragraphs):
                    if len(paragraph) < 50:  # Skip very short paragraphs
                        continue
                        
                    # Process each paragraph
                    rule = self._extract_rule_from_paragraph(section_title, paragraph)
                    if rule:
                        rules.append(rule)
            else:
                # Process entire section
                prompt = f"""
                Task: Extract structured financial regulatory rules from the following text.
                
                Text: 
                {section_content}
                
                For each rule, identify:
                1. The specific fields it applies to
                2. The rule type (format validation, value validation, cross-field validation, etc.)
                3. The exact requirement or condition
                
                Format your response as a JSON list, with each rule having 'fields', 'type', and 'requirement' keys.
                """
                
                try:
                    response = self._query_model(prompt)
                    
                    # Try to extract JSON from response
                    json_str = self._extract_json_from_text(response)
                    if json_str:
                        extracted_rules = json.loads(json_str)
                        if isinstance(extracted_rules, list):
                            for rule in extracted_rules:
                                rule["section"] = section_title
                                rule["text"] = section_content
                                rules.append(rule)
                        elif isinstance(extracted_rules, dict):
                            extracted_rules["section"] = section_title
                            extracted_rules["text"] = section_content
                            rules.append(extracted_rules)
                except Exception as e:
                    print(f"Error extracting rules using HF API: {e}")
                    # Fallback to extracting a rule directly from the section
                    rule = self._extract_rule_from_paragraph(section_title, section_content)
                    if rule:
                        rules.append(rule)
        
        return rules
    
    def _extract_rule_from_paragraph(self, section_title: str, paragraph: str) -> Optional[Dict[str, Any]]:
        """
        Extract a single rule from a paragraph of text.
        
        Args:
            section_title: The title of the section containing the paragraph
            paragraph: The text paragraph
            
        Returns:
            A dictionary representing the extracted rule, or None if no rule was found
        """
        if len(paragraph.strip()) < 10:
            return None
            
        # Check if paragraph contains rule-like language
        if any(keyword in paragraph.lower() for keyword in ["must", "required", "shall", "should", "report"]):
            # Use HF API to extract fields and rule type if available
            if self.api_key:
                prompt = f"""
                Task: Extract regulatory rule details from this text:
                
                "{paragraph}"
                
                Identify:
                1. Which data fields this rule applies to
                2. What type of rule this is (format validation, value validation, cross-field validation, etc.)
                3. The specific requirement
                
                Format your response as a JSON object with 'fields', 'type', and 'requirement' keys.
                """
                
                try:
                    response = self._query_model(prompt)
                    json_str = self._extract_json_from_text(response)
                    if json_str:
                        rule_details = json.loads(json_str)
                        rule_details["section"] = section_title
                        rule_details["text"] = paragraph
                        return rule_details
                except Exception as e:
                    print(f"Error extracting rule details: {e}")
            
            # Fallback to simple extraction
            fields = self._extract_fields(paragraph)
            
            return {
                "section": section_title,
                "text": paragraph,
                "fields": fields,
                "type": "validation",
                "requirement": paragraph
            }
        
        return None
    
    def _extract_rules_simple(self, processed_doc: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Simple rule extraction without using the HF API.
        
        Args:
            processed_doc: The processed document
            
        Returns:
            List of extracted rules
        """
        rules = []
        
        for section_title, section_content in processed_doc["rule_sections"].items():
            # Split content into paragraphs
            paragraphs = section_content.split('\n\n')
            
            for paragraph in paragraphs:
                if any(keyword in paragraph.lower() for keyword in ["must", "required", "shall", "should", "report"]):
                    fields = self._extract_fields(paragraph)
                    
                    rule = {
                        "section": section_title,
                        "text": paragraph,
                        "fields": fields,
                        "type": "validation",
                        "requirement": paragraph
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
        # This is a simplified field extraction
        common_fields = [
            "Unique ID", "Identifier Type", "Identifier Value", 
            "Private Placement", "Security Description", 
            "Amortized Cost", "Market Value", "Current Face Value",
            "Original Face Value", "Allowance for Credit Losses",
            "Writeoffs", "Accounting Intent", "Price", 
            "Pricing Date", "Book Yield", "Purchase Date", "Currency"
        ]
        
        mentioned_fields = []
        
        for field in common_fields:
            if field in text:
                mentioned_fields.append(field)
        
        return mentioned_fields
    
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
            r'\[\s*{.*}\s*\]',  # JSON array pattern
            r'{.*}',            # JSON object pattern
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


# Function to load document text from file
def load_document(file_path: str) -> str:
    """
    Load document text from a file.
    
    Args:
        file_path: Path to the document file
        
    Returns:
        Text content of the document
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        text = f.read()
    
    return text



