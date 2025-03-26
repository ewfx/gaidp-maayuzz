import os
import json
from typing import Dict, List, Any, Optional
import pandas as pd
from datetime import datetime

class RuleStorage:
    """Store and retrieve validation rules."""
    
    def __init__(self, storage_dir: str = "rules"):
        """
        Initialize the rule storage.
        
        Args:
            storage_dir: Directory to store rule files
        """
        self.storage_dir = storage_dir
        
        # Create storage directory if it doesn't exist
        os.makedirs(storage_dir, exist_ok=True)
    
    def save_rules(self, rules: List[Dict[str, Any]], source_name: str) -> str:
        """
        Save rules to storage.
        
        Args:
            rules: List of formalized rules
            source_name: Name of the regulatory source
            
        Returns:
            Path to the saved rules file
        """
        # Create a timestamp for the filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Clean source name for filename
        clean_source_name = "".join(c if c.isalnum() else "_" for c in source_name)
        
        # Create filename
        filename = f"{clean_source_name}_{timestamp}.json"
        filepath = os.path.join(self.storage_dir, filename)
        
        # Add metadata to the rules
        rules_with_metadata = {
            "metadata": {
                "source": source_name,
                "extraction_date": datetime.now().isoformat(),
                "rule_count": len(rules)
            },
            "rules": rules
        }
        
        # Write rules to file
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(rules_with_metadata, f, indent=2)
        
        print(f"Saved {len(rules)} rules to {filepath}")
        return filepath
    
    def load_rules(self, filepath: str) -> List[Dict[str, Any]]:
        """
        Load rules from a file.
        
        Args:
            filepath: Path to the rules file
            
        Returns:
            List of formalized rules
        """
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Extract rules from the loaded data
        if isinstance(data, dict) and "rules" in data:
            return data["rules"]
        elif isinstance(data, list):
            return data
        else:
            raise ValueError(f"Invalid rule file format: {filepath}")
    
    def load_latest_rules(self, source_name: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Load the most recent rules file.
        
        Args:
            source_name: Optional name of the regulatory source to filter by
            
        Returns:
            List of formalized rules from the most recent file
        """
        # List all rule files
        rule_files = os.listdir(self.storage_dir)
        
        # Filter by source name if provided
        if source_name:
            clean_source_name = "".join(c if c.isalnum() else "_" for c in source_name)
            rule_files = [f for f in rule_files if f.startswith(clean_source_name)]
        
        if not rule_files:
            return []
        
        # Sort files by creation time (most recent first)
        rule_files.sort(key=lambda f: os.path.getctime(os.path.join(self.storage_dir, f)), reverse=True)
        
        # Load the most recent file
        latest_file = rule_files[0]
        return self.load_rules(os.path.join(self.storage_dir, latest_file))
    
    def get_rules_by_field(self, field_name: str) -> List[Dict[str, Any]]:
        """
        Get all rules applicable to a specific field.
        
        Args:
            field_name: The field name to find rules for
            
        Returns:
            List of rules applicable to the field
        """
        # Load all rule files
        all_rules = []
        for filename in os.listdir(self.storage_dir):
            if filename.endswith('.json'):
                try:
                    rules = self.load_rules(os.path.join(self.storage_dir, filename))
                    all_rules.extend(rules)
                except:
                    continue
        
        # Filter rules by field
        return [rule for rule in all_rules if field_name in rule.get("fields", [])]
    
    def get_rules_by_type(self, rule_type: str) -> List[Dict[str, Any]]:
        """
        Get all rules of a specific type.
        
        Args:
            rule_type: The type of rules to find
            
        Returns:
            List of rules of the specified type
        """
        # Load all rule files
        all_rules = []
        for filename in os.listdir(self.storage_dir):
            if filename.endswith('.json'):
                try:
                    rules = self.load_rules(os.path.join(self.storage_dir, filename))
                    all_rules.extend(rules)
                except:
                    continue
        
        # Filter rules by type
        return [rule for rule in all_rules if rule.get("type", "") == rule_type]
    
    def execute_rule(self, rule: Dict[str, Any], data: pd.DataFrame) -> pd.Series:
        """
        Execute a rule against a DataFrame.
        
        Args:
            rule: The rule to execute
            data: The DataFrame to validate
            
        Returns:
            Series of boolean values indicating which rows passed validation
        """
        # Extract validation code
        validation_code = rule.get("validation_code")
        if not validation_code:
            return pd.Series([True] * len(data), index=data.index)
        
        # Create a namespace for the function
        namespace = {'pd': pd}
        
        # Execute the validation code to define the function
        exec(validation_code, namespace)
        
        # Get the validate_rule function
        validate_rule = namespace['validate_rule']
        
        # Apply the function to each row
        results = data.apply(validate_rule, axis=1)
        
        return results