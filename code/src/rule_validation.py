import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Union, Tuple
import json
import os
from datetime import datetime

class RuleValidator:
    """Apply validation rules to securities data."""
    
    def __init__(self, rules: Optional[List[Dict[str, Any]]] = None):
        """
        Initialize the rule validator.
        
        Args:
            rules: Optional list of validation rules
        """
        self.rules = rules or []
    
    def load_rules(self, rules_file: str) -> None:
        """
        Load rules from a JSON file.
        
        Args:
            rules_file: Path to the rules file
        """
        with open(rules_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Extract rules from the loaded data
        if isinstance(data, dict) and "rules" in data:
            self.rules = data["rules"]
        elif isinstance(data, list):
            self.rules = data
        else:
            raise ValueError(f"Invalid rule file format: {rules_file}")
        
        print(f"Loaded {len(self.rules)} rules from {rules_file}")
    
    def add_rule(self, rule: Dict[str, Any]) -> None:
        """
        Add a validation rule.
        
        Args:
            rule: The rule to add
        """
        self.rules.append(rule)
    
    def validate(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Validate the data against all rules.
        
        Args:
            data: DataFrame containing securities data
            
        Returns:
            DataFrame with validation results
        """
        if not self.rules:
            print("No rules to apply")
            return data
        
        # Create a copy of the data for results
        result = data.copy()
        
        # Add validation columns
        result['validation_passed'] = True
        result['failed_rules'] = [[] for _ in range(len(result))]
        result['validation_messages'] = [[] for _ in range(len(result))]
        
        # Apply each rule
        for rule in self.rules:
            rule_id = rule.get('id', 'unknown')
            rule_name = rule.get('name', 'Unknown Rule')
            validation_code = rule.get('validation_code')
            
            if not validation_code:
                continue
            
            print(f"Applying rule: {rule_id} - {rule_name}")
            
            try:
                # Create a namespace for the function
                namespace = {'pd': pd, 'np': np}
                
                # Execute the validation code to define the function
                exec(validation_code, namespace)
                
                # Get the validate_rule function
                validate_rule = namespace['validate_rule']
                
                # Apply the function to each row
                rule_results = data.apply(validate_rule, axis=1)
                
                # Update the validation results
                result['validation_passed'] = result['validation_passed'] & rule_results
                
                # Track failed rules and messages
                for idx, passed in enumerate(rule_results):
                    if not passed:
                        result.at[idx, 'failed_rules'].append(rule_id)
                        result.at[idx, 'validation_messages'].append(f"Failed rule: {rule_name}")
            
            except Exception as e:
                print(f"Error applying rule {rule_id}: {e}")
        
        return result
    
    def validate_rule(self, rule_id: str, data: pd.DataFrame) -> pd.Series:
        """
        Validate the data against a specific rule.
        
        Args:
            rule_id: ID of the rule to apply
            data: DataFrame containing securities data
            
        Returns:
            Series of boolean values indicating which rows passed validation
        """
        # Find the rule
        rule = next((r for r in self.rules if r.get('id') == rule_id), None)
        
        if not rule:
            print(f"Rule {rule_id} not found")
            return pd.Series([True] * len(data), index=data.index)
        
        validation_code = rule.get('validation_code')
        
        if not validation_code:
            print(f"Rule {rule_id} has no validation code")
            return pd.Series([True] * len(data), index=data.index)
        
        try:
            # Create a namespace for the function
            namespace = {'pd': pd, 'np': np}
            
            # Execute the validation code to define the function
            exec(validation_code, namespace)
            
            # Get the validate_rule function
            validate_rule = namespace['validate_rule']
            
            # Apply the function to each row
            return data.apply(validate_rule, axis=1)
        
        except Exception as e:
            print(f"Error applying rule {rule_id}: {e}")
            return pd.Series([True] * len(data), index=data.index)
    
    def validate_field(self, field_name: str, data: pd.DataFrame) -> pd.DataFrame:
        """
        Validate a specific field against all relevant rules.
        
        Args:
            field_name: Name of the field to validate
            data: DataFrame containing securities data
            
        Returns:
            DataFrame with validation results for the field
        """
        # Find rules applicable to this field
        field_rules = [rule for rule in self.rules if field_name in rule.get('fields', [])]
        
        if not field_rules:
            print(f"No rules found for field {field_name}")
            # Return a copy of data with validation columns added
            result = data.copy()
            result['validation_passed'] = True
            result['failed_rules'] = [[] for _ in range(len(result))]
            result['validation_messages'] = [[] for _ in range(len(result))]
            return result
        
        # Create a temporary validator with just the field rules
        field_validator = RuleValidator(field_rules)
        
        # Validate the data
        return field_validator.validate(data)
    
    def generate_validation_report(self, validation_results: pd.DataFrame) -> Dict[str, Any]:
        """
        Generate a validation report from validation results.
        
        Args:
            validation_results: DataFrame with validation results
            
        Returns:
            Dictionary with validation report
        """
        # Count passed and failed records
        passed_count = validation_results['validation_passed'].sum()
        failed_count = len(validation_results) - passed_count
        
        # Get failed rules frequency
        all_failed_rules = []
        for failed_rules in validation_results['failed_rules']:
            all_failed_rules.extend(failed_rules)
        
        rule_frequency = pd.Series(all_failed_rules).value_counts().to_dict()
        
        # Create report
        report = {
            'timestamp': datetime.now().isoformat(),
            'total_records': len(validation_results),
            'passed_records': int(passed_count),
            'failed_records': int(failed_count),
            'pass_rate': float(passed_count / len(validation_results)) if len(validation_results) > 0 else 1.0,
            'rule_frequency': rule_frequency,
            'failed_record_ids': validation_results[~validation_results['validation_passed']]['Unique_ID'].tolist()
        }
        
        return report
    
    def save_validation_report(self, report: Dict[str, Any], output_file: str) -> None:
        """
        Save validation report to a JSON file.
        
        Args:
            report: Validation report dictionary
            output_file: Path to save the report
        """
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2)
        
        print(f"Saved validation report to {output_file}")
    
    def get_failed_records(self, validation_results: pd.DataFrame) -> pd.DataFrame:
        """
        Get records that failed validation.
        
        Args:
            validation_results: DataFrame with validation results
            
        Returns:
            DataFrame with failed records
        """
        return validation_results[~validation_results['validation_passed']]
    
    def get_rule_failures(self, validation_results: pd.DataFrame, rule_id: str) -> pd.DataFrame:
        """
        Get records that failed a specific rule.
        
        Args:
            validation_results: DataFrame with validation results
            rule_id: ID of the rule to check
            
        Returns:
            DataFrame with records that failed the rule
        """
        # Find records where failed_rules contains the rule_id
        mask = validation_results['failed_rules'].apply(lambda x: rule_id in x)
        return validation_results[mask]