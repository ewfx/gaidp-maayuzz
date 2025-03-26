import numpy as np
import pandas as pd
from typing import Dict, List, Any, Tuple, Optional
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
import matplotlib.pyplot as plt
import seaborn as sns

class AnomalyDetector:
    """Detect anomalies in securities data using unsupervised ML."""
    
    def __init__(self, contamination: float = 0.05):
        """
        Initialize the anomaly detector.
        
        Args:
            contamination: Expected proportion of anomalies in the data
        """
        self.contamination = contamination
        self.model = None
        self.preprocessor = None
        self.feature_names = None
    
    def fit(self, data: pd.DataFrame) -> None:
        """
        Fit the anomaly detection model to the data.
        
        Args:
            data: DataFrame containing securities data
        """
        # Identify numeric and categorical columns
        numeric_features = data.select_dtypes(include=['int64', 'float64']).columns.tolist()
        categorical_features = data.select_dtypes(include=['object', 'category']).columns.tolist()
        
        # Store feature names for later reference
        self.feature_names = numeric_features + categorical_features
        
        # Create preprocessing pipeline
        numeric_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())
        ])
        
        categorical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('onehot', OneHotEncoder(handle_unknown='ignore'))
        ])
        
        self.preprocessor = ColumnTransformer(transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ])
        
        # Preprocess the data
        X_processed = self.preprocessor.fit_transform(data)
        
        # Fit an Isolation Forest model for anomaly detection
        self.model = IsolationForest(
            contamination=self.contamination,
            random_state=42,
            n_jobs=-1
        )
        
        self.model.fit(X_processed)
        
        print(f"Fitted anomaly detection model on {len(data)} records with {len(self.feature_names)} features")
    
    def detect_anomalies(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Detect anomalies in the data.
        
        Args:
            data: DataFrame containing securities data
            
        Returns:
            DataFrame with original data and anomaly scores
        """
        if self.model is None or self.preprocessor is None:
            raise ValueError("Model not fitted. Call fit() first.")
        
        
        missing_cols = set(self.feature_names) - set(data.columns)
        if missing_cols:
            raise ValueError(f"Missing columns in input data: {missing_cols}")
        
        # Preprocess the data
        X_processed = self.preprocessor.transform(data[self.feature_names])
        
        # Predict anomalies (-1 for anomalies, 1 for normal)
        anomaly_labels = self.model.predict(X_processed)
        
        # Get anomaly scores (lower score = more anomalous)
        anomaly_scores = self.model.decision_function(X_processed)
        
        # Add results to the data
        result = data.copy()
        result['anomaly'] = anomaly_labels == -1
        result['anomaly_score'] = anomaly_scores
        
        return result
    
    def get_feature_importance(self) -> pd.DataFrame:
        """
        Get feature importance for anomaly detection.
        
        Returns:
            DataFrame with feature importance scores
        """
        if self.model is None:
            raise ValueError("Model not fitted. Call fit() first.")
        
        
        n_features = self.model.n_features_in_
        
        # Initialize importance array
        importance_values = np.zeros(n_features)
        
        # Get feature importances from each tree in the forest
        if hasattr(self.model, 'estimators_'):
            n_estimators = len(self.model.estimators_)
            
            # Sum feature importances across all trees
            for tree in self.model.estimators_:
                if hasattr(tree, 'tree_'):
                    # Get the feature used at each node
                    feature = tree.tree_.feature
                    # Count the number of samples each node is responsible for
                    n_samples = tree.tree_.n_node_samples
                    
                    # Compute the weighted sum of feature importance
                    for i in range(len(feature)):
                        if feature[i] != -1:  # -1 indicates a leaf node
                            importance_values[feature[i]] += n_samples[i]
            
            # Normalize importances
            if np.sum(importance_values) > 0:
                importance_values = importance_values / np.sum(importance_values)
        
        # Create feature names
        if hasattr(self.preprocessor, 'get_feature_names_out'):
            try:
                feature_names = self.preprocessor.get_feature_names_out()
            except:
                feature_names = [f"feature_{i}" for i in range(n_features)]
        else:
            feature_names = [f"feature_{i}" for i in range(n_features)]
        
        # Ensure the right number of feature names
        if len(feature_names) != len(importance_values):
            feature_names = [f"feature_{i}" for i in range(len(importance_values))]
        
        # Create a DataFrame of feature importances
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': importance_values
        })
        
        # Sort by importance
        importance_df = importance_df.sort_values('importance', ascending=False)
        
        return importance_df
    
    def plot_anomaly_distribution(self, result: pd.DataFrame, save_path: Optional[str] = None) -> None:
        """
        Plot the distribution of anomaly scores.
        
        Args:
            result: DataFrame with anomaly detection results
            save_path: Optional path to save the plot
        """
        plt.figure(figsize=(10, 6))
        sns.histplot(data=result, x='anomaly_score', hue='anomaly', kde=True)
        plt.title('Distribution of Anomaly Scores')
        plt.xlabel('Anomaly Score')
        plt.ylabel('Count')
        
        if save_path:
            plt.savefig(save_path)
            print(f"Saved anomaly distribution plot to {save_path}")
        else:
            plt.show()
    
    def identify_anomaly_features(self, data: pd.DataFrame, anomaly_indices: List[int]) -> Dict[int, List[dict]]:
        """
        Identify which features are most anomalous for each anomaly.
        
        Args:
            data: Original DataFrame
            anomaly_indices: Indices of anomalous records
            
        Returns:
            Dictionary mapping anomaly indices to lists of anomalous features
        """
        if self.model is None or self.preprocessor is None:
            raise ValueError("Model not fitted. Call fit() first.")
        
        # Get numeric and categorical features
        numeric_features = data.select_dtypes(include=['int64', 'float64']).columns.tolist()
        categorical_features = data.select_dtypes(include=['object', 'category']).columns.tolist()
        
        # Calculate statistics for numeric features
        numeric_stats = {}
        for feature in numeric_features:
            numeric_stats[feature] = {
                'mean': data[feature].mean(),
                'std': data[feature].std(),
                'q1': data[feature].quantile(0.25),
                'q3': data[feature].quantile(0.75),
                'iqr': data[feature].quantile(0.75) - data[feature].quantile(0.25),
                'min': data[feature].min(),
                'max': data[feature].max()
            }
        
        # Calculate frequencies for categorical features
        categorical_stats = {}
        for feature in categorical_features:
            value_counts = data[feature].value_counts(normalize=True)
            categorical_stats[feature] = {
                'frequencies': value_counts,
                'most_common': value_counts.idxmax(),
                'least_common': value_counts.idxmin(),
                'unique_values': data[feature].nunique()
            }
        
        # Identify anomalous features for each anomaly
        anomaly_features = {}
        
        for idx in anomaly_indices:
            # Features with values outside normal range are considered anomalous
            anomalous_features = []
            
            # Get the record at this index
            record = data.iloc[idx]
            
            # Check numeric features
            for feature in numeric_features:
                value = float(record[feature])  # Convert to float to avoid Series
                feature_stats = numeric_stats[feature]
                
                # Check different criteria for anomalousness
                is_anomalous = False
                reason = ""
                z_score = None
                
                # Calculate z-score if std > 0
                if feature_stats['std'] > 0:
                    z_score = (value - feature_stats['mean']) / feature_stats['std']
                    # Lower threshold to 1.2 to catch more anomalies
                    if abs(z_score) > 1.2:
                        is_anomalous = True
                        reason = f"z-score: {z_score:.2f}"
                
                # Check IQR method (more robust to outliers)
                if feature_stats['iqr'] > 0:
                    lower_bound = feature_stats['q1'] - 1.5 * feature_stats['iqr']
                    upper_bound = feature_stats['q3'] + 1.5 * feature_stats['iqr']
                    
                    if value < lower_bound or value > upper_bound:
                        is_anomalous = True
                        if value < lower_bound:
                            deviation = (value - lower_bound) / feature_stats['iqr']
                            reason += f" Below Q1: {deviation:.2f} IQR"
                        else:
                            deviation = (value - upper_bound) / feature_stats['iqr']
                            reason += f" Above Q3: {deviation:.2f} IQR"
                
                # Check extreme values
                if value == feature_stats['min'] or value == feature_stats['max']:
                    is_anomalous = True
                    if value == feature_stats['min']:
                        reason += " Minimum value"
                    else:
                        reason += " Maximum value"
                
                if is_anomalous:
                    anomalous_features.append({
                        'feature': feature,
                        'value': value,
                        'z_score': z_score if z_score is not None else 0,
                        'reason': reason.strip(),
                        'type': 'numeric'
                    })
            
            # Check categorical features
            for feature in categorical_features:
                value = record[feature]
                feature_stats = categorical_stats[feature]
                
                is_anomalous = False
                reason = ""
                
                # Rare value (frequency < 5%)
                frequency = feature_stats['frequencies'].get(value, 0)
                if frequency < 0.05:
                    is_anomalous = True
                    reason = f"Rare value: {frequency:.1%} frequency"
                
                # Least common value
                if value == feature_stats['least_common']:
                    is_anomalous = True
                    reason += " Least common value"
                
                if is_anomalous:
                    anomalous_features.append({
                        'feature': feature,
                        'value': value,
                        'frequency': frequency,
                        'reason': reason.strip(),
                        'type': 'categorical'
                    })
            
            # Sort anomalous features - numeric first by z-score, then categorical by frequency
            numeric_features_list = [f for f in anomalous_features if f['type'] == 'numeric']
            categorical_features_list = [f for f in anomalous_features if f['type'] == 'categorical']
            
            numeric_features_list.sort(key=lambda x: abs(x['z_score']), reverse=True)
            categorical_features_list.sort(key=lambda x: x.get('frequency', 1), reverse=False)
            
            # Combine sorted features
            sorted_features = numeric_features_list + categorical_features_list
            
            # Store the anomalous features
            anomaly_features[idx] = sorted_features
        
        return anomaly_features

