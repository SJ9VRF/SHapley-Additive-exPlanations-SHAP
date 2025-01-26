
"""
SHAP (SHapley Additive exPlanations) implementation for machine learning model interpretation.
This module provides comprehensive functionality for explaining model predictions using SHAP values.
"""

import numpy as np
import pandas as pd
from typing import List, Optional, Union, Callable, Dict, Tuple
from sklearn.base import BaseEstimator
import itertools
from concurrent.futures import ThreadPoolExecutor
import warnings
from dataclasses import dataclass
import matplotlib.pyplot as plt
import seaborn as sns

@dataclass
class ShapResult:
    """Container for SHAP explanation results."""
    values: np.ndarray
    base_value: float
    feature_names: List[str]
    
class ShapError(Exception):
    """Base exception class for SHAP-related errors."""
    pass

class InvalidModelError(ShapError):
    """Raised when the provided model is invalid."""
    pass

class InvalidDataError(ShapError):
    """Raised when the provided data is invalid."""
    pass

class SHAPExplainer:
    """
    SHAP (SHapley Additive exPlanations) implementation for ML model interpretation.
    
    Attributes:
        model: The machine learning model to explain
        background_data: Reference dataset for calculating feature contributions
        feature_names: Names of the features (optional)
        feature_count: Number of features in the model
        background_value: Average prediction on background data
        random_state: Random state for reproducibility
    """
    
    def __init__(
        self,
        model: BaseEstimator,
        background_data: np.ndarray,
        feature_names: Optional[List[str]] = None,
        random_state: Optional[int] = None
    ):
        """
        Initialize SHAP explainer.
        
        Args:
            model: Trained ML model with predict method
            background_data: Representative background dataset
            feature_names: Optional list of feature names
            random_state: Random state for reproducibility
            
        Raises:
            InvalidModelError: If model doesn't have required methods
            InvalidDataError: If background data is invalid
        """
        self._validate_model(model)
        self._validate_data(background_data)
        
        self.model = model
        self.background_data = background_data
        self.feature_count = background_data.shape[1]
        self.random_state = random_state
        
        if random_state is not None:
            np.random.seed(random_state)
            
        if feature_names is None:
            self.feature_names = [f"feature_{i}" for i in range(self.feature_count)]
        else:
            if len(feature_names) != self.feature_count:
                raise InvalidDataError("Number of feature names doesn't match data dimensionality")
            self.feature_names = feature_names
            
        self.background_value = self._calculate_background_value()
        
    def _validate_model(self, model: BaseEstimator) -> None:
        """Validate that model has required methods."""
        required_methods = ['predict']
        missing_methods = [method for method in required_methods 
                         if not hasattr(model, method)]
        
        if missing_methods:
            raise InvalidModelError(
                f"Model missing required methods: {', '.join(missing_methods)}"
            )
            
    def _validate_data(self, data: np.ndarray) -> None:
        """Validate input data format and values."""
        if not isinstance(data, (np.ndarray, pd.DataFrame)):
            raise InvalidDataError("Data must be numpy array or pandas DataFrame")
            
        if len(data.shape) != 2:
            raise InvalidDataError("Data must be 2-dimensional")
            
        if data.size == 0:
            raise InvalidDataError("Data cannot be empty")
            
        if np.any(np.isnan(data)):
            raise InvalidDataError("Data contains NaN values")
            
    def _calculate_background_value(self) -> float:
        """Calculate average model prediction on background data."""
        return float(np.mean(self.model.predict(self.background_data)))
        
    def _get_contribution(
        self,
        instance: np.ndarray,
        feature_subset: List[int],
        background_instance: Optional[np.ndarray] = None
    ) -> float:
        """
        Calculate marginal contribution for a feature subset.
        
        Args:
            instance: Input instance
            feature_subset: List of feature indices to include
            background_instance: Optional specific background instance
            
        Returns:
            float: Contribution value
        """
        if background_instance is None:
            background_instance = self.background_data.mean(axis=0)
            
        mixed_instance = background_instance.copy()
        mixed_instance[feature_subset] = instance[feature_subset]
        
        return float(self.model.predict([mixed_instance])[0])
        
    def _get_weight(self, coalition_size: int, total_features: int) -> float:
        """
        Calculate coalition weight using Shapley formula.
        
        Args:
            coalition_size: Size of the current feature coalition
            total_features: Total number of features
            
        Returns:
            float: Shapley weight value
        """
        if coalition_size == 0 or coalition_size == total_features:
            return 0
        
        return float(
            factorial(coalition_size - 1) * factorial(total_features - coalition_size)
        ) / factorial(total_features)
        
    def _sample_combinations(
        self,
        features: List[int],
        sample_size: int
    ) -> List[List[int]]:
        """
        Sample random feature combinations for approximation.
        
        Args:
            features: List of feature indices
            sample_size: Number of combinations to sample
            
        Returns:
            List of feature combination lists
        """
        all_combinations = []
        for r in range(len(features) + 1):
            all_combinations.extend(itertools.combinations(features, r))
            
        if sample_size >= len(all_combinations):
            return list(all_combinations)
            
        return list(np.random.choice(
            all_combinations,
            size=sample_size,
            replace=False
        ))
        
    def explain_instance(
        self,
        instance: np.ndarray,
        sample_size: Optional[int] = None,
        background_instance: Optional[np.ndarray] = None
    ) -> ShapResult:
        """
        Generate SHAP values for a single instance.
        
        Args:
            instance: Input instance to explain
            sample_size: Optional number of permutations to sample
            background_instance: Optional specific background instance
            
        Returns:
            ShapResult containing SHAP values and metadata
            
        Raises:
            InvalidDataError: If instance format is invalid
        """
        self._validate_data(instance.reshape(1, -1))
        
        shap_values = np.zeros(self.feature_count)
        features = list(range(self.feature_count))
        
        # Get combinations
        combinations = (
            self._sample_combinations(features, sample_size)
            if sample_size
            else [list(c) for r in range(self.feature_count + 1)
                  for c in itertools.combinations(features, r)]
        )
        
        # Calculate Shapley values
        for subset in combinations:
            subset = list(subset)
            coalition_size = len(subset)
            
            # Get marginal contributions
            with_feature = self._get_contribution(
                instance,
                subset,
                background_instance
            )
            
            for feature in subset:
                without_feature = self._get_contribution(
                    instance,
                    [f for f in subset if f != feature],
                    background_instance
                )
                
                weight = self._get_weight(coalition_size, self.feature_count)
                shap_values[feature] += weight * (with_feature - without_feature)
                
        return ShapResult(
            values=shap_values,
            base_value=self.background_value,
            feature_names=self.feature_names
        )
        
    def explain_dataset(
        self,
        instances: np.ndarray,
        sample_size: Optional[int] = None,
        batch_size: int = 100,
        n_jobs: int = 1
    ) -> List[ShapResult]:
        """
        Generate SHAP values for multiple instances.
        
        Args:
            instances: Array of instances to explain
            sample_size: Optional number of permutations to sample
            batch_size: Number of instances to process at once
            n_jobs: Number of parallel jobs
            
        Returns:
            List of ShapResult objects
            
        Raises:
            InvalidDataError: If instances format is invalid
        """
        self._validate_data(instances)
        
        results = []
        
        with ThreadPoolExecutor(max_workers=n_jobs) as executor:
            future_to_instance = {
                executor.submit(
                    self.explain_instance,
                    instance,
                    sample_size
                ): i for i, instance in enumerate(instances)
            }
            
            for future in future_to_instance:
                results.append(future.result())
                
        return results
    
    def plot_feature_importance(
        self,
        shap_values: Union[ShapResult, List[ShapResult]],
        max_display: int = 10,
        show: bool = True
    ) -> Optional[plt.Figure]:
        """
        Plot feature importance based on SHAP values.
        
        Args:
            shap_values: SHAP results to plot
            max_display: Maximum number of features to display
            show: Whether to display the plot
            
        Returns:
            matplotlib Figure if show=False, else None
        """
        if isinstance(shap_values, ShapResult):
            values = np.abs([shap_values.values])
        else:
            values = np.abs([result.values for result in shap_values])
            
        mean_importance = np.mean(values, axis=0)
        feature_importance = pd.DataFrame({
            'feature': self.feature_names,
            'importance': mean_importance
        })
        
        feature_importance = feature_importance.nlargest(max_display, 'importance')
        
        plt.figure(figsize=(10, 6))
        sns.barplot(
            data=feature_importance,
            y='feature',
            x='importance',
            palette='viridis'
        )
        plt.title('SHAP Feature Importance')
        plt.xlabel('Mean |SHAP value|')
        
        if show:
            plt.show()
            return None
        return plt.gcf()
        
def factorial(n: int) -> int:
    """Calculate factorial of n."""
    if n == 0:
        return 1
    return n * factorial(n - 1)

def verify_additivity(
    shap_result: ShapResult,
    model_output: float,
    tolerance: float = 1e-5
) -> bool:
    """
    Verify SHAP additivity property.
    
    Args:
        shap_result: SHAP explanation result
        model_output: Actual model output
        tolerance: Numerical tolerance
        
    Returns:
        bool: Whether additivity property holds
    """
    sum_shap = np.sum(shap_result.values)
    diff = model_output - shap_result.base_value
    
    return abs(sum_shap - diff) < tolerance
