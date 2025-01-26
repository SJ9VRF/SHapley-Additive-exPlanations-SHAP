# Utility functions
"""Utility functions for SHAP implementation."""

import numpy as np
import pandas as pd
from typing import List, Dict, Union, Optional, Tuple
from sklearn.preprocessing import StandardScaler
from dataclasses import dataclass

@dataclass
class FeatureStats:
    """Statistics for feature analysis."""
    mean: float
    std: float
    min: float
    max: float
    quantiles: Dict[str, float]

def preprocess_data(
    data: Union[np.ndarray, pd.DataFrame],
    normalize: bool = True
) -> Tuple[np.ndarray, Optional[StandardScaler]]:
    """Preprocess input data for SHAP analysis."""
    if isinstance(data, pd.DataFrame):
        data = data.to_numpy()
    
    scaler = None
    if normalize:
        scaler = StandardScaler()
        data = scaler.fit_transform(data)
    
    return data, scaler

def calculate_feature_stats(
    data: Union[np.ndarray, pd.DataFrame],
    feature_names: Optional[List[str]] = None
) -> Dict[str, FeatureStats]:
    """Calculate comprehensive feature statistics."""
    if isinstance(data, np.ndarray):
        data = pd.DataFrame(data, columns=feature_names)
    
    stats = {}
    for column in data.columns:
        values = data[column]
        stats[column] = FeatureStats(
            mean=float(values.mean()),
            std=float(values.std()),
            min=float(values.min()),
            max=float(values.max()),
            quantiles={
                '25%': float(values.quantile(0.25)),
                '50%': float(values.quantile(0.50)),
                '75%': float(values.quantile(0.75))
            }
        )
    
    return stats

def generate_kernel_weights(
    num_features: int,
    kernel_width: float = 0.25
) -> np.ndarray:
    """Generate kernel weights for SHAP sampling."""
    distances = np.linspace(0, 1, num_features)
    weights = np.exp(-(distances ** 2) / (2 * kernel_width ** 2))
    return weights / np.sum(weights)

def verify_shap_properties(
    shap_values: np.ndarray,
    model_output: float,
    base_value: float,
    tolerance: float = 1e-5
) -> Dict[str, bool]:
    """Verify key SHAP properties."""
    return {
        'local_accuracy': abs(np.sum(shap_values) - (model_output - base_value)) < tolerance,
        'efficiency': abs(np.sum(shap_values)) > tolerance,
        'symmetry': len(set(abs(shap_values))) < len(shap_values)
    }

def sample_background_data(
    data: np.ndarray,
    sample_size: int,
    random_state: Optional[int] = None
) -> np.ndarray:
    """Sample background data for SHAP analysis."""
    if random_state is not None:
        np.random.seed(random_state)
    
    indices = np.random.choice(len(data), size=sample_size, replace=False)
    return data[indices]
