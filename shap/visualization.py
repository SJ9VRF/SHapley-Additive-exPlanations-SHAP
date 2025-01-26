# Plotting functions
"""Visualization functions for SHAP analysis."""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Optional, Union, Dict
from dataclasses import dataclass

@dataclass
class PlotConfig:
    """Configuration for SHAP plots."""
    figsize: tuple = (10, 6)
    cmap: str = 'viridis'
    title_fontsize: int = 12
    label_fontsize: int = 10
    
class SHAPVisualizer:
    def __init__(self, config: Optional[PlotConfig] = None):
        self.config = config or PlotConfig()
        self.set_style()
        
    def set_style(self):
        """Set default plotting style."""
        plt.style.use('seaborn')
        sns.set_palette('husl')
        
    def feature_importance(
        self,
        shap_values: np.ndarray,
        feature_names: List[str],
        max_display: int = 10,
        show: bool = True
    ) -> Optional[plt.Figure]:
        """Plot feature importance bar chart."""
        mean_shap = np.abs(shap_values).mean(axis=0)
        feature_importance = pd.DataFrame({
            'feature': feature_names,
            'importance': mean_shap
        }).nlargest(max_display, 'importance')
        
        fig, ax = plt.subplots(figsize=self.config.figsize)
        sns.barplot(data=feature_importance, x='importance', y='feature', ax=ax)
        ax.set_title('SHAP Feature Importance', fontsize=self.config.title_fontsize)
        
        if show:
            plt.show()
            return None
        return fig
        
    def summary_plot(
        self,
        shap_values: np.ndarray,
        features: np.ndarray,
        feature_names: List[str],
        show: bool = True
    ) -> Optional[plt.Figure]:
        """Create SHAP summary plot."""
        fig, ax = plt.subplots(figsize=self.config.figsize)
        
        for i in range(shap_values.shape[1]):
            sv = shap_values[:, i]
            fv = features[:, i]
            
            plt.scatter(
                sv,
                np.ones_like(sv) * i,
                c=fv,
                cmap=self.config.cmap,
                alpha=0.5
            )
            
        ax.set_yticks(range(len(feature_names)))
        ax.set_yticklabels(feature_names)
        ax.set_xlabel('SHAP value')
        
        if show:
            plt.show()
            return None
        return fig
        
    def force_plot(
        self,
        shap_values: np.ndarray,
        features: np.ndarray,
        feature_names: List[str],
        base_value: float,
        show: bool = True
    ) -> Optional[plt.Figure]:
        """Create SHAP force plot."""
        fig, ax = plt.subplots(figsize=self.config.figsize)
        
        total_effect = base_value
        x_pos = 0
        
        for i, (value, name) in enumerate(zip(shap_values, feature_names)):
            if value != 0:
                width = abs(value)
                color = 'red' if value > 0 else 'blue'
                
                ax.barh(0, width, left=x_pos, color=color, alpha=0.6)
                ax.text(
                    x_pos + width/2,
                    0,
                    f'{name}\n{value:.2f}',
                    ha='center',
                    va='center'
                )
                
                x_pos += value
                total_effect += value
        
        ax.set_ylim(-0.5, 0.5)
        ax.set_xlabel('Feature contribution')
        ax.set_title(f'Base value: {base_value:.2f}, Final prediction: {total_effect:.2f}')
        
        if show:
            plt.show()
            return None
        return fig
        
    def dependence_plot(
        self,
        shap_values: np.ndarray,
        features: np.ndarray,
        feature_names: List[str],
        feature_idx: int,
        interaction_idx: Optional[int] = None,
        show: bool = True
    ) -> Optional[plt.Figure]:
        """Create SHAP dependence plot."""
        fig, ax = plt.subplots(figsize=self.config.figsize)
        
        feature = features[:, feature_idx]
        shap_value = shap_values[:, feature_idx]
        
        if interaction_idx is not None:
            interaction = features[:, interaction_idx]
            scatter = ax.scatter(
                feature,
                shap_value,
                c=interaction,
                cmap=self.config.cmap,
                alpha=0.6
            )
            plt.colorbar(scatter, label=feature_names[interaction_idx])
        else:
            ax.scatter(feature, shap_value, alpha=0.6)
            
        ax.set_xlabel(feature_names[feature_idx])
        ax.set_ylabel(f'SHAP value for {feature_names[feature_idx]}')
        
        if show:
            plt.show()
            return None
        return fig
