"""
Analytics and Visualization Module
Generates dashboards, plots, and insights for agricultural data
Uses Matplotlib, Seaborn, and Plotly for interactive visualizations
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)


class AgriculturalAnalytics:
    """
    Comprehensive analytics and visualization for smart agriculture
    """
    
    def __init__(self):
        # Use matplotlib-compatible colors instead of Plotly colors
        self.colors = ['#66c2a5', '#fc8d62', '#8da0cb', '#e78ac3', '#a6d854', '#ffd92f', '#e5c494', '#b3b3b3']
        
    def plot_feature_distributions(self, df, features=None, save_path=None):
        """Plot distribution of key features"""
        print("\n Generating feature distribution plots...")
        
        if features is None:
            features = df.select_dtypes(include=[np.number]).columns.tolist()
            features = features[:8]  # Limit to 8 features
        
        n_features = len(features)
        n_cols = 3
        n_rows = (n_features + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 4*n_rows))
        axes = axes.flatten() if n_features > 1 else [axes]
        
        for idx, feature in enumerate(features):
            if feature in df.columns:
                sns.histplot(data=df, x=feature, kde=True, ax=axes[idx], color=self.colors[idx % len(self.colors)])
                axes[idx].set_title(f'Distribution of {feature}', fontsize=12, fontweight='bold')
                axes[idx].set_xlabel(feature)
                axes[idx].set_ylabel('Frequency')
        
        # Hide unused subplots
        for idx in range(n_features, len(axes)):
            axes[idx].axis('off')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f" Saved to {save_path}")
        
        plt.show()
        return fig
    
    def plot_correlation_heatmap(self, df, save_path=None):
        """Plot correlation heatmap"""
        print("\n Generating correlation heatmap...")
        
        # Select numeric columns only
        numeric_df = df.select_dtypes(include=[np.number])
        
        # Calculate correlation
        corr = numeric_df.corr()
        
        # Create figure
        fig, ax = plt.subplots(figsize=(12, 10))
        
        # Create mask for upper triangle
        mask = np.triu(np.ones_like(corr, dtype=bool))
        
        # Plot heatmap
        sns.heatmap(corr, mask=mask, annot=True, fmt='.2f', cmap='coolwarm', 
                   center=0, square=True, linewidths=1, cbar_kws={"shrink": 0.8},
                   ax=ax)
        
        ax.set_title('Feature Correlation Heatmap', fontsize=16, fontweight='bold', pad=20)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f" Saved to {save_path}")
        
        plt.show()
        return fig
    
    def plot_feature_importance(self, feature_importance_df, top_n=15, save_path=None):
        """Plot feature importance"""
        print(f"\n Generating feature importance plot (Top {top_n})...")
        
        # Get top N features
        top_features = feature_importance_df.head(top_n)
        
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Create horizontal bar plot
        bars = ax.barh(range(len(top_features)), top_features['importance'])
        ax.set_yticks(range(len(top_features)))
        ax.set_yticklabels(top_features['feature'])
        ax.set_xlabel('Importance Score', fontsize=12)
        ax.set_title(f'Top {top_n} Feature Importances', fontsize=14, fontweight='bold')
        ax.invert_yaxis()
        
        # Color bars
        colors = plt.cm.viridis(np.linspace(0.3, 0.9, len(bars)))
        for bar, color in zip(bars, colors):
            bar.set_color(color)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f" Saved to {save_path}")
        
        plt.show()
        return fig
    
    def plot_confusion_matrix(self, confusion_matrix, labels, save_path=None):
        """Plot confusion matrix"""
        print("\n Generating confusion matrix...")
        
        fig, ax = plt.subplots(figsize=(10, 8))
        
        sns.heatmap(confusion_matrix, annot=True, fmt='d', cmap='Blues',
                   xticklabels=labels, yticklabels=labels, ax=ax)
        
        ax.set_xlabel('Predicted Label', fontsize=12)
        ax.set_ylabel('True Label', fontsize=12)
        ax.set_title('Confusion Matrix', fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f" Saved to {save_path}")
        
        plt.show()
        return fig
    
    def plot_model_comparison(self, model_results, metric='accuracy', save_path=None):
        """Compare multiple models"""
        print(f"\n Generating model comparison plot ({metric})...")
        
        models = list(model_results.keys())
        scores = [model_results[model][metric] for model in models]
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        bars = ax.bar(models, scores, color=self.colors[:len(models)])
        ax.set_ylabel(metric.capitalize(), fontsize=12)
        ax.set_title(f'Model Comparison - {metric.capitalize()}', fontsize=14, fontweight='bold')
        ax.set_ylim(0, 1.1)
        
        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.3f}',
                   ha='center', va='bottom', fontsize=10, fontweight='bold')
        
        # Add target line if applicable
        if metric == 'accuracy':
            ax.axhline(y=0.90, color='red', linestyle='--', linewidth=2, label='Target (90%)')
            ax.legend()
        
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f" Saved to {save_path}")
        
        plt.show()
        return fig
    
    def plot_prediction_vs_actual(self, y_true, y_pred, save_path=None):
        """Plot predicted vs actual values (for regression)"""
        print("\n Generating prediction vs actual plot...")
        
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Scatter plot
        ax.scatter(y_true, y_pred, alpha=0.5, s=50, color='blue')
        
        # Perfect prediction line
        min_val = min(y_true.min(), y_pred.min())
        max_val = max(y_true.max(), y_pred.max())
        ax.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect Prediction')
        
        ax.set_xlabel('Actual Values', fontsize=12)
        ax.set_ylabel('Predicted Values', fontsize=12)
        ax.set_title('Predicted vs Actual Values', fontsize=14, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Calculate R 
        from sklearn.metrics import r2_score
        r2 = r2_score(y_true, y_pred)
        ax.text(0.05, 0.95, f'R  = {r2:.4f}', transform=ax.transAxes,
               fontsize=12, verticalalignment='top',
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f" Saved to {save_path}")
        
        plt.show()
        return fig
    
    def plot_residuals(self, y_true, y_pred, save_path=None):
        """Plot residual distribution"""
        print("\n Generating residual plot...")
        
        residuals = y_true - y_pred
        
        fig, axes = plt.subplots(1, 2, figsize=(15, 5))
        
        # Residual plot
        axes[0].scatter(y_pred, residuals, alpha=0.5, s=50)
        axes[0].axhline(y=0, color='r', linestyle='--', linewidth=2)
        axes[0].set_xlabel('Predicted Values', fontsize=12)
        axes[0].set_ylabel('Residuals', fontsize=12)
        axes[0].set_title('Residual Plot', fontsize=14, fontweight='bold')
        axes[0].grid(True, alpha=0.3)
        
        # Residual histogram
        axes[1].hist(residuals, bins=30, edgecolor='black', alpha=0.7)
        axes[1].set_xlabel('Residuals', fontsize=12)
        axes[1].set_ylabel('Frequency', fontsize=12)
        axes[1].set_title('Residual Distribution', fontsize=14, fontweight='bold')
        axes[1].axvline(x=0, color='r', linestyle='--', linewidth=2)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f" Saved to {save_path}")
        
        plt.show()
        return fig
    
    def create_interactive_dashboard(self, df, predictions_df=None, save_path='dashboard.html'):
        """Create interactive dashboard using Plotly"""
        print("\n Creating interactive dashboard...")
        
        # Create subplots
        fig = make_subplots(
            rows=3, cols=2,
            subplot_titles=(
                'Temperature Distribution', 'Humidity Distribution',
                'Rainfall Distribution', 'NPK Levels',
                'pH Distribution', 'Crop Distribution'
            ),
            specs=[
                [{'type': 'histogram'}, {'type': 'histogram'}],
                [{'type': 'histogram'}, {'type': 'bar'}],
                [{'type': 'histogram'}, {'type': 'pie'}]
            ]
        )
        
        # Temperature
        if 'temperature' in df.columns:
            fig.add_trace(
                go.Histogram(x=df['temperature'], name='Temperature', marker_color='indianred'),
                row=1, col=1
            )
        
        # Humidity
        if 'humidity' in df.columns:
            fig.add_trace(
                go.Histogram(x=df['humidity'], name='Humidity', marker_color='lightblue'),
                row=1, col=2
            )
        
        # Rainfall
        if 'rainfall' in df.columns:
            fig.add_trace(
                go.Histogram(x=df['rainfall'], name='Rainfall', marker_color='lightgreen'),
                row=2, col=1
            )
        
        # NPK Levels
        if all(col in df.columns for col in ['N', 'P', 'K']):
            fig.add_trace(
                go.Bar(x=['Nitrogen', 'Phosphorus', 'Potassium'],
                      y=[df['N'].mean(), df['P'].mean(), df['K'].mean()],
                      marker_color=['blue', 'green', 'red']),
                row=2, col=2
            )
        
        # pH
        if 'ph' in df.columns:
            fig.add_trace(
                go.Histogram(x=df['ph'], name='pH', marker_color='purple'),
                row=3, col=1
            )
        
        # Crop distribution
        if 'label' in df.columns:
            crop_counts = df['label'].value_counts()
            fig.add_trace(
                go.Pie(labels=crop_counts.index, values=crop_counts.values),
                row=3, col=2
            )
        
        # Update layout
        fig.update_layout(
            title_text="Smart Agriculture Analytics Dashboard",
            showlegend=False,
            height=1000
        )
        
        # Save
        fig.write_html(save_path)
        print(f" Interactive dashboard saved to {save_path}")
        
        return fig
    
    def plot_crop_suitability_map(self, crops, suitability_scores, save_path=None):
        """Plot crop suitability scores"""
        print("\n Generating crop suitability map...")
        
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Sort by suitability
        sorted_indices = np.argsort(suitability_scores)[::-1]
        sorted_crops = [crops[i] for i in sorted_indices]
        sorted_scores = [suitability_scores[i] for i in sorted_indices]
        
        # Color code
        colors = ['green' if score > 0.8 else 'orange' if score > 0.6 else 'red' 
                 for score in sorted_scores]
        
        bars = ax.barh(sorted_crops, sorted_scores, color=colors)
        ax.set_xlabel('Suitability Score', fontsize=12)
        ax.set_title('Crop Suitability Analysis', fontsize=14, fontweight='bold')
        ax.set_xlim(0, 1)
        
        # Add legend
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor='green', label='Highly Suitable (>80%)'),
            Patch(facecolor='orange', label='Moderately Suitable (60-80%)'),
            Patch(facecolor='red', label='Less Suitable (<60%)')
        ]
        ax.legend(handles=legend_elements, loc='lower right')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f" Saved to {save_path}")
        
        plt.show()
        return fig
    
    def generate_summary_statistics(self, df):
        """Generate comprehensive summary statistics"""
        print("\n Generating summary statistics...")
        
        stats = {
            'Total Records': len(df),
            'Number of Features': len(df.columns),
            'Numeric Features': len(df.select_dtypes(include=[np.number]).columns),
            'Categorical Features': len(df.select_dtypes(include=['object']).columns),
            'Missing Values': df.isnull().sum().sum(),
            'Duplicate Rows': df.duplicated().sum()
        }
        
        print("\n" + "="*60)
        print("DATASET SUMMARY STATISTICS")
        print("="*60)
        for key, value in stats.items():
            print(f"{key:<25}: {value}")
        print("="*60)
        
        return stats
    
    def plot_time_series(self, dates, values, title='Time Series', ylabel='Value', save_path=None):
        """Plot time series data"""
        print(f"\n Generating time series plot: {title}...")
        
        fig, ax = plt.subplots(figsize=(14, 6))
        
        ax.plot(dates, values, linewidth=2, marker='o', markersize=4)
        ax.set_xlabel('Date', fontsize=12)
        ax.set_ylabel(ylabel, fontsize=12)
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f" Saved to {save_path}")
        
        plt.show()
        return fig


if __name__ == "__main__":
    print("="*60)
    print("ANALYTICS AND VISUALIZATION MODULE")
    print("="*60)
    print("\n Module ready for integration")
