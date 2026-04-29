import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_absolute_error
import logging
from pathlib import Path

# Local imports
from src.config import OUTPUTS_DIR

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ETAPerformanceEvaluator:
    def __init__(self, y_true, y_pred, df_features):
        self.y_true = y_true
        self.y_pred = y_pred
        self.df = df_features.copy()
        self.df['actual'] = y_true
        self.df['predicted'] = y_pred
        self.df['error'] = np.abs(y_true - y_pred)
        
        OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

    def calculate_global_metrics(self):
        mae = mean_absolute_error(self.y_true, self.y_pred)
        rmse = np.sqrt(np.mean((self.y_true - self.y_pred)**2))
        logger.info(f"Global MAE: {mae:.2f}s")
        return {"MAE": mae, "RMSE": rmse}

    def run_segment_analysis(self):
        """Analyze errors across different data segments."""
        logger.info("Running segment-based error analysis...")
        
        # 1. By Hour
        hour_mae = self.df.groupby('hour')['error'].mean()
        
        # 2. By Distance Buckets
        self.df['dist_bucket'] = pd.cut(self.df['distance_km'], bins=[0, 2, 5, 10, 20, 50], labels=['0-2km', '2-5km', '5-10km', '10-20km', '20km+'])
        dist_mae = self.df.groupby('dist_bucket', observed=True)['error'].mean()
        
        # 3. By Rush Hour
        rush_mae = self.df.groupby('is_rush_hour')['error'].mean()
        
        return {
            "hour_mae": hour_mae,
            "dist_mae": dist_mae,
            "rush_mae": rush_mae
        }

    def plot_visualizations(self):
        """Generates stakeholder-ready plots."""
        logger.info("Generating visualizations...")
        sns.set_theme(style="whitegrid")
        
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        
        # Plot 1: Predicted vs Actual
        sns.scatterplot(x='actual', y='predicted', data=self.df.sample(min(5000, len(self.df))), alpha=0.3, ax=axes[0])
        axes[0].plot([self.y_true.min(), self.y_true.max()], [self.y_true.min(), self.y_true.max()], 'r--', lw=2)
        axes[0].set_title('Predicted vs. Actual Trip Duration')
        axes[0].set_xlabel('Actual Duration (s)')
        axes[0].set_ylabel('Predicted Duration (s)')
        
        # Plot 2: Error Distribution
        sns.histplot(self.df['error'], bins=50, kde=True, ax=axes[1], color='purple')
        axes[1].set_title('Distribution of Prediction Errors (MAE)')
        axes[1].set_xlabel('Absolute Error (s)')
        
        plt.tight_layout()
        plt.savefig(OUTPUTS_DIR / "performance_overview.png")
        logger.info(f"Saved visualization to {OUTPUTS_DIR / 'performance_overview.png'}")

    def print_stakeholder_report(self, metrics, segments):
        """Prints a professional summary for stakeholders."""
        print("\n" + "█"*50)
        print("  ETA PREDICTION PERFORMANCE REPORT")
        print("█"*50)
        
        print(f"\nOVERALL PERFORMANCE:")
        print(f"▸ Mean Absolute Error: {metrics['MAE']:.2f} seconds")
        print(f"▸ RMSE:                {metrics['RMSE']:.2f} seconds")
        
        print(f"\nSEGMENT INSIGHTS:")
        
        # Rush Hour Insight
        rush_diff = segments['rush_mae'][1] - segments['rush_mae'][0]
        rush_status = "Higher" if rush_diff > 0 else "Lower"
        print(f"▸ Traffic Impact: Errors are {abs(rush_diff):.1f}s {rush_status} during Rush Hour segments.")
        
        # Distance Insight
        hardest_dist = segments['dist_mae'].idxmax()
        print(f"▸ Distance Sensitivity: Model struggles most with {hardest_dist} trips.")
        
        # Hour Insight
        peak_error_hour = segments['hour_mae'].idxmax()
        print(f"▸ Time Sensitivity: Peak prediction error occurs at {peak_error_hour}:00.")
        
        print("\nSTRATEGIC RECOMMENDATIONS:")
        if metrics['MAE'] > 300:
            print("1. [CRITICAL] Investigate additional weather features to stabilize variances.")
        else:
            print("1. [OPTIMIZATION] The model is performing well; consider sub-clustering for heavy traffic zones.")
        print("2. [DATA] Increase sampling for long-distance trips (>20km) to reduce tail errors.")
        print("█"*50 + "\n")

def run_evaluation(y_true, y_pred, X_test):
    evaluator = ETAPerformanceEvaluator(y_true, y_pred, X_test)
    metrics = evaluator.calculate_global_metrics()
    segments = evaluator.run_segment_analysis()
    evaluator.plot_visualizations()
    evaluator.print_stakeholder_report(metrics, segments)

if __name__ == "__main__":
    # Example integration:
    # run_evaluation(y_val, lgb_preds, X_val)
    pass
