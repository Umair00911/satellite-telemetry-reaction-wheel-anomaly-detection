import os
import sys
import json
import numpy as np
import pandas as pd

# Add workspace to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.data_pipeline import SatelliteDataPipeline
from src.models.kan_model import train_kan_model
from src.models.gnn_model import train_gnn_model
from src.models.mamba_model import train_mamba_model
from src.models.xlstm_model import train_xlstm_model
from src.models.gan_augmentor import GANAugmentor, augment_dataset
from src.models.federated_learning import train_federated_learning
from src.evaluation.comparator import ModelComparator
from src.evaluation.stat_analysis import StatisticalAnalyzer
from src.explainability.shap_xai import SatelliteXAIExplainer

def run_satellite_anomaly_detection_pipeline():
    print("=" * 80)
    print("🚀 SATELLITE ANOMALY DETECTION SYSTEM - END-TO-END PIPELINE RUNNER")
    print("=" * 80)

    # Remove previous result files before starting a new experiment
    files_to_remove = [
        "benchmark_results.csv",
        "feature_importance.csv",
        "statistical_analysis.csv",
        "best_generator.pt"
    ]

    for file in files_to_remove:
        if os.path.exists(file):
            os.remove(file)
            print(f"🗑 Deleted old file: {file}")

    # 1-6. Data Pipeline & Preprocessing
    pipeline = SatelliteDataPipeline(data_dir='downloaded_files', sequence_length=10, max_files_per_category=20)
    data = pipeline.process_pipeline()

    X_train_2d, y_train_2d = data['2d']['X_train'], data['2d']['y_train']
    X_val_2d, y_val_2d = data['2d']['X_val'], data['2d']['y_val']
    X_test_2d, y_test_2d = data['2d']['X_test'], data['2d']['y_test']

    X_train_3d, y_train_3d = data['3d']['X_train'], data['3d']['y_train']
    X_val_3d, y_val_3d = data['3d']['X_val'], data['3d']['y_val']
    X_test_3d, y_test_3d = data['3d']['X_test'], data['3d']['y_test']
    feature_names = data['feature_names']

    comparator = ModelComparator()

    # 7-8. Train & Evaluate KAN
    kan_res = train_kan_model(X_train_2d, y_train_2d, X_val_2d, y_val_2d, X_test_2d, y_test_2d, epochs=20)
    comparator.evaluate_model('KAN', y_test_2d, kan_res['y_pred'], kan_res['y_prob'], kan_res['train_time'], kan_res['latency_ms'])

    # 9-10. Train & Evaluate GNN
    gnn_res = train_gnn_model(X_train_2d, y_train_2d, X_val_2d, y_val_2d, X_test_2d, y_test_2d, epochs=150)
    comparator.evaluate_model('GNN', y_test_2d, gnn_res['y_pred'], gnn_res['y_prob'], gnn_res['train_time'], gnn_res['latency_ms'])

    # 11-12. Train & Evaluate Mamba
    mamba_res = train_mamba_model(X_train_3d, y_train_3d, X_val_3d, y_val_3d, X_test_3d, y_test_3d, epochs=20)
    comparator.evaluate_model('Mamba', y_test_3d, mamba_res['y_pred'], mamba_res['y_prob'], mamba_res['train_time'], mamba_res['latency_ms'])

    # 13-14. Train & Evaluate xLSTM
    xlstm_res = train_xlstm_model(X_train_3d, y_train_3d, X_val_3d, y_val_3d, X_test_3d, y_test_3d, epochs=20)
    comparator.evaluate_model('xLSTM', y_test_3d, xlstm_res['y_pred'], xlstm_res['y_prob'], xlstm_res['train_time'], xlstm_res['latency_ms'])

    # 15-16. Train GAN & Retrain Model
    anomaly_idx = np.where(y_train_2d == 1)[0]
    X_anomaly = X_train_2d[anomaly_idx]
    gan_aug = GANAugmentor(feature_dim=X_train_2d.shape[1], latent_dim=64)
    gan_aug.train_gan(X_anomaly, epochs=70)

    X_train_aug, y_train_aug = augment_dataset(X_train_2d, y_train_2d, gan_aug, num_synth=500)
    kan_retrained_res = train_kan_model(X_train_aug, y_train_aug, X_val_2d, y_val_2d, X_test_2d, y_test_2d, epochs=20)
    comparator.evaluate_model('KAN (GAN-Augmented)', y_test_2d, kan_retrained_res['y_pred'], kan_retrained_res['y_prob'], kan_retrained_res['train_time'], kan_retrained_res['latency_ms'])

    # 17. Train Federated Learning Simulator
    fl_res = train_federated_learning(mamba_res['model'], X_train_3d, y_train_3d, X_test_3d, y_test_3d, num_nodes=5, fl_rounds=15)
    comparator.evaluate_model('Federated Learning', y_test_3d, fl_res['y_pred'], fl_res['y_prob'], fl_res['train_time'], fl_res['latency_ms'])

    # 18. Benchmark Comparison Table
    comparator.print_comparison_table()
    results_df = comparator.get_summary_dataframe()

    # 19. Statistical Significance Analysis
    stat_analyzer = StatisticalAnalyzer(results_df.to_dict(orient='records'))
    stat_results = stat_analyzer.run_hypothesis_tests()

    # 20. SHAP / XAI Explainability
    explainer = SatelliteXAIExplainer(mamba_res['model'], feature_names)
    xai_rankings = explainer.compute_feature_attributions(X_test_3d[:100])

    # Save Pipeline Benchmarks & Reports directly to CSV for pure CLI analysis
    results_df.to_csv('benchmark_results.csv', index=False)
    xai_rankings.to_csv('feature_importance.csv', index=False)
    pd.DataFrame(stat_results['pairwise_tests']).to_csv('statistical_analysis.csv', index=False)

    print("\n🎉 SATELLITE ANOMALY DETECTION SYSTEM PIPELINE COMPLETED SUCCESSFULLY!")
    print("💾 Generated CLI Benchmark CSV Reports:")
    print("   - benchmark_results.csv (Model Comparison Suite)")
    print("   - statistical_analysis.csv (ANOVA & Hypothesis Tests)")
    print("   - feature_importance.csv (SHAP XAI Sensor Attributions)")
    return {
        'model_benchmarks': results_df.to_dict(orient='records'),
        'statistical_analysis': stat_results,
        'feature_importance': xai_rankings.to_dict(orient='records')
    }

if __name__ == '__main__':
    run_satellite_anomaly_detection_pipeline()
