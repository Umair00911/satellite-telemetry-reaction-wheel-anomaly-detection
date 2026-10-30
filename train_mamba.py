import os
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.data_pipeline import SatelliteDataPipeline
from src.models.mamba_model import train_mamba_model
from src.evaluation.comparator import ModelComparator

def main():
    print("=" * 70)
    print("🐍 INDIVIDUAL TRAINING RUNNER: Mamba (Selective SSM)")
    print("=" * 70)

    # 1. Load & Process Telemetry Data
    pipeline = SatelliteDataPipeline(data_dir='downloaded_files', sequence_length=10)
    data = pipeline.process_pipeline()

    X_train, y_train = data['3d']['X_train'], data['3d']['y_train']
    X_val, y_val = data['3d']['X_val'], data['3d']['y_val']
    X_test, y_test = data['3d']['X_test'], data['3d']['y_test']

    # 2. Train & Evaluate Mamba
    res = train_mamba_model(X_train, y_train, X_val, y_val, X_test, y_test, epochs=25)

    # 3. Print Performance Report
    comparator = ModelComparator()
    comparator.evaluate_model('Mamba', y_test, res['y_pred'], res['y_prob'], res['train_time'], res['latency_ms'])
    comparator.print_comparison_table()

if __name__ == '__main__':
    main()
