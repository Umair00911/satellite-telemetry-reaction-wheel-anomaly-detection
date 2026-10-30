# Satellite Telemetry Reaction-Wheel Anomaly Detection 

Research code for binary Reaction-Wheel anomaly detection in satellite telemetry, comparing custom neural architectures, synthetic anomaly augmentation, and simulated federated learning.

Maintainer: [Umair00911](https://github.com/Umair00911). Intended repository: `Umair00911/satellite-telemetry-anomaly-detection`.

## Overview

The pipeline loads normal and reaction-wheel attack telemetry, engineers physical features, balances classes, and compares six experimental configurations. Implementations are local Python/PyTorch code; the project does not require external KAN, Mamba, xLSTM, or graph-learning packages.

| Configuration | Implementation | Input |
| --- | --- | --- |
| KAN | Three KAN layers using a linear base plus learned Chebyshev basis weights, with SiLU activations | Tabular features |
| GNN | Dynamic sensor adjacency, three graph convolutions, residual connection, and attention-weighted readout | Tabular features treated as sensor nodes |
| Mamba | Four custom selective state-space blocks, depthwise convolution, gating, residuals, and temporal attention pooling | Length-10 windows |
| xLSTM | Two custom stabilized exponential-gating sLSTM layers and classifier | Length-10 windows |
| KAN + GAN | WGAN-GP trained on anomaly vectors; 500 synthetic anomalies added before retraining KAN | Tabular features |
| Federated learning | Local simulation of five nodes with sample-weighted FedAvg, initialized from the trained Mamba model | Length-10 windows |

The GNN training function selects `GNNTelemetryClassifierSimple`; a separate multi-head graph-attention architecture is also defined but is not selected by the provided runner. These are custom research implementations, not claims of exact reproduction of the original architecture papers.

## Installation

Use an isolated Python environment. Python 3.10 is a practical starting point matching the inspected system environment; a complete clean-environment training run has not been validated. CPU execution is supported; CUDA is selected automatically when PyTorch detects it.

```powershell
git clone https://github.com/Umair00911/satellite-telemetry-anomaly-detection.git
cd satellite-telemetry-anomaly-detection
python -m venv scad_env
.\scad_env\Scripts\python.exe -m pip install --upgrade pip
.\scad_env\Scripts\python.exe -m pip install -r requirements.txt
```

On Linux/macOS, use `python3 -m venv scad_env` and `scad_env/bin/python` in place of the Windows interpreter path. For a particular GPU environment, install a compatible PyTorch build for that system before using the scripts.

`requirements.txt` lists dependencies imported by the source. Versions are intentionally unpinned: the original environment export omitted PyTorch and contained unrelated visualization/Jupyter packages, so it is not a validated reproducibility lockfile. `imbalanced-learn` supplies optional SMOTE; without it the pipeline uses random oversampling. PyTorch is required by the full pipeline because the GAN imports it unconditionally. NumPy fallback classes in other modules are simplified alternatives and are not equivalent to the neural models.

## Dataset

Place locally authorized telemetry CSV files in this layout:

```text
downloaded_files/
├── baseline/       # Normal telemetry
├── attack_rwa/     # Reaction-wheel A attacks
├── attack_rwb/     # Reaction-wheel B attacks
└── attack_rwc/     # Reaction-wheel C attacks
```

The inspected local collection contains FireSat-named metadata: 121 baseline CSV files, 601 CSV files in each attack directory, and one long-baseline CSV. Counts include category metadata files. The loader uses every `*.csv` candidate, including metadata files, and takes at most the first 20 files per category by default. `long_baseline/` and root `all_metadata.csv` are not directly loaded. No dataset download script, authoritative source URL, or redistribution license was identified; obtain the data separately from its provider and confirm permitted use. Raw datasets and metadata are excluded from this repository.

If `result_label` exists, `baseline` maps to class 0 and every other value maps to class 1. Otherwise the folder determines the label. Telemetry columns include angular velocity, wheel speeds, battery charge, magnetic field, attitude, and position. Object/text columns are dropped; numeric columns are retained. The exact feature set depends on supplied files.

When no usable CSV is loaded, the loader automatically generates a 1,200-row demonstration dataset with 20% injected anomalies. This is a synthetic demonstration, not a replacement for real-data validation.

## Preprocessing

The implemented order is:

1. Load and concatenate selected CSVs and derive binary labels.
2. Drop metadata/text columns; coerce remaining values to numeric; fill missing values with column medians, then zero.
3. Where source columns exist, derive wheel-speed magnitude, absolute A/B wheel-speed difference, angular-velocity magnitude, and load-to-charge ratio.
4. Fit `RobustScaler` on all rows.
5. Apply SMOTE when available and eligible, otherwise random oversampling.
6. Perform stratified random splits into 70% training, 15% validation, and 15% test with split seed 42.
7. Create overlapping length-10 windows independently within each split; sequence labels are truncated from the start of each split.

See the methodological limitations below before interpreting metrics.

## Usage

Run commands from the repository root. Individual training scripts report metrics in the terminal and do not save the three combined CSV reports.

### 1. Train one model

```powershell
.\scad_env\Scripts\python.exe train_kan.py
.\scad_env\Scripts\python.exe train_gnn.py
.\scad_env\Scripts\python.exe train_mamba.py
.\scad_env\Scripts\python.exe train_xlstm.py
```

Each standalone runner requests 25 epochs. GNN uses learning rate 0.001 and batch size 32 in its standalone runner. Hyperparameters are specified in Python functions; no command-line argument parser is implemented.

### 2. Run the complete comparison

**Existing behavior:** `main_pipeline.py` deletes `benchmark_results.csv`, `feature_importance.csv`, `statistical_analysis.csv`, and `best_generator.pt` from the current working directory before training. Preserve any new experiment outputs before rerunning it. The supplied historical CSVs are additionally archived in `results/original/`, outside those deletion targets.

```powershell
.\scad_env\Scripts\python.exe main_pipeline.py
```

Alternatively use `run.bat` or `run.ps1`, which expect the `scad_env` environment above. On Windows terminals that cannot print the scripts' Unicode symbols, set `$env:PYTHONIOENCODING = 'utf-8'` first.

The full runner requests KAN/Mamba/xLSTM training for 20 epochs each, GNN for 150, GAN for 70, augmented KAN for 20, and FedAvg for 15 rounds with eight local epochs per round. Training may be expensive, particularly for the GNN and federated simulation. Outputs are:

| File | Contents |
| --- | --- |
| `benchmark_results.csv` | Classification metrics, confusion counts, training time, and per-sample latency |
| `feature_importance.csv` | Normalized feature attribution percentages from up to 100 test windows |
| `statistical_analysis.csv` | Pairwise tests and confidence intervals based on simulated fold scores |
| `best_generator.pt` | Generator state saved during GAN training; excluded from Git |

### 3. Inspect saved reports without training

```powershell
.\scad_env\Scripts\python.exe -c "import pandas as pd; print(pd.read_csv('results/original/benchmark_results.csv').to_string(index=False))"
```

### 4. Optional file server

```powershell
.\scad_env\Scripts\python.exe web_dashboard.py
```

Open `http://localhost:8080`. Despite its name, this script serves repository files using Python's standard HTTP server; no dashboard HTML or interactive visualization is included. It binds all network interfaces, so run it only where exposing the served directory is intended. Stop with Ctrl+C.

## Evaluation and saved results

`ModelComparator` reports accuracy, precision, recall, F1, ROC-AUC, average precision (stored under `PR-AUC`), TP/FP/TN/FN, training seconds, and inference milliseconds per sample. ROC-AUC and average precision default to 0.5 when their calculation raises an exception.

The following values are transcribed from the preserved `benchmark_results.csv`; they were not rerun or independently reproduced during repository preparation.

| Model | Accuracy | F1 | ROC-AUC |
| --- | ---: | ---: | ---: |
| KAN | 0.9841 | 0.9843 | 0.9967 |
| GNN | 0.9086 | 0.9086 | 0.9710 |
| Mamba | 0.9877 | 0.9878 | 0.9995 |
| xLSTM | 0.9832 | 0.9833 | 0.9986 |
| KAN (GAN-Augmented) | 0.9885 | 0.9887 | 0.9989 |
| Federated Learning | 0.9880 | 0.9881 | 0.9994 |

## Methodological limitations

- Median filling, scaling, and oversampling occur before the split. This exposes held-out information and can place duplicate or related synthetic observations across splits, inflating evaluation estimates.
- Random splitting occurs before window construction, so windows do not preserve original chronological order or simulation boundaries. Sequence labels use the first-row position rather than an explicit endpoint or any-anomaly rule.
- File discovery is unsorted and includes metadata CSVs. Sampling and retained numeric metadata can affect results; some attack-related numeric fields may act as shortcuts. There is no file/session-held-out evaluation.
- FedAvg chooses its best round using **test** accuracy, compromising an independent final test. It is a sequential local simulation, not an operational distributed or privacy-preserving deployment.
- Statistical analysis draws five random scores around each single-run F1 value. ANOVA, paired t-tests, Wilcoxon tests, and confidence intervals therefore describe simulated scores, not measured cross-validation folds, and do not establish empirical significance. The no-SciPy path uses approximate placeholder p-values.
- `shap_xai.py` computes absolute Gradient × Input for the anomaly logit, averaged and normalized over samples/time. It does not implement SHAP. Without a differentiable PyTorch model it ranks mean absolute input magnitude instead.
- Seeds are only partially controlled. Runtime hardware, complete dependency locks, split manifests, and repeated-run measurements are not recorded. Sequence and tabular evaluations also have different sample counts.
- NumPy fallbacks differ materially from the PyTorch implementations; the KAN fallback can have shape incompatibilities with small feature sets. The full pipeline requires PyTorch.

Algorithms and historical reports are preserved as supplied. Addressing these limitations requires a separately reviewed research change; the published snapshot should not be presented as a fully validated benchmark.

## Repository layout

```text
main_pipeline.py               # Complete six-configuration experiment
train_{kan,gnn,mamba,xlstm}.py  # Standalone model runners
src/data_pipeline.py           # Loading, features, balancing, splits, windows
src/models/                    # Neural models, GAN, and FedAvg
src/evaluation/                # Metrics and simulated statistical analysis
src/explainability/shap_xai.py  # Gradient × Input attribution
results/original/              # Untouched historical CSV snapshots
benchmark_results.csv          # Original report; overwritten by full runner
feature_importance.csv         # Original report; overwritten by full runner
statistical_analysis.csv       # Original report; overwritten by full runner
web_dashboard.py               # Optional static file server
read_req.py                    # Legacy requirements display helper
run.bat / run.ps1              # Windows launchers
requirements.txt              # Source-derived direct dependencies
.gitignore                     # Data, environment, checkpoint, secret exclusions
```

## Preservation and licensing

Repository preparation keeps source files byte-for-byte unchanged and retains historical CSVs. The original local dataset, virtual environment, editor settings, and generator checkpoint remain outside the upload copy. No license was supplied or inferred; public visibility alone does not grant redistribution rights. Dataset permissions are separate from code permissions.
