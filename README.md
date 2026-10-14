# Bitcoin Fraud Detection with Graph Neural Networks

Detecting illicit transactions in the Bitcoin network using GNNs on the Elliptic dataset.

![Python](https://img.shields.io/badge/Python-3.10+-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.11-orange)
![PyG](https://img.shields.io/badge/PyG-2.7-red)
![CPU](https://img.shields.io/badge/CPU-supported-green)

## Research objective

This project tests whether graph structure improves illicit-transaction detection,
how performance changes under temporal distribution shift, and whether stricter
evaluation or mitigation improves reliability. These are hypotheses to test;
the repository does not treat prior notebook metrics as findings.

## Dataset

The [Elliptic Bitcoin Dataset](https://www.kaggle.com/datasets/ellipticco/elliptic-data-set) 
contains 203,769 Bitcoin transactions and 234,355 payment flows, with 2% labeled as illicit.

![Class Distribution](assets/class_distribution.png)

## Evaluation protocol

The reproducible runners use the same chronological split for labelled nodes:

| Split | Timesteps | Use |
|---|---:|---|
| Training | 1–24 | Fit model parameters |
| Validation | 25–34 | Select checkpoints and operating thresholds |
| Testing | 35–49 | Final evaluation only |

Unknown labels are excluded from supervised loss and labelled metrics. Internal
labels are `-1=unknown`, `0=licit`, and `1=illicit`. Test labels do not select
epochs, hyperparameters, or thresholds.

The original notebooks contain historical experiments with different splits;
their saved scores and plots are exploratory artifacts, not comparable results
under this protocol.

## Temporal Analysis

The dataset spans 49 time steps. The experiment runner writes per-time-step
ROC-AUC, PR-AUC, precision, and illicit-class recall, alongside transaction
counts, labelled illicit proportions, and feature PSI relative to training.
Any observed degradation—including around timestep 44—must be interpreted
from generated results; the project does not attribute it to a cause without
independent evidence.

![Temporal Analysis](assets/temporal_analysis.png)
![Temporal Generalization](assets/temporal_generalization.png)

## Graph Structure

The graph image is an existing exploratory artifact and is not evidence for a
causal or predictive claim.

![Graph Structure](assets/graph_structure.png)

## Graph Visualization

1-hop neighborhood around illicit nodes (timestep 5). Red = illicit, green = licit, gray = unknown.
Treat this existing visualization as exploratory rather than as evidence that
illicit nodes cluster or that graph structure improves prediction.

![Graph Visualization](assets/graph_visualization.png)

## Research extension

The original notebooks and baseline classes are preserved. The research
extension adds `src/data_utils.py`, `src/taf_gnn.py`, `src/temporal_model.py`,
`src/fusion.py`, `src/losses.py`, `src/evaluation.py`, and the experiment
runners. It uses train timesteps 1–24, validation 25–34, and test timesteps
35–49. `--protocol transductive` intentionally exposes the complete graph,
including future nodes and edges. `--protocol inductive` exposes only nodes
and edges whose endpoints are at or before each prediction timestep.

Install dependencies and run:

```bash
python -m src.baseline_experiments --data-root data/elliptic --output results/baselines/model_comparison.csv --protocol transductive
python -m src.baseline_experiments --data-root data/elliptic --output results/inductive/model_comparison.csv --protocol inductive
python -m src.baseline_experiments --data-root data/elliptic --output results/ablations/model_comparison.csv --ablations --seeds 42,123,2024,2025,2026
python -m src.baseline_experiments --data-root data/elliptic --output results/rolling/model_comparison.csv --protocol inductive --rolling-window 10
python -m src.run_experiments --data-root data/elliptic --output-root results/taf-inductive --protocol inductive --seeds 42,123,2024,2025,2026
```

The baseline runner reports ROC-AUC, average precision (PR-AUC), precision,
recall, F1, confusion counts, Precision@100, and validation-selected thresholds.
ROC-AUC and PR-AUC are undefined when a period contains only one class. Five
seeds are the default; use `--seeds 42 --epochs 2` for a smoke run. Outputs
include per-seed and aggregate CSVs, per-period metrics, confusion matrices,
predictions, temporal plots, feature PSI, configurations, threshold
recalibration, and (optionally) rolling-window metrics. The `--ablations` flag
enables GCN depth, edge, feature-width, and feature-only controls. See
[REPORT.md](REPORT.md) for protocol details and limitations.

## Project Structure

```
bitcoin-fraud-gnn/
├── notebooks/
│   ├── 01_eda.ipynb                # Exploratory data analysis
│   ├── 02_baseline.ipynb           # Random Forest baseline
│   ├── 03_gcn.ipynb                # Graph Convolutional Network
│   ├── 04_gat.ipynb                # Graph Attention Network
│   └── 05_temporal_analysis.ipynb  # Temporal generalization analysis
├── src/
│   ├── models.py                   # Configurable-depth GCN and GAT definitions
│   ├── taf_gnn.py                  # Feature, graph, temporal, and fusion model
│   ├── temporal_model.py           # GRU/LSTM encoder
│   ├── fusion.py                   # Adaptive softmax fusion gate
│   ├── losses.py                   # CE, weighted CE, and focal loss
│   ├── evaluation.py               # Metrics and threshold selection
│   ├── data_utils.py               # Metadata alignment and inductive edge filter
│   ├── baseline_experiments.py     # RF/GCN/GAT common-protocol runner
│   └── run_experiments.py          # TAF-GNN, ablations, and result artifacts
├── app.py                          # Streamlit research interface
├── assets/                         # Plots and visualizations
└── requirements.txt
```

## Results policy

No comparative metrics or research conclusions are asserted here. Generate
results with the commands above, retain the configuration and seeds with the
artifacts, and base conclusions on those runs. The first-94-feature ablation
uses the first 94 model-input columns. The Elliptic study describes its first
94 features as local features ([Weber et al., arXiv:1908.02591](https://arxiv.org/abs/1908.02591));
this repository's processed feature matrix has 165 columns and excludes the
explicit timestep metadata column.

## How to Run

```powershell
git clone https://github.com/SanjanaSaw/Mtech-ML-PRJ.git
cd Mtech-ML-PRJ
python -m venv venv
venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Obtain the Elliptic dataset from its official distribution and place its raw
class, edge-list, and feature files under `data/elliptic/raw/`. The processed
PyG graph is expected at `data/elliptic/processed/data.pt`; do not commit the
dataset or generated model checkpoints.

When using PyTorch Geometric's dataset downloader instead, initialize the
existing dataset once from the repository root:

```powershell
python -c "from torch_geometric.datasets import EllipticBitcoinDataset; EllipticBitcoinDataset(root='data/elliptic')"
```

### Streamlit research interface

From the project root on Windows:

```powershell
streamlit run app.py
```

The interface uses cached local graph data and displays generated baseline
artifacts only when the current baseline runner has produced them. Other
research outputs remain unavailable until their corresponding experiment has
produced them.

### Reproducible baseline experiments

The common evaluation runner uses the chronological protocol shown in the UI:

```powershell
python -m src.baseline_experiments --data-root data/elliptic --output results/baselines/model_comparison.csv --protocol transductive
python -m src.baseline_experiments --data-root data/elliptic --output results/inductive/model_comparison.csv --protocol inductive
```

Both runs train on timesteps 1–24, select checkpoints and thresholds on
25–34, and report final metrics on 35–49. On Windows, keep the MKL/OpenMP
settings used by `app.py` if PyTorch reports a `c10.dll` initialization error.
