# Research Report: TAF-GNN

## Problem and existing system

The Elliptic dataset is a temporal transaction graph with 203,769 nodes, 49
time periods, 165 numeric node attributes, and labels that include unknown,
licit, and illicit transactions. The original notebooks provide EDA, a
Random Forest, a three-layer GCN, a three-layer GAT, and a temporal plot.
Their headline numbers are retained as historical baseline results; they are
not regenerated or treated as directly comparable until the common
1-28/29-34/35-49 protocol is run.

## Methodological audit

* The raw feature file stores `txId,timestep,feature_1,...,feature_165`.
  Earlier EDA correctly recovered timestep from column 1, but it also used
  outgoing degree only. `src/data_utils.py` uses both endpoints for an
  undirected degree.
* Unknown labels are excluded from supervised loss and metrics. Internal
  labels are `-1=unknown, 0=licit, 1=illicit`.
* The original GCN/GAT call on the complete graph is **transductive**:
  future nodes and their edges can influence message passing even when their
  labels are not used. This is not the same as leakage-free inductive
  deployment.
* The research protocol is train 1-28, validation 29-34, test 35-49. The
  historical train 1-34/test 35-49 split remains available in the old
  notebooks for comparison, but should not be used for threshold tuning.

## Proposed model

TAF-GNN maps each transaction into:

1. `H_f = MLP(X)`;
2. `H_g = GNN(X,E)` using GCN by default, with GAT as a configurable option;
3. `H_t = GRU({mean(H_g,t) + embedding(t)})`, broadcast back to each node.

The fusion gate produces:

`a = softmax(MLP([H_f || H_g || H_t]))`

and `Z = a_f H_f + a_g H_g + a_t H_t`. A linear classifier produces illicit
probabilities. The per-node weights are saved in `results/fusion_weights.csv`.

Weighted cross entropy and focal loss are both implemented. The validation
period selects the operating threshold by maximum F1 on the precision-recall
curve; final test labels never select a threshold.

## Experiments

M1 feature only, M2 feature plus graph, M3 feature plus graph plus temporal,
M4 adaptive fusion, and M5 adaptive fusion with focal loss are run by
`src/run_experiments.py`. Baseline RF/GCN/GAT evaluation is in
`src/baseline_experiments.py`. Run both scripts after installing
`requirements.txt`; no result is fabricated by this repository.

The transductive run uses the complete edge index. The strict inductive run
filters edges so both endpoints are at or before timestep 34. This is a
conservative deployment protocol: future nodes may use their own transaction
features, but future graph connections cannot pass messages.

## Literature positioning and research gap

GCN, GAT, recurrent temporal encoders, focal loss, and attention/gating-based
fusion are established techniques. This implementation therefore does not
claim that any individual component is novel. The defensible contribution is a
reproducible adaptation of these components to Elliptic with (i) an explicit
learned feature/graph/time gate, (ii) validation-only threshold selection, and
(iii) side-by-side transductive and strict inductive temporal evaluation.
Novelty must be strengthened by a future systematic literature review and
statistical multi-seed comparison.

## Outputs, limitations, and future work

The runner writes predictions, metrics, ablation and temporal CSVs, error
categories, fusion weights, curves, temporal/fusion figures, and
`models/taf_gnn_best.pt`. The current
implementation is full-graph and may exceed available CPU/GPU memory; it
should be replaced by PyG neighbor sampling if required. Explainability is
bounded permutation importance by default; GNNExplainer should be applied only
to selected small neighborhoods. Results must be reported only after the
actual commands complete, including device, runtime, parameter count, and
failed experiments.
