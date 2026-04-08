# RAG-TrajAD

**Retrieval-Augmented Cross-Domain Trajectory Anomaly Detection**

RAG-TrajAD is a retrieval-augmented framework for cross-domain trajectory anomaly detection that requires **zero target-domain labels**. It transfers knowledge from labeled source domains to detect anomalies in unseen target domains by retrieving and comparing against normal trajectory patterns.

## Architecture

RAG-TrajAD consists of three modules:

```
                        ┌─────────────────────┐
   GPS Trajectory ───>  │  SharedTrajEncoder   │ ──> z (trajectory embedding)
                        │  (Transformer + GRL) │
                        └─────────────────────┘
                                  │
                                  ▼
                        ┌─────────────────────┐
   z (query) ────────>  │  CrossDomainMemory   │ ──> Top-K normal neighbors
                        │  (Normal-only store) │
                        └─────────────────────┘
                                  │
                                  ▼
                        ┌─────────────────────┐
   z + neighbors ────>  │ CrossAttentionScorer │ ──> anomaly score
                        │ (Query slots + kNN)  │
                        └─────────────────────┘
```

1. **SharedTrajEncoder**: 4-layer Transformer with spatial/motion/temporal tokenization, masked token reconstruction pretraining, and domain-adversarial training via gradient reversal (GRL).
2. **CrossDomainMemory**: Stores compressed token-level features from normal source-domain trajectories for retrieval.
3. **CrossAttentionScorer**: Learnable query slots cross-attend to retrieved normal trajectory tokens, with automatic scoring mode selection (learned scorer / kNN distance / blend).

## Training Pipeline

```
Stage 1: Pretrain encoder (masked reconstruction + domain adversarial)
    ↓
Stage 2: Build normal-only memory from source domains
    ↓
Stage 3: Train scorer (frozen encoder, cross-attention on retrieved normals)
    ↓
Evaluate: Auto-select best scoring mode per target
```

## Results

Leave-one-domain-out evaluation (seed 42, zero-shot):

| Method | Porto | T-Drive | GeoLife | Mean AUROC |
|--------|-------|---------|---------|------------|
| DeepSAD | **0.658** | 0.871 | 0.516 | 0.682 |
| T2Vec+KNN | 0.637 | 0.879 | 0.582 | 0.699 |
| **RAG-TrajAD (Ours)** | 0.621 | **0.899** | **0.686** | **0.735** |

RAG-TrajAD achieves +7.8% over DeepSAD and +5.2% over T2Vec+KNN on mean AUROC, with the strongest gains on GeoLife (+17.0% over DeepSAD).

## Requirements

```bash
pip install -r requirements.txt
```

- Python >= 3.8
- PyTorch >= 2.0
- NumPy >= 1.24
- pandas >= 1.5
- scikit-learn >= 1.2

## Data Preparation

Prepare three trajectory datasets under `data/`:

```
data/
├── porto/
│   ├── train.csv
│   └── test.csv
├── tdrive/
│   ├── train.csv
│   └── test.csv
└── geolife/
    ├── train.csv
    └── test.csv
```

Each CSV should contain columns: `trajectory_id`, `latitude`, `longitude`, `timestamp`.

- **Porto**: [Porto taxi dataset](https://www.kaggle.com/c/pkdd-15-predict-taxi-service-trajectory-i)
- **T-Drive**: [T-Drive taxi trajectory dataset](https://www.microsoft.com/en-us/research/publication/t-drive-trajectory-data-sample/)
- **GeoLife**: [GeoLife GPS trajectory dataset](https://www.microsoft.com/en-us/research/project/geolife-building-social-networks-using-human-location-history/)

## Usage

### Train and Evaluate

```bash
# Train RAG-TrajAD with Porto as target (zero-shot)
python src/train.py \
    --data_path data/ \
    --source tdrive \
    --target porto \
    --seed 42 \
    --d_model 128 \
    --n_heads 4 \
    --n_encoder_layers 4 \
    --pretrain_epochs 10 \
    --scorer_epochs 15 \
    --grl_max_weight 0.15

# Train with T-Drive as target
python src/train.py \
    --data_path data/ \
    --source porto \
    --target tdrive \
    --seed 42

# Train with GeoLife as target
python src/train.py \
    --data_path data/ \
    --source porto \
    --target geolife \
    --seed 42
```

### Run Baselines

```bash
python src/run_baselines.py --data_path data/ --seed 42
```

### Key Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--d_model` | 128 | Transformer hidden dimension |
| `--n_heads` | 4 | Number of attention heads |
| `--n_encoder_layers` | 4 | Number of Transformer layers |
| `--top_k` | 20 | Number of neighbors to retrieve |
| `--pretrain_epochs` | 10 | Pretraining epochs |
| `--scorer_epochs` | 15 | Scorer training epochs |
| `--grl_max_weight` | 0.5 | Max GRL weight (use 0.15 for Porto) |
| `--no_domain_adv` | False | Disable domain adversarial training |
| `--k_shot` | 0 | Number of target anomaly labels (0 = zero-shot) |

## Project Structure

```
RAG-TrajAD/
├── README.md
├── requirements.txt
├── .gitignore
└── src/
    ├── model.py           # RAGTrajAD model (Encoder + Memory + Scorer)
    ├── dataset.py         # Dataset loading (Porto, T-Drive, GeoLife)
    ├── train.py           # 3-stage training pipeline
    ├── evaluate.py        # Evaluation metrics (AUROC, AUPRC, FPR@95)
    ├── baselines.py       # Baseline implementations (DeepSAD, T2Vec+KNN, etc.)
    └── run_baselines.py   # Baseline runner script
```

## Citation

If you find this work useful, please cite:

```bibtex
@inproceedings{ragtrajad2026,
  title={RAG-TrajAD: Retrieval-Augmented Cross-Domain Trajectory Anomaly Detection},
  author={Anonymous},
  booktitle={International Conference on Wireless Algorithms, Systems, and Applications (WASA)},
  year={2026}
}
```

## License

This project is released under the MIT License.
