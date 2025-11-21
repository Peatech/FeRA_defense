# FeRA: Federated Representation Attention Defense

FeRA (Federated Representation Attention) is a defense mechanism for federated learning systems against backdoor attacks. This repository contains the **FeRA** implementation, which computes and visualizes per-client metrics for backdoor detection and applies configurable filtering logic to identify and exclude malicious clients.

## Overview

FeRA is an anomaly detection defense that:
- Computes 6 per-client metrics per round for backdoor detection
- Applies configurable multi-component filtering logic
- Supports two filter variants (v1 and v2) with switchable components
- Works with various attacks (Neurotoxin, BadNets, Chameleon, etc.)
- Supports cross-silo federated learning settings

### Metrics Computed

1. **Spectral Norm**: Largest eigenvalue of representation delta covariance
2. **Delta Norm**: Frobenius norm of representation difference
3. **Combined Score**: Weighted combination of normalized spectral + delta norms
4. **TDA (Temporal Direction Alignment)**: Cosine similarity with global model
5. **Mutual Similarity**: Mean pairwise cosine similarity with other clients
6. **Spectral norm ratio**

### Filter Variants

**Variant 1 (Default)**: Multi-component filtering with independently configurable components:
- Default Filter: Combined ≤ threshold, TDA ≤ threshold, MutualSim ≥ threshold
- Scaled Norm Filter: spectral_ratio > threshold


## Dependencies

Install the required packages:

```bash
pip install -r requirements.txt
```

### Key Dependencies

- `torch==2.6.0` - PyTorch framework
- `torchvision==0.21.0` - Vision utilities
- `ray[default]==2.10.0` - Distributed computing
- `hydra-core>=1.3.0` - Configuration management (used in main.py)
- `omegaconf>=2.3.0` - Configuration handling
- `numpy>=1.24.0` - Numerical computing
- `pandas>=2.0.0` - Data manipulation
- `scikit-learn>=1.2.0` - Machine learning utilities
- `hdbscan>=0.8.29` - Clustering algorithms
- `wandb>=0.15.0` - Experiment tracking (optional)

See `requirements.txt` for the complete list.

## Installation

1. Clone this repository:
```bash
git clone <repository-url>
cd Fera_defence
```

2. Create a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Prepare your datasets:
   - Place datasets in the `data/` directory
   - Supported datasets: CIFAR-10, CIFAR-100, MNIST, EMNIST, TinyImageNet, GTSRB, etc.

## Usage

### Basic Usage (with Hydra)

The `main.py` file uses Hydra for configuration management. Example:

```bash
python main.py \
    --config-name cifar10 \
    aggregator=fera_visualize \
    aggregator_config.fera_visualize.default_filter.enabled=true \
    aggregator_config.fera_visualize.default_filter.combined_threshold=0.60 \
    aggregator_config.fera_visualize.default_filter.tda_threshold=0.60 \
    aggregator_config.fera_visualize.default_filter.mutual_sim_threshold=0.60 \
    atk_config=cifar10_multishot \
    atk_config.data_poison_method=pattern \
    atk_config.model_poison_method=neurotoxin \
    model=resnet18 \
    alpha=0.5 \
    num_rounds=100 \
    seed=42
```

### Key Configuration Parameters

#### FeRA Visualize Parameters

- `aggregator=fera_visualize` - Use FeRA Visualize defense
- `aggregator_config.fera_visualize.spectral_weight` - Weight for spectral norm (default: 0.6)
- `aggregator_config.fera_visualize.delta_weight` - Weight for delta norm (default: 0.4)

#### Filter Configuration

**Default Filter:**
- `aggregator_config.fera_visualize.default_filter.enabled` - Enable/disable (default: true)
- `aggregator_config.fera_visualize.default_filter.combined_threshold` - Combined score threshold (default: 0.50)
- `aggregator_config.fera_visualize.default_filter.tda_threshold` - TDA threshold (default: 0.50)
- `aggregator_config.fera_visualize.default_filter.mutual_sim_threshold` - Mutual similarity threshold (default: 0.70)

#### Attack Configuration

- `atk_config.data_poison_method` - Data poisoning method: `pattern`, `badnets`, `blended`, etc.
- `atk_config.model_poison_method` - Model poisoning method: `neurotoxin`, `base`, `chameleon`, etc.
- `atk_config.poison_start_round` - Round to start poisoning
- `atk_config.poison_end_round` - Round to end poisoning

#### Cross-Silo Mode

For cross-silo federated learning (same clients participate every round):

- `cross_silo=true` - Enable cross-silo mode
- `cross_silo_num_clients=10` - Total number of clients
- `cross_silo_num_attackers=2` - Number of attackers

### Example: Running with Neurotoxin Attack

```bash
python main.py \
    --config-name cifar10 \
    aggregator=fera_visualize \
    aggregator_config.fera_visualize.default_filter.enabled=true \
    aggregator_config.fera_visualize.default_filter.combined_threshold=0.60 \
    aggregator_config.fera_visualize.default_filter.tda_threshold=0.60 \
    aggregator_config.fera_visualize.default_filter.mutual_sim_threshold=0.60 \
    aggregator_config.fera_visualize.collusion_filter.enabled=false \
    aggregator_config.fera_visualize.outlier_filter.enabled=false \
    aggregator_config.fera_visualize.scaled_norm_filter.enabled=false \
    atk_config=cifar10_multishot \
    atk_config.data_poison_method=pattern \
    atk_config.model_poison_method=neurotoxin \
    atk_config.poison_start_round=2001 \
    atk_config.poison_end_round=2100 \
    model=resnet18 \
    alpha=0.5 \
    num_rounds=100 \
    seed=42 \
    checkpoint=checkpoints/CIFAR10_unweighted_fedavg_0.5/resnet18_round_2000_dir_0.5.pth \
    training_mode=parallel \
    num_gpus=1.0 \
    num_cpus=12
```

## Project Structure

```
Fera_defence/
├── backfed/
│   ├── clients/          # Client implementations (benign and malicious)
│   ├── datasets/          # Dataset loaders
│   ├── models/            # Model architectures
│   ├── poisons/           # Poison implementations
│   ├── servers/           # Server implementations (defenses)
│   │   └── fera_visualize_server.py  # FeRA Visualize defense
│   └── utils/             # Utility modules
├── checkpoints/           # Model checkpoints (empty, add your own)
├── data/                 # Dataset directory (empty, add your own)
├── outputs/              # Output directory for results
├── main.py               # Main entry point
└── requirements.txt      # Python dependencies
```

## Critical Changes from BackFed

This repository is based on the [BackFed](https://github.com/thinh-dao/BackFed.git) framework, with the following critical modifications:

### 1. FeRA Visualize Implementation
- **New Defense**: Added `FeraVisualizeServer` with multi-component filtering logic
- **Metrics**: Implements 6 per-client metrics (spectral norm, delta norm, combined score, TDA, mutual similarity, param change count)
- **Filter Variants**: Supports two filter variants (v1 and v2) with configurable components
- **Location**: `backfed/servers/fera_visualize_server.py`

### 2. Configuration Management
- **Note**: `main.py` uses Hydra for configuration, but the core defense logic is framework-agnostic
- The FeRA Visualize server can be integrated into other frameworks by importing `FeraVisualizeServer` directly

## Supported Attacks

- **Data Poisoning**: Pattern, BadNets, Pixel, Blended, Distributed, Centralized, Edge Case, A3FL, IBA
- **Model Poisoning**: Neurotoxin, Chameleon, Anticipate, Adaptive BadNet, Base Malicious Client

## Supported Defenses

- **FeRA Visualize** (this implementation)
- FedAvg (baseline)
- Trimmed Mean
- Median (Geometric, Coordinate)
- Multi-Krum / Krum
- FLAME
- FoolsGold
- DeepSight
- RFLBAT
- FLDetector
- FLTrust
- Flare
- RobustLR
- Indicator
- LocalDP
- FedProx
- FedSPECTRE variants
- And more...

## Output

FeRA Visualize generates the following outputs per round:

1. **CSV Files**: 
   - `spectral_norm_ranked_round_{round}.csv`
   - `delta_norm_ranked_round_{round}.csv`
   - `combined_score_ranked_round_{round}.csv`
   - `tda_ranked_round_{round}.csv`
   - `mutual_sim_ranked_round_{round}.csv`
   - `param_change_count_ranked_round_{round}.csv`
   - `all_metrics_round_{round}.csv` (master file)

2. **Console Logs**: Ranked tables for quick inspection

3. **Model Checkpoints**: Saved in `checkpoints/` directory

4. **Experiment Logs**: Saved in `outputs/` directory

## Citation

If you use FeRA in your research, please cite:

```bibtex
@article{obioma2025defending,
  title={Defending the Edge: Representative-Attention Defense against Backdoor Attacks in Federated Learning},
  author={Obioma, Chibueze Peace and Sun, Youcheng and Mustafa, Mustafa A},
  journal={arXiv preprint arXiv:2505.10297},
  year={2025}
}
```

## Acknowledgments

This project is built on top of the **BackFed** framework. We acknowledge and thank the BackFed developers for providing the foundational federated learning infrastructure, client implementations, attack methods, and defense mechanisms that made this work possible.

**BackFed Framework**: Most of the framework code, including:
- Base server and client implementations
- Dataset loaders and model architectures
- Attack implementations (Neurotoxin, BadNets, Chameleon, etc.)
- Other defense mechanisms
- Utility modules and system setup

**Our Contributions**:
- FeRA Visualize defense implementation
- Cross-silo federated learning support
- Multi-component filtering logic with configurable variants

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Notes

- **Hydra Usage**: The `main.py` file uses Hydra for configuration management. If you prefer not to use Hydra, you can directly instantiate `FeraVisualizeServer` and integrate it into your own framework.
- **Checkpoints**: The `checkpoints/` directory is empty. You need to provide pre-trained model checkpoints or train from scratch.
- **Data**: The `data/` directory is empty. You need to download and place your datasets here.
- **Outputs**: Results and logs will be saved in the `outputs/` directory.

