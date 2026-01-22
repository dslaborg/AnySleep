# Base Module

This module contains the core components for the AnySleep sleep stage classification framework. It provides the building
blocks for loading polysomnography (PSG) data, defining neural network architectures, training models, and evaluating
their performance.

## Overview

The `base/` module is organized into several submodules:

```
base/
├── config.py          # Global configuration management
├── utils.py           # Utility functions
├── data/              # Dataset classes for loading and sampling PSG data
├── model/             # Neural network architectures (USleep, AnySleep)
├── training/          # Training loop and learning rate schedulers
├── evaluation/        # Model evaluation during and after training
└── results/           # Metrics tracking and result logging
```

## Submodules

### Configuration (`config.py`)

The `Config` class implements a singleton pattern to provide global access to the Hydra configuration throughout the
codebase.

### Utilities (`utils.py`)

Helper functions used across the codebase:

- **`DelayedExecutor`**: Schedules function execution with a delay, canceling previous pending calls. Used for efficient
  file writing during training.
- **`calc_mode(data, axis)`**: Computes the mode (most frequent value) along an array axis with random tie-breaking.
  Used for majority voting across channels.
- **`choice(listable, n, random_state)`**: Randomly selects n items from a list with optional reproducibility.

### Data Module (`data/`)

Classes for loading and sampling polysomnography (PSG) data from HDF5 files.

#### Expected Data Format

The HDF5 files should follow this structure:

```
File
├── Dataset_Name/
│   ├── Subject_Name/
│   │   ├── PSG/
│   │   │   ├── EEG_Channel_1
│   │   │   ├── EEG_Channel_2
│   │   │   └── EOG_Channel_1
│   │   ├── hypnogram        # Sleep stage labels (0=Wake, 1=N1, 2=N2, 3=N3, 4=REM)
│   │   └── class_to_index/  # Precomputed indices for balanced sampling
│   │       ├── 0            # Indices of Wake epochs
│   │       ├── 1            # Indices of N1 epochs
│   │       └── ...
│   └── ...
└── ...
```

#### Dataset Classes

| Class                  | Purpose                                                            |
|------------------------|--------------------------------------------------------------------|
| `BaseDataset`          | Abstract base class with common functionality                      |
| `UsleepTrainDataset`   | Training dataset with balanced sampling (Perslev et al., 2021)     |
| `UsleepEvalDataset`    | Evaluation dataset returning full recordings                       |
| `AnySleepTrainDataset` | Training dataset for attention models with variable channel counts |
| `AnySleepEvalDataset`  | Evaluation dataset for attention models                            |

#### Balanced Sampling Strategy

The training datasets implement the sampling strategy
from [U-Sleep (Perslev et al., 2021)](https://doi.org/10.1038/s41746-021-00440-5):

1. **Dataset sampling**: Combines uniform sampling across datasets with size-weighted sampling
2. **Segment sampling**: Ensures rare sleep stages (N1, N3) are adequately represented by:
    - Uniformly selecting a sleep stage
    - Choosing a random 30-second epoch of that stage
    - Positioning a window around the chosen epoch

#### Custom DataLoader

`CustomDataloader` extends PyTorch's DataLoader to ensure correct random sampling across multiple workers:

### Model Module (`model/`)

Neural network architectures for sleep stage classification.

```
model/
├── base_model.py              # Abstract base class with save/load
├── usleep.py                  # USleep: standard 2-channel U-Net
├── anysleep.py                # AnySleep: mid-fusion attention on skip connections
├── anysleep_early_fusion.py   # AnySleepEF: early fusion attention before encoder
├── anysleep_late_fusion.py    # AnySleepLF: late fusion attention after decoder
├── usleep_base_components.py  # Shared encoder, decoder, classifier blocks
└── utilities.py               # Filter size calculation
```

#### Architecture Overview

**USleep** (baseline): Standard U-Net architecture with fixed 2-channel input (EEG + EOG).

**AnySleep variants** support variable channel counts through learned attention:

| Variant                | Fusion Point     | Architecture                                       | Trade-offs                                            |
|------------------------|------------------|----------------------------------------------------|-------------------------------------------------------|
| **AnySleep** (mid)     | Skip connections | Attention combines residuals during decoding       | Balanced: attention sees partially processed features |
| **AnySleepEF** (early) | Before encoder   | Attention on raw input → virtual channels → U-Net  | Faster: processes fewer channels through U-Net        |
| **AnySleepLF** (late)  | After decoder    | Each channel through U-Net → attention on features | More expressive: attention on learned representations |

#### Available Models

| Model        | File                       | Description                                             |
|--------------|----------------------------|---------------------------------------------------------|
| `USleep`     | `usleep.py`                | Original U-Net for sleep staging (Perslev et al., 2021) |
| `AnySleep`   | `anysleep.py`              | Mid-fusion: attention on skip connections               |
| `AnySleepEF` | `anysleep_early_fusion.py` | Early fusion: attention before encoder                  |
| `AnySleepLF` | `anysleep_late_fusion.py`  | Late fusion: attention after decoder                    |

#### Model Components

- **`USleepEncoderBlock`**: Conv1d → ELU → BatchNorm → MaxPool
- **`USleepDecoderBlock`**: Upsample → Conv1d → ELU → BatchNorm → Concatenate with residual
- **`SegmentClassifier`**: Final classification head producing per-epoch predictions
- **`SkipConnectionBlock`**: Channel attention for AnySleep (mid-fusion)
- **`ChannelAttention`**: Channel attention for early/late fusion variants

### Training Module (`training/`)

Components for model training with early stopping and learning rate scheduling.

#### ClasTrainer

Main training orchestrator with:

- Early stopping based on macro F1 score
- Periodic model checkpoints
- Gradient clipping

#### CyclicCosineDecayLR

Custom learning rate scheduler with:

- Optional warmup phase
- Cosine annealing decay
- Optional cyclic restarts with geometric interval growth

### Evaluation Module (`evaluation/`)

Classes for evaluating model performance during and after training.

#### ClasEvaluator

Runs inference on validation/test sets and tracks metrics:

### Results Module (`results/`)

Metric tracking and result persistence.

#### SleepStageResultTracker

Comprehensive metrics tracking with multiple granularity levels:

- **Datasplit level**: Overall performance on train/val/test
- **Dataset level**: Performance per dataset (e.g., MASS, Sleep-EDF)
- **Recording level**: Per-subject performance
- **Channel level**: Performance for specific EEG/EOG combinations

Features:

- **Concatenation**: Standard metrics on all predictions
- **Majority voting**: Consensus across channel combinations per recording

#### FastSleepStageResultTracker

Lightweight tracker for training that only computes datasplit-level metrics.

## Sleep Stage Labels

| Value | Stage    | Description                    |
|-------|----------|--------------------------------|
| 0     | Wake (W) | Awake state                    |
| 1     | N1       | Light sleep, Stage 1           |
| 2     | N2       | Light sleep, Stage 2           |
| 3     | N3       | Deep sleep (slow-wave)         |
| 4     | REM      | Rapid eye movement             |
| 9     | Artifact | Excluded from loss computation |

## References

- Perslev, M., Darkner, S., Kempfner, L., Nikolic, M., Jennum, P. J., & Igel, C. (2021). U-Sleep: resilient
  high-frequency sleep staging. *npj Digital Medicine*, 4(1), 72.
