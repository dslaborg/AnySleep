# AnySleep

Research codebase for automatic sleep stage classification from EEG/EOG signals using deep learning. This framework
implements the U-Sleep and AnySleep architectures and supports training/evaluation at various sampling frequencies on
multiple public sleep datasets.

## Installation

```bash
# Clone the repository
git clone https://github.com/your-username/anysleep.git
cd anysleep

# Install dependencies
pip install -r requirements.txt

# Depending on your system, you might need to install PyTorch manually.
# see https://pytorch.org/get-started/locally/
```

## Inference on EDF Files

For running inference on your own EDF files without the full training pipeline, use the standalone scripts in
`examples/`. You can find trained checkpoints of our models in the [models](models) folder.

1. **Edit the configuration** in `examples/predict_edf_file_logits_plain.py`:

   ```python
   # Path to the input EDF file, e.g. from the Sleep-EDFx dataset
   # (https://physionet.org/content/sleep-edfx/1.0.0/)
   input_eeg = "path/to/your/file.edf"

   # Channel names to extract (must match EDF channel names exactly)
   input_channels = ["EEG Fpz-Cz", "EEG Pz-Oz", "EOG horizontal"]

   # Device for inference
   device = "cuda"  # or "cpu"

   # Predictions per 30-second epoch (1 = standard, higher = finer resolution)
   sleep_stage_frequency = 1
   ```

2. **Run the script**:

   ```bash
   cd examples
   python predict_edf_file_logits_plain.py
   ```

The script handles preprocessing (resampling to 128 Hz, noise clipping, robust scaling) and outputs:

- `prediction_{file_id}.npy`: Raw logits of shape `(1, epochs, 5)`
- A matplotlib figure showing the predicted hypnogram with probability overlay

## Reproducing our results

To reproduce the results from the paper:

1. Prepare the data using the [Sleep Datasets](https://github.com/dslaborg/Sleep-Datasets) repository
2. Follow the instructions in `config/exp00X/run.md` for each experiment, which contain the exact commands for training
   and evaluation

### Data Preparation

Data preprocessing is handled by a separate repository: [Sleep Datasets](https://github.com/dslaborg/Sleep-Datasets).
This repository provides scripts for preprocessing and converting public sleep datasets into the HDF5 format required by
AnySleep.

The framework expects data in HDF5 format with the following structure:

```
processed_data.h5
    ├── Dataset_Name/
    │   ├── Subject_Name/
    │   │   ├── PSG/
    │   │   │   ├── EEG_Channel (e.g., 'F3-M2')
    │   │   │   └── EOG_Channel (e.g., 'E1-M2')
    │   │   ├── hypnogram      # Sleep stage labels per 30s epoch
    │   │   └── class_to_index/  # Indices grouped by sleep stage to filter for all epochs of a given stage
    │   └── ...
    └── ...
```

Sleep stage encoding:

- 0: Wake
- 1: N1
- 2: N2
- 3: N3
- 4: REM
- 9: Artifact (filtered during evaluation)

Update the data path in your config files (e.g., `config/exp001/usleep_config.yaml`):

```yaml
data:
  path: '/path/to/your/processed_data.h5'
```

### Scripts

All training and evaluation scripts are located in the `scripts/` directory. The scripts use configuration files from
the
`config/` directory (see section "Configuration" for details).

#### train.py

Performs model training as specified in the configuration file, writes logs to the console, and saves checkpoints and
results to the `logs/` directory.

Arguments:

- This is a Hydra-based script; any configuration can be overwritten using command line arguments
- `-cn=<experiment>/<sub-experiment>`: name of the experiment to run, for which a `<sub-experiment>.yaml` file must
  exist in the `config/<experiment>` directory

Sample call:

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/train.py -cn=exp001/exp001a
```

#### evaluate.py

Evaluates a trained model on a specified dataset split.

Arguments:

- `-cn=<experiment>/<sub-experiment>`: name of the experiment configuration
- `model.path`: path to the model checkpoint to evaluate; either an absolute path or a relative path inside the
  `models/` folder
- `+training.trainer.evaluators.test`: evaluator configuration for the test set

Sample call:

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/evaluate.py -cn=exp001/exp001a \
    model.path="your_model.pth" \
    +training.trainer.evaluators.test="\${evaluators.test}"
```

#### predict-high-freq.py

Generates per-subject sleep stage predictions with majority voting across channel combinations.

Arguments:

- `-cn=<experiment>/<sub-experiment>`: name of the experiment configuration
- `+high_freq_predict.dataloader`: dataloader configuration for the dataset to predict on
- `model.path`: path to the model checkpoint to evaluate; either an absolute path or a relative path inside the
  `models/` folder
- `model.sleep_stage_frequency`: number of predictions per 30-second epoch (1, 2, 4, ..., 128)

Sample call:

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/predict-high-freq.py -cn=exp002/exp002a \
    +high_freq_predict.dataloader="\${data.test_dataloader}" \
    model.path="your_model.pth" \
    model.sleep_stage_frequency=1
```

Output files:

- `predictions.npz`: Majority-voted predictions per subject
- `labels.npz`: Ground truth labels per subject

#### predict-high-freq_full_logits.py

Generates raw logits (pre-softmax outputs) for each recording and channel combination. Unlike `predict-high-freq.py`,
this preserves the full probability distribution rather than just the predicted class.

Sample call:

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/predict-high-freq_full_logits.py -cn=exp002/exp002a \
    +high_freq_predict.dataloader="\${data.test_dataloader}" \
    model.path="your_model.pth" \
    model.sleep_stage_frequency=1
```

Output files:

- `predictions.npz`: Raw logits per recording (shape per key: `(num_epochs, 5)`)
- `labels.npz`: Ground truth labels per subject

#### predict-confusion-matrix.py

Generates per-recording confusion matrices for detailed error analysis. Computes a 5×5 confusion matrix for each
recording and channel combination.

Sample call:

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/predict-confusion-matrix.py -cn=exp002/exp002a \
    model.path="your_model.pth" \
    +predict_cm.dataloader="\${data.test_dataloader}"
```

Output files:

- `pred_cms.npz`: Confusion matrices per recording (shape per key: `(5, 5)`)

### Experiments

| Experiment | Description                                         |
|------------|-----------------------------------------------------|
| exp001     | Original USleep architecture                        |
| exp002     | AnySleep with channel attention on skip connections |
| exp003     | AnySleepEF (attention before encoder)               |
| exp004     | AnySleepLF (attention after decoder)                |

## Project Structure

```
anysleep/
├── base/                    # Core framework
│   ├── config.py           # Configuration singleton
│   ├── data/               # Dataset classes and dataloaders
│   ├── model/              # Neural network architectures
│   ├── training/           # Training loop and schedulers
│   ├── evaluation/         # Evaluation pipeline
│   └── results/            # Metric trackers
├── config/                  # Hydra configuration files
│   ├── base_config.yaml    # Default settings
│   ├── usleep_split.yaml   # Datasplit used in the U-Sleep study
│   └── exp001-exp004/      # Experiment-specific configs
├── logs/                    # Outputs of the experiments
├── models/                  # Saved model weights
├── scripts/                 # Training, evaluation, and analysis scripts
│   ├── train.py
│   ├── evaluate.py
│   ├── predict*.py
│   ├── arousals/           # Scripts to predict arousals from high-frequency sleep stages
│   ├── final-figures/      # Figure generation scripts
│   └── high-freq/          # Scripts to predict age/sex/OSA from high-frequency sleep stages
├── examples/                # Standalone usage examples
│   ├── anysleep_no_hydra.py              # AnySleep model without Hydra dependencies
│   └── predict_edf_file_logits_plain.py  # Predict sleep stages from EDF files
├── requirements.txt
└── experiments.md          # Experiment descriptions
```

## Configuration

The configuration is implemented using the [Hydra](https://hydra.cc/) framework based on YAML files. If you are not
familiar with Hydra, see the [official documentation](https://hydra.cc/docs/intro) for an introduction.

All configuration files are located in the `config/` directory with the following hierarchy:

- `config/base_config.yaml`: Base configuration for all experiments
- `config/exp00X/<model>_config.yaml`: Model-specific base configuration (e.g., `usleep_config.yaml`,
  `anysleep_config.yaml`)
- `config/exp00X/exp00Xa.yaml`: Sub-experiment configuration that can override parameters from higher levels

Configuration files lower in the hierarchy override parameters from higher-level files. All configurations can also be
overwritten using command line arguments when running a script.

To inspect the final resolved configuration of a run, check the `.hydra/` folder in the output directory after running
a script.

## Citation

If you use this code in your research, please cite:

```bibtex
@article{Grieger2025,
    title = {AnySleep: a channel-agnostic deep learning system for high-resolution sleep staging in multi-center cohorts},
    author = {Niklas Grieger and Jannik Raskob and Siamak Mehrkanoon and Stephan Bialonski},
    year = {2025},
    eprint = {2512.14461},
    archivePrefix = {arXiv},
    url = {https://arxiv.org/abs/2512.14461},
}
```

## License

This project is released under the MIT License. See LICENSE file for details.
