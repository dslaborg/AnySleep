# Experiment Descriptions

This document describes the different experiments and configurations available in this codebase.

## Experiments

### exp001 - USleep
- **exp001a**: Original USleep architecture - U-Net based deep learning model for sleep stage classification

### exp002 - AnySleep
- **exp002a**: AnySleep model with channel attention on skip connections

### exp003 - AnySleep early fusion
- **exp003a**: AnySleep with attention mechanism placed before the encoder

### exp004 - AnySleep late fusion
- **exp004a**: AnySleep with attention mechanism placed after the decoder

## Configuration Structure

Each experiment has its own configuration directory under `config/exp00X/`:
- `{model}_config.yaml`: Base model configuration (architecture, training settings)
- `exp00Xa.yaml`, `exp00Xb.yaml`, etc.: Specific experiment variants

See the `run.md` files in each experiment directory for example training and evaluation commands.
