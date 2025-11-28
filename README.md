# HF Utils 
HuggingFace utilities for DataDecide datasets and model management.

## Overview 
This package provides utilities for working with HuggingFace datasets and models, particularly for the DataDecide framework. It offers convenient functions to load and process perplexity evaluation results, downstream task evaluation data, and model checkpoints.

## Installation 
```bash 
# Clone the repository
git clone https://github.com/drothermel/hf_utils.git
cd hf_utils

# Install with uv (recommended)
uv sync

# Or install with pip
pip install -e .
```

### Environment Setup 
The package requires environment variables to locate data directories: 
```bash 
# Add to your .env file or shell profile
export DATA_DIR="/path/to/your/data/directory"
export REPO_DIR="/path/to/your/repository/directory"
```

## Core Functionality
### Basic Usage 
```python 
from hf_utils import DataDecide, DataDecidePaths
# Load DataDecide datasets
dd = DataDecide()
# Access dataset paths
paths = DataDecidePaths()
dataset_path = paths.dataset_path("750M")
```

### Load Evaluation Results 
```python 
from hf_utils import (
 load_datadecide_perplexity_results,
 load_datadecide_downstream_results_parsed
)
# Load perplexity evaluation results
ppl_df = load_datadecide_perplexity_results()
# Load downstream task evaluation results
downstream_df = load_datadecide_downstream_results_parsed()
```

## Scripts and Tools This repository includes several utility scripts for data analysis and debugging:
### Standalone Scripts
 - **`scripts/analyze_dataset_comparison.py`**
 - Comprehensive dataset comparison analysis
 - **`scripts/feature_correlation_analysis.py`**
 - Feature correlation analysis across model sizes

### Scripts Requiring ddpred Package
 - **`scripts/debug_shapes.py`**
 - Debug shape misalignment issues in training pipeline
 - **`scripts/debug_window_features.py`**
 - Debug window feature extraction

### Legacy Notebooks
 - **`notebooks/old_ipynbs/simple_preds.py`**
 - Complex data analysis with polynomial fitting
 - **`notebooks/old_ipynbs/test_parsing.py`**
 - Machine learning experimentation with LightGBM

## Dependencies

### Core Dependencies
 - `datasets`
 - HuggingFace datasets library
 - `pandas`
 - Data manipulation and analysis
 - `numpy`
 - Numerical computing
 - `torch`
 - PyTorch for model handling
 - `huggingface_hub`
 - HuggingFace model hub integration

### Optional Dependencies For full functionality of some scripts and notebooks, you may need:
 - `ddpred` package for advanced data processing and ML training
 - `lightgbm` for machine learning experiments
 - `scikit-learn` for preprocessing and metrics

## API Reference

### DataDecide Class 
Main class for loading and managing DataDecide datasets. 
```python 
dd = DataDecide(force_reload=False, verbose=True)
dd.load_df("ppl_parsed_df") # Load specific dataset
```

### DataDecidePaths Class 
Utility class for managing dataset file paths. 
```python 
paths = DataDecidePaths()
paths.dataset_path("750M")    # Get dataset path
paths.parquet_path("results") # Get parquet file path
```

### Loader Functions 
Convenient functions to load specific datasets: 
```python 
# Load parsed perplexity results
ppl_df = load_datadecide_perplexity_results(force_reload=False)
# Load parsed downstream evaluation results
downstream_df = load_datadecide_downstream_results_parsed(force_reload=False)
```

## Migration Notes
This package was extracted from the ddpred repository to create a clean separation between HuggingFace utilities and ML prediction functionality. Scripts and notebooks that require ddpred functionality are clearly marked and include appropriate import error handling.

