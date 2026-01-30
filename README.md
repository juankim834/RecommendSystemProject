# RecommendSystemProject

## What is in this system?

This project is a config-driven recommendation system built for personal practice and exploration of modern recommender systems.

The system is designed to be fully configurable via YAML files, allowing users to define features, model structures, and training strategies without modifying core code. It emphasizes representation learning rather than popularity-based heuristics, and supports both static and sequential user behavior modeling.

The project currently includes:

- A configurable two-tower (user tower & item tower) DSSM model
- Unified handling of sparse, dense, and sequential features
- Optional Transformer-based sequence modeling for user behavior
- Support for hard negative sampling during training
- A flexible training pipeline with configurable optimization and early stopping

This repository is intended as a learning-oriented project rather than a production system, with a focus on clarity, extensibility, and experimentation.

## 🚀 Executive Summary

This project implements a **Session-based Two-Tower Transformer** architecture that achieves **35.6% Recall@50** on anonymous user sessions (length $\le$ 20), completely without static user profiles.

**Key Achievements:**
* **Solved Cold Start:** Achieved **35.6% Recall@50** on purely anonymous sessions by leveraging Transformer encoders on short interaction sequences.
* **15x Training Speedup:** Engineered a fully vectorized data pipeline, reducing training time for 1M+ samples to **< 30 seconds** on a single GPU.
* **Zero-Overhead Abstraction:** Decoupled feature engineering from model logic via YAML configuration, enabling rapid hypothesis testing without code changes.

## Requirements

This project is developed and tested under the following environment:

- Python 3.11
- PyTorch 2.6.1
- NumPy
- Pandas
- PyYAML
- tqdm

The code is primarily intended for experimentation and learning purposes.  
Environment compatibility beyond the versions listed above is not guaranteed.

## 🚀 Quick Start (Dataset: MovieLens-1M)

This project is configured to run out-of-the-box with the MovieLens 1M dataset. Follow the steps below to reproduce the training results.

### Data Preparation
 1.  Download the **MovieLens 1M Dataset** (`ml-1m.zip`) from [GroupLens](https://grouplens.org/datasets/movielens/1m/).
 2.  Unzip the file and place the data files into the `./data/ml-1m` directory.
 3.  Create a folder named `cleaned` inside `./data` for processed files.
 4.  Run the preprocessing script in `./ml-1m_demo/parsing.py`to generate train/val/test sets:

```bash
mkdir -p data/cleaned
# The script will parse raw .dat files and save processed .pkl to data/cleaned/
python ml-1m_demo/parsing.py
```
### Run the Demo
---
#### Run the Cold Start DEMO
 5. To reproduce the specific **Cold Start results** (35.6% Recall in 26s/epoch) mentioned in the summary:
    - Open the `analysis.ipynb` notebook.
    - Run all cells. It uses the specialized `cold_start_demo.yaml` configuration to demonstrate the high-performance pipeline.
#### Run the Standard Training DEMO (Exploration Mode)
 6. Before running the model, you need to initialize the local configuration files by copying the templates from the `ml-1m_demo/` folder. This ensures the default templates remain unchanged while you customize your local settings.
 ```bash
# Copy templates to root directory
cp ml-1m_demo/config.yaml ./
cp ml-1m_demo/metadata_config.yaml ./
 ```
 7. Run the `train_twotower.py` and start training.
 ```bash
 python train_twotower.py
 ```
## 📚 Acknowledgments
- **Dataset**: [MovieLens 1M](https://grouplens.org/datasets/movielens/1m/) provided by GroupLens Research.
