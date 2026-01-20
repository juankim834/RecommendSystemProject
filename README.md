# RecommendSystemProject

This is for personal practice project

## What is in this system?

This project is a config-driven recommendation system built for personal practice and exploration of modern recommender systems.

The system is designed to be fully configurable via YAML files, allowing users to define features, model structures, and training strategies without modifying core code. It emphasizes representation learning rather than popularity-based heuristics, and supports both static and sequenctial user behavior modeling.

The project currently includes:

- A configurable two-tower (user tower & item tower) DSSM model
- Unified handling of sparse, dense, and sequential features
- Optional Transformer-based sequence modeling for user behavior
- Support for hard negative sampling during training
- A flexible training pupeline with configurable optimizaiton and early stopping

This repository is intended as a learning-oriented project rather than a production system, with a focus on clarity, extensibility, and experimentaion. 

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
