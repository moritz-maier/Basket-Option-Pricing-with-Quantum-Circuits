# Basket Option Pricing with Quantum Circuits

This repository contains the implementation for the bachelor thesis:

**“Basket Option Pricing with Quantum Circuits”**  
Ludwig-Maximilians-Universität München (LMU), 2026  
Author: Moritz Maier  

The full thesis document is included in this repository:

[Thesis PDF](latex/text/main.pdf)

---

## Overview

The objective of this project is the approximation of basket option prices using:

- Classical neural networks (Culkin and Ferguson architectures)
- Variational Quantum Circuits (VQCs)

The study compares the models under different experimental conditions:

- Different payoff structures:
  - Worst-of
  - Best-of
  - Average
- Varying training dataset sizes
- Artificial noise
- Random vs. temporally ordered train-test splits
- Spectral analysis of the target functions using NUFFT


---

## Repository Structure

```
config/
    main.py
data/
    processed/
    yf/
latex/
    text/
notebooks/
    01_generate_data.ipynb
    02_run_single_experiment.ipynb
    03_results_analysis.ipynb
results/
slurm/
    run_array.sh
    run_single.sh
src/
    data_generation/
        compute_basket_price.py
        Data.py
        DataGenerator.py
        DataManager.py
        MLData.py
        utils_data.py
    fourier_analysis/
        compute_spectrum.py
        plot.py
    
    models/
        ClassicMLModel.py
        DataScaler.py
        JaxBaseModel.py
        protocols.py
        QuantumModel.py
        utils.py
    run/
        params.py
        pipeline.py
        Result.py
        result_repository.py
        RunParams.py
    visualize/
        metrics.py
        plot_utils.py
        visualize.py
    paths.py
README.md
requirements.txt
```

---

## Clone Repository

The experiment results are stored in a separate GitLab repository and included in this project as a Git submodule in the `results/` directory.

Results repository:
https://gitlab2.cip.ifi.lmu.de/maiermo/basket-option-pricing-with-quantum-circuits-results

### Clone without results

If you only want the code and repository structure, you can clone the repository normally:

```bash
git clone https://github.com/moritz-maier/Basket-Option-Pricing-with-Quantum-Circuits.git
cd Basket-Option-Pricing-with-Quantum-Circuits
```

In this case, the `results/` directory will be empty.

---

### Clone including results

To clone the repository **including the experiment results**, use:

```bash
git clone --recurse-submodules https://github.com/moritz-maier/Basket-Option-Pricing-with-Quantum-Circuits.git
cd Basket-Option-Pricing-with-Quantum-Circuits
```

This command automatically downloads the `results/` submodule from the GitLab repository.

---

### Download results after cloning

If you already cloned the repository without submodules, you can download the results afterwards with:

```bash
git submodule update --init --recursive
```

This will fetch the `results/` repository and populate the `results/` directory.

---

## Installation

Create a virtual environment:

```bash
python -m venv venv
source venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```


---

## Running Experiments

### Generate Data

The dataset can be generated using:

```
notebooks/01_generate_data.ipynb
```

This notebook performs:

- Historical data loading
- Volatility and correlation estimation
- Monte Carlo basket price generation
- Dataset storage in `data/processed/`

---

### Run a Single Experiment

A single experiment can be executed using:

```
notebooks/02_run_single_experiment.ipynb
```

This is useful for:

- Testing model configurations
- Debugging
- Inspecting individual seeds
- Visualizing predictions

---

### Run All Experiments

All experiments can be created and executed via:

```
config/main.py
```

This script:

- Generates the full experiment grid (all model configurations, seeds, noise levels, dataset sizes, etc.)
- Splits the experiment configurations into independent jobs
- Allows parallel execution across multiple processes or compute nodes

For large experiment grids, it is strongly recommended to execute the experiments via SLURM array jobs:

```
slurm/run_array.sh
```

To launch the SLURM array job, use for example:

```bash
sbatch --array=1-30%15 slurm/run_array.sh
```

The array setup distributes experiment configurations across multiple jobs, enabling efficient large-scale execution on HPC clusters.

**Note:**  
Depending on your cluster setup, you may need to adjust:

- The Python environment path
- The virtual environment activation
- The project root path
- Resource specifications (time, memory, CPU/GPU)

inside the SLURM scripts.

---

### Downloading Results from the Cluster

From the repository root:

```bash
cd results/
```

Synchronize results from the cluster:

```bash
rsync -av --progress -e "ssh -J USER@remote.cluster.domain" \
USER@compute-node:<remote-project-path>/results/ ./
```

Replace:

- `USER` with your cluster username
- `remote.cluster.domain` with your login/jump host (if required)
- `<remote-project-path>` with the path where the repository is located on the cluster

---

### Analyze Results

After running experiments, results can be analyzed using:

```
notebooks/03_results_analysis.ipynb
```

This notebook provides:

- Aggregated R^2 statistics
- Boxplots and comparison figures
- Noise and dataset-size analysis
- Visualization of temporal splits

---

## Included Dataset and Results

The repository includes:

- The processed dataset used in the thesis (`data/processed/`)
- All experiment results stored in the `results/` directory.

The results are stored in a separate Git repository and included here as a Git submodule.

If the `results/` directory is empty after cloning, run:

```bash
git submodule update --init --recursive
```

**Note** 

The experiment results are stored in a separate GitLab repository hosted by LMU.

Since this repository is tied to a university account, long-term availability of the results repository cannot be guaranteed indefinitely. If the results repository becomes unavailable in the future, the project can still be reproduced by generating the dataset and rerunning the experiments using the provided code.