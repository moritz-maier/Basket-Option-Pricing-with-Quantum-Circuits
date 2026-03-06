import argparse
import random
from typing import List

from src.run.params import Params, DataParams, QuantumModelParams, TrainingParams, ClassicModelParams
from src.data_generation.utils_data import TRADING_DAYS
from src.run.Result import save_result
from src.run.RunParams import run_params


# ---------------------------------------------------------------------
# SLURM batch runner:
# - Builds a large grid of experiment configurations ("combos")
# - Distributes combos across MAX_JOBS jobs (e.g., SLURM array tasks)
# - For each job: runs its assigned combos for multiple random seeds
# - Saves each result into a structured results folder
# ---------------------------------------------------------------------

MAX_JOBS = 30                       # Number of job buckets (e.g., SLURM array size)
subdir = "slurm"            # Subfolder inside results/ for this experiment batch

# ---------------------------------------------------------------------
# Reproducible list of seeds: each combo is evaluated with n_seeds seeds
# ---------------------------------------------------------------------
random.seed(12345)
n_seeds = 30
seeds = [random.randint(0, 2**32 - 1) for _ in range(n_seeds)]

# ---------------------------------------------------------------------
# Hyperparameter grid definition
# ---------------------------------------------------------------------

tickers_list = [["IBM", "NKE"]]
start_date_list = ["2000-01-01"]
end_date_list = ["2025-01-01"]

# Train/test split type (MLData supports "random" (grouped) and "time")
split_mode_list = ["random" ,"time"]

# Number of training samples drawn from the training set
n_samples_list = [25, 50, 100, 250, 500, 1_000, 25_000]

use_percentage = True
option_type_list = ["best", "worst", "average"]

# Maturity (in trading days)
maturity = TRADING_DAYS / 4

# Noise added to y_train (multiplicative noise in MLData)
noiseScale_list = [0.0, 0.05, 0.1, 0.25, 0.5]

# Training settings
epochs_list = [None]               # using total_steps, so epochs=None
learning_rate = 0.001
batch_size = None
total_steps = 5_000

# Model variants to run
model_list = ["quantum", "classic_ferguson", "classic_culkin"]

# Classic model grids
units_list_ferguson = [100, 5, 9, 14]
units_list_culkin = [100]

dropout_list_ferguson = [0.0]
dropout_list_culkin = [0.25, 0]

# Quantum model grids
encoding_base_unary = 1
layers_list_unary = [1, 3, 4, 5]
n_trainable_blocks_list_unary = [10, 10, 17, 25]
quantum_pairs_unary = list(zip(n_trainable_blocks_list_unary, layers_list_unary))

encoding_base_ternary = 3
layers_list_ternary = [2, 3]
n_trainable_blocks_list_ternary = [20, 30]
quantum_pairs_ternary = list(zip(n_trainable_blocks_list_ternary, layers_list_ternary))


quantum_pairs_all = (
    [(encoding_base_unary, B, L) for B, L in (quantum_pairs_unary or [])]
    + [(encoding_base_ternary, B, L) for B, L in (quantum_pairs_ternary or [])]
)
print(quantum_pairs_all)



def create_all_combos():
    """
    Create the full list of configuration dictionaries ("combos")
    from the grid above.

    Each combo corresponds to one experiment *before* seeding.
    Seeds are applied later to replicate the combo multiple times.
    """
    combos = []
    for split_mode in split_mode_list:
        for start_date in start_date_list:
            for end_date in end_date_list:
                for n_samples in n_samples_list:
                    for tickers in tickers_list:
                        for option_type in option_type_list:
                            for noiseScale in noiseScale_list:
                                for epochs in epochs_list:
                                    for model in model_list:
                                        if model == "classic_ferguson":
                                            for units in units_list_ferguson:
                                                for dropout in dropout_list_ferguson:
                                                    combos.append({
                                                        "end_date": end_date,
                                                        "tickers": tickers,
                                                        "start_date": start_date,
                                                        "option_type": option_type,
                                                        "noiseScale": noiseScale,
                                                        "epochs": epochs,
                                                        "model": model,
                                                        "units": units,
                                                        "dropout": dropout,
                                                        "n_trainable_blocks": None,
                                                        "prefactors": None,
                                                        "n_samples": n_samples,
                                                        "split_mode": split_mode,
                                                    })

                                        elif model == "classic_culkin":
                                            for units in units_list_culkin:
                                                for dropout in dropout_list_culkin:
                                                    combos.append({
                                                        "end_date": end_date,
                                                        "tickers": tickers,
                                                        "start_date": start_date,
                                                        "option_type": option_type,
                                                        "noiseScale": noiseScale,
                                                        "epochs": epochs,
                                                        "model": model,
                                                        "units": units,
                                                        "dropout": dropout,
                                                        "n_trainable_blocks": None,
                                                        "prefactors": None,
                                                        "n_samples": n_samples,
                                                        "split_mode": split_mode,
                                                    })

                                        elif model == "quantum":
                                            for encoding_base, n_trainable_blocks, layers in quantum_pairs_all:
                                                combos.append({
                                                    "end_date": end_date,
                                                    "tickers": tickers,
                                                    "start_date": start_date,
                                                    "option_type": option_type,
                                                    "noiseScale": noiseScale,
                                                    "epochs": epochs,
                                                    "model": model,
                                                    "units": None,
                                                    "dropout": None,
                                                    "encoding_base": encoding_base,
                                                    "n_trainable_blocks": n_trainable_blocks,
                                                    "layers": layers,
                                                    "n_samples": n_samples,
                                                    "split_mode": split_mode,
                                                })

    return combos


def group_combos(combos):
    """
    Group combos so that similar model configurations stay together.
    """
    groups = {}

    for c in combos:
        if c["model"] == "classic":
            key = ("classic", c["units"], c["dropout"])
        else:
            key = ("quantum", c["n_trainable_blocks"])

        groups.setdefault(key, []).append(c)

    return list(groups.values())


def distribute_to_jobs(groups, n_jobs):
    """
    Distribute grouped combos to job buckets.

    Result:
      job_buckets[j] is a list of combos assigned to job index j.
    """
    job_buckets = [[] for _ in range(n_jobs)]

    idx = 0
    for group in groups:
        for combo in group:
            job_buckets[idx % n_jobs].append(combo)
            idx += 1

    return job_buckets


def create_configs(job_id: int) -> List[Params]:
    """
    Create the list of Params objects for a single job.

    job_id is assumed to be 1-based:
      job_id=1 selects the first bucket, job_id=2 the second, ...

    For each selected combo:
      create one Params per seed (replicated experiments).
    """
    combos = create_all_combos()
    print(f"{len(combos)} combos")

    groups = group_combos(combos)
    distributed = distribute_to_jobs(groups, MAX_JOBS)

    # job_id is 1-based; list indices are 0-based
    selected = distributed[job_id - 1]

    params_list: List[Params] = []

    for combo in selected:
        for seed in seeds:
            # ---- Data parameters (dataset + split configuration) ----
            dataParams = DataParams(
                tickers=combo["tickers"],
                start_date=combo["start_date"],
                end_date=combo["end_date"],
                use_percentage=use_percentage,
                option_type=combo["option_type"],
                maturity=maturity,
                noiseScale=combo["noiseScale"],
                n_samples=combo["n_samples"],
                split_mode=combo["split_mode"],
            )

            # ---- Training parameters (epochs or total_steps) ----
            trainingParams = TrainingParams(
                epochs=combo["epochs"],
                learning_rate=learning_rate,
                batch_size=batch_size,
                total_steps=total_steps,
            )

            # ---- Model parameters depend on model type ----
            if "classic" in combo["model"].lower():
                modelParams = ClassicModelParams(
                    units=combo["units"],
                    dropout=combo["dropout"],
                    model_name=combo["model"],
                )
            else:
                modelParams = QuantumModelParams(
                    layers=combo["layers"],
                    n_trainable_blocks=combo["n_trainable_blocks"],
                    encoding_base= combo["encoding_base"],
                )

            # ---- Final experiment params bundle ----
            params = Params(
                seed=seed,
                data=dataParams,
                model_params=modelParams,
                training_params=trainingParams,
            )

            params_list.append(params)

    return params_list


def main():
    """
    Entrypoint for running one job's worth of experiments.

    Typical SLURM usage:
      python -m configs.main --config $SLURM_ARRAY_TASK_ID

    """
    parser = argparse.ArgumentParser(description="Select which config set to run.")
    parser.add_argument(
        "--config",
        type=int,
        choices=range(1, 100),
        default=1,
        help="Which job bucket to run (typically the SLURM array task ID).",
    )
    args = parser.parse_args()

    configs = create_configs(args.config)
    print(f"{len(configs)} configs selected")
    print(f"This: {len(configs)}")

    # Run each Params configuration sequentially in this job
    for i, config in enumerate(configs, start=1):
        try:
            print(config)

            # Run training + evaluation
            result = run_params(config)

            # Persist result (metrics + weights) to results/<subdir>/...
            save_result(result, subdir)

        except Exception as e:
            # Continue with the next config if one fails
            print("-" * 80)
            print(f"Fehler in config {i}: {e}")
            print("-" * 80)
            continue


if __name__ == "__main__":
    main()