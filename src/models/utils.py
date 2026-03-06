import math


def _auto_batch_size(n_train: int, bmin: int = 8, bmax: int = 64) -> int:
    """
     for selecting a reasonable batch size based on dataset size.

        - Use approximately 10% of the training set
        - Enforce lower and upper bounds (bmin, bmax)
        - Ensure batch size does not exceed dataset size
        - Ensure batch size is at least 1

    Parameters
    ----------
    n_train : int
        Number of training samples.
    bmin : int
        Minimum allowed batch size.
    bmax : int
        Maximum allowed batch size.

    Returns
    -------
    int
        Selected batch size.
    """
    b = max(bmin, n_train // 10)   # roughly 10% of dataset
    b = min(bmax, b)              # upper bound
    b = min(b, n_train)           # cannot exceed dataset size
    return max(1, b)              # safety lower bound


def compute_log_every(total_steps: int, num_logs: int = 1000) -> int:
    """
    Determine logging frequency for step-based training.

    The goal is to produce approximately `num_logs` log entries
    over `total_steps` optimization steps.

    Parameters
    ----------
    total_steps : int
        Total number of optimizer update steps.
    num_logs : int
        Desired number of logging points.

    Returns
    -------
    int
        Number of steps between two log entries.
    """
    if total_steps is None:
        raise ValueError("total_steps must be specified.")

    if num_logs <= 0:
        raise ValueError("num_logs must be greater than 0.")

    # Spread log entries approximately evenly across total_steps
    return max(1, int(math.ceil(total_steps / num_logs)))