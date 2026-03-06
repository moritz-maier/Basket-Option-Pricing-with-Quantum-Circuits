from src.models.JaxBaseModel import JaxBaseModel
import pennylane as qml
import jax
import jax.numpy as jnp


class QuantumModel(JaxBaseModel):
    """
    Parameterized quantum model implemented using PennyLane + JAX.

    Architecture:
        - L data re-uploading layers
        - Each layer consists of:
            * Feature encoding block S(x, l)
            * Variational ladder block W_ladder(...)
        - Final measurement: expectation value of PauliZ on the last qubit

    Inherits from JaxBaseModel, which provides:
        - Optimizer (Optax)
        - Training loop
        - Cost function
        - Scaling logic
    """

    # Cache compiled QNodes to avoid recompilation for identical architectures
    _compiled_qnodes = {}

    def __init__(
        self,
        L: int,
        n_trainable_blocks: int,
        seed: int,
        n_features: int,
        learning_rate: float,
        dev=None,
        weights_scaling: int = 1,
        encoding_base: float = 3.0,
    ):
        """
        Parameters
        ----------
        L : int
            Number of data re-uploading layers.
        n_trainable_blocks : int
            Number of variational blocks per layer.
        seed : int
            Random seed for parameter initialization.
        n_features : int
            Number of input features (equals number of qubits).
        learning_rate : float
            Optimizer learning rate.
        dev : optional PennyLane device
            If None, default.qubit simulator is used.
        weights_scaling : float
            Scaling factor for initial weights.
        """

        self.n_features = n_features
        self.n_qubits = self.n_features  # one feature per qubit
        self.L = L
        self.encoding_base = encoding_base
        self.n_trainable_blocks = n_trainable_blocks
        self.dev = dev or qml.device("default.qubit", wires=self.n_qubits)

        # JAX PRNG key for reproducible initialization
        key = jax.random.PRNGKey(seed)

        # Initialize rotation parameters uniformly in [0, 2π)
        uniform_init = jax.random.uniform(
            key,
            shape=(self.L + 1, self.n_trainable_blocks, self.n_qubits, 3),
        )
        weights = weights_scaling * 2 * jnp.pi * uniform_init

        # Use architecture signature as cache key
        compile_key = (self.n_qubits, self.n_trainable_blocks, self.L)

        # Reuse compiled QNode if identical architecture already exists
        if compile_key in QuantumModel._compiled_qnodes:
            self.predict_fn = QuantumModel._compiled_qnodes[compile_key]
        else:

            @qml.qnode(self.dev, interface="jax")
            def qnode_func(weights, x):
                """
                Quantum circuit structure:

                    W_0
                    For l in 0..L-1:
                        S(x, l)
                        W_l+1

                Returns expectation value <Z> on final qubit.
                """
                # Initial variational block
                self.W_ladder(weights[0], self.n_trainable_blocks, self.n_qubits)

                # Data re-uploading structure
                for l in range(self.L):
                    self.S(x, l, self.n_qubits)
                    self.W_ladder(weights[l + 1], self.n_trainable_blocks, self.n_qubits)

                # Measurement
                return qml.expval(qml.PauliZ(wires=self.n_qubits - 1))

            # JIT compile for performance
            jit_fn = jax.jit(qnode_func)
            self.predict_fn = jit_fn
            QuantumModel._compiled_qnodes[compile_key] = jit_fn

        # Initialize parent class (optimizer, cost function, etc.)
        super().__init__(weights=weights, key=key, learning_rate=learning_rate)

    def S(self, x, l, num_wires):
        """
        Feature encoding block.

        Applies RX rotations with polynomial scaling factor 3^l.
        This implements data re-uploading:
            deeper layers encode features at increasing frequency scales.
        """
        for w in range(num_wires):
            qml.RX((self.encoding_base ** l) * x[w], wires=w)

    def W_ladder(self, theta, blocks, num_wires):
        """
        Variational ladder block.

        Structure:
            - Single-qubit rotations (Rot gates)
            - Followed by nearest-neighbor CNOT chain

        Repeated 'blocks' times.
        """
        for b in range(blocks):
            # First qubit rotation
            qml.Rot(
                theta[b, 0, 0],
                theta[b, 0, 1],
                theta[b, 0, 2],
                wires=0,
            )

            # Ladder entangling structure
            for j in range(num_wires - 1):
                qml.CNOT(wires=[j, j + 1])
                qml.Rot(
                    theta[b, j + 1, 0],
                    theta[b, j + 1, 1],
                    theta[b, j + 1, 2],
                    wires=j + 1,
                )