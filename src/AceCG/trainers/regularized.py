# AceCG/trainers/analytic.py
import numpy as np
import copy
from typing import List, Sequence, Tuple, Any

from .analytic import MultiTrainerAnalytic

class Gate(object):

    def __init__(self, k, log_alpha_init=None, beta_k = 0.666, omega_k=-0.1, zeta_k=-0.1):

        if log_alpha_init is not None:
            self.log_alpha = log_alpha_init
        else:
            self.log_alpha = [1 for _ in range(k)]
        self.rng = np.random.default_rng(seed=5)
        self.beta_k = beta_k
        self.omega_k = omega_k
        self.zeta_k = zeta_k
        self.pi = self.sigmoid(self.log_alpha - self.beta_k * np.log(-np.divide(self.omega_k, self.zeta_k)))
        self.u = [None for _ in range(k)]
        self.forward_s = [None for _ in range(k)]
        self.z = [None for _ in range(k)]

    def sigmoid(self, x):

        return 1 / (1 + np.exp(-x))

    def get_determinsitic_values(self):

        s_k = self.sigmoid(self.log_alpha / self.beta_k)
        s_k = np.multiply(s_k, (self.zeta_k - self.omega_k)) + self.omega_k
        return np.maximum(np.zeros_like(s_k), np.minimum(np.ones_like(s_k), s_k))
    def forward(self, step_index: int = 0):

        u_k = self.rng.uniform(size=len(self.z))
        logistic_u_k = np.subtract(np.log(u_k), np.log(np.subtract(np.ones_like(u_k), u_k)))

        s_k = self.sigmoid(np.subtract(logistic_u_k, self.log_alpha) / self.beta_k)
        self.forward_s = s_k
        s_k = np.multiply(s_k, (self.zeta_k - self.omega_k)) + self.omega_k
        self.z = np.maximum(np.zeros_like(s_k), np.minimum(np.ones_like(s_k), s_k))

    def dpi_dlogalpha(self):

        return np.multiply(self.pi, np.subtract(np.ones_like(self.pi, self.pi)))

    def dz_logalpha(self):

        temp = np.multiply(self.forward_s, np.subtract(np.ones_like(self.forward_s), self.forward_s))
        return np.multiply(temp, (self.zeta_k - self.omega_k)/self.beta_k)

class L0MultiTrainerAnalytic(MultiTrainerAnalytic):

    def __init__(self,
        potential,
        optimizer,
        trainer_list: Sequence[BaseTrainer],
        weight_array: np.ndarray,
        beta: float = None,
        logger=None, log_alpha_init=None, beta_k = 0.666, omega_k=-0.1, zeta_k=-0.1
    ):
        super().__init__(potential, optimizer, trainer_list, weight_array[:-1], beta, logger)

        k = len(potential)
        self.gate = Gate(k, log_alpha_init=log_alpha_init, beta_k=beta_k, omega_k=omega_k, zeta_k=zeta_k)
        self.set_optimizer()
        self.weights = weight_array


    def get_modified_potential(self):

        if self.gate.z[0] is None:
            raise("The gate is None. Must run the gate forward() step to establish Z values.")

        potential = cp.deepcopy(self.potential)

        i = 0
        for pair, pot in potential.items():
            pot = pot.get_scaled_potential(self.gate.z[i])
            i+=1

        return potential

    def update_potential(self, L_new: np.ndarray):

        self.optimizer.L = L_new.copy()
        self.set_optimizer()

    def step(self, param_list: Sequence[Tuple[Any, ...]], return_idx_list: Sequence[Sequence[int]]):
        """
        Run one combined step over all sub-trainers and apply a weighted update.

        Each sub-trainer is called as:
            result_i = trainers[i].step(*param_list[i])

        We collect:
        - the **last** element of each result (assumed to be that trainer's update),
        - any additional elements specified by `return_idx_list[i]`.

        Then compute the final update as a weighted combination of sub-updates:
            final_update = weights @ stack(updates)      # (n_params,)

        Apply this to `self.optimizer.L`, append `final_update` to the returned
        list, and finally call `clamp_and_update()` to synchronize parameters.

        Parameters
        ----------
        param_list : list[tuple]
            A list with length equal to `len(self.trainers)`. Entry i is the
            positional-argument tuple passed to `self.trainers[i].step(...)`.
        return_idx_list : list[list[int]]
            A list of index lists. For trainer i, we keep `result[i][j]` for
            each `j` in `return_idx_list[i]`. These are appended to the final
            return list in order. The **last** element of each trainer’s result
            is assumed to be its `update` vector and is *always* consumed to
            build the combined update (you do not need to list it here).

        Returns
        -------
        selected_and_update : list
            A flat list of selected items from each trainer’s returns (in the
            order specified by `return_idx_list`), with the **final combined
            update** appended at the end. The combined update has shape
            `(n_params,)` and has already been applied to `self.optimizer.L`.

        Raises
        ------
        AssertionError
            If lengths/shapes/types are inconsistent, or if updates returned by
            sub-trainers do not match the meta-optimizer parameter shape.

        Notes
        -----
        - All sub-trainers must return an update vector as the **last** item
          from their `.step(...)`.
        - The stacking/weighting assumes all updates share the same shape as
          `self.optimizer.L`.
        """
        param_gradients = np.zeros_like(self.get_params())
        gate_gradients = np.zeros_like(self.gate.log_alpha)

        for i, trainer in self.trainers:
            trainer.set_scale_factors(self.gate.z)
            param_gradients = np.add(self.weights[i] * param_gradients, trainer.get_gradients(*param_list[i]))
            trainer.unset_scale_factors()
            d_dlogalpha = np.multiply(trainer.d_dz(*param_list[i]), self.gate.dz_logalpha())
            gate_gradients = np.add(gate_gradients, self.weights[i] * d_dlogalpha)

        gate_gradients = np.add(gate_gradients, self.weights[-1]*self.gate.dpi_dlogalpha())

        total_gradients = param_gradients.extend(gate_gradients)

        final_update = self.optimizer.step(total_gradients)

        # === clamp & sync meta parameters ===
        self.clamp_and_update()

        # === sync sub-trainers' potentials with the new global L ===
        # (Ensure consistent L across sub-trainers after meta update)
        for tr in self.trainers:
            tr.update_potential(self.optimizer.L[:len(self.get_params())])
        self.gate.log_alpha = self.optimizer.L[len(self.get_params()):]

        return final_update


    def get_gated_potential(self):

        assert  self.gate.z[0] is not None
        self.get_scaled_potential(self.gate.z)

    def forward(self):
        self.gate.forward()

    def set_optimizer(self):

        correct_length = len(self.get_parans()) + len(self.gate.log_alpha)

        if len(self.optimizer.L) != correct_length:
            if len(self.optimizer.L) == len(self.get_params()):
                self.optimizer.L.extend(self.gate.log_alpha)
                self.optimizer.mask.extend([False for _ in range(len(self.gate.log_alpha))])
            else:
                raise ValueError("Optimizer of length " + str(len(self.optimizer.L)) + "is in an illegal state.")









