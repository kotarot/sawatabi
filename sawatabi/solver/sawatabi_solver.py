# Copyright 2021 Kotaro Terada
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# import logging
import math
import time

import dimod
import numpy as np

import sawatabi.constants as constants
from sawatabi.model.physical_model import PhysicalModel
from sawatabi.solver.abstract_solver import AbstractSolver

# logger = logging.getLogger(__name__)


class SawatabiSolver(AbstractSolver):
    def __init__(self):
        self._model = None
        self._original_bqm = None
        self._bqm = None
        self._rng = None
        super().__init__()

    def solve(
        self,
        model,
        num_reads=1,
        num_sweeps=100,
        cooling_rate=0.9,
        initial_temperature=100.0,
        initial_states=None,
        reverse_options=None,
        pickup_mode=constants.PICKUP_MODE_RANDOM,
        seed=None,
        need_stats=False,
    ):
        self._check_argument_type("model", model, PhysicalModel)

        if len(model._raw_interactions[constants.INTERACTION_LINEAR]) == 0 and len(model._raw_interactions[constants.INTERACTION_QUADRATIC]) == 0:
            raise ValueError("Model cannot be empty.")

        if initial_states and (len(initial_states) != num_reads):
            raise ValueError("Length of initial_states must be the same as num_reads.")

        allowed_pickup_mode = [constants.PICKUP_MODE_RANDOM, constants.PICKUP_MODE_SEQUENTIAL]
        if pickup_mode not in allowed_pickup_mode:
            raise ValueError(f"pickup_mode must be one of {allowed_pickup_mode}")

        if reverse_options:
            self._check_argument_type("reverse_options", reverse_options, dict)
            if "reverse_period" not in reverse_options:
                raise ValueError("reverse_options must contain 'reverse_period'")
            if "reverse_temperature" not in reverse_options:
                raise ValueError("reverse_options must contain 'reverse_temperature'")

        # Use a rangom generator so that this random sequence is isolated
        if seed:
            self._rng = np.random.default_rng(seed)
        else:
            self._rng = np.random.default_rng()

        bqm = model.to_bqm(sign=1.0)
        self._original_bqm = bqm

        # To Ising model for SawatabiSolver annealing process
        if bqm.vartype is not dimod.SPIN:
            bqm = bqm.change_vartype(dimod.SPIN, inplace=False)

            # Convert initial states as well
            if initial_states:
                for i, initial_state in enumerate(initial_states):
                    for k, v in initial_state.items():
                        if v == 0:
                            initial_states[i][k] = -1

        self._model = model
        self._bqm = bqm

        # For speed up, store coefficients into a list (array)
        self._bqm_linear = [0.0 for _ in range(self._bqm.num_variables)]
        for label, coeff in self._bqm.linear.items():
            index = self._model._label_to_index[label]
            self._bqm_linear[index] = coeff
        self._bqm_adj = [{} for _ in range(self._bqm.num_variables)]
        for label, adj in self._bqm.adj.items():
            index = self._model._label_to_index[label]
            adj_dict = {}
            for alabel, coeff in adj.items():
                aindex = self._model._label_to_index[alabel]
                adj_dict[aindex] = coeff
            self._bqm_adj[index] = adj_dict

        start_sec = time.perf_counter()

        samples = []
        energies = []
        stats = []
        for r in range(num_reads):
            initial_state_for_this_read = None
            if initial_states:
                initial_state_for_this_read = initial_states[r]
            sample, energy, energy_hist, temperature_hist, acceptance_hist = self.annealing(
                num_reads=num_reads,
                num_sweeps=num_sweeps,
                cooling_rate=cooling_rate,
                initial_temperature=initial_temperature,
                initial_state=initial_state_for_this_read,
                reverse_options=reverse_options,
                pickup_mode=pickup_mode,
            )
            # These samples and energies are in the Ising (SPIN) format
            samples.append(sample)
            energies.append(energy)
            stats.append(
                {
                    "energy_history": energy_hist,
                    "temperature_history": temperature_hist,
                    "acceptance_history": acceptance_hist,
                }
            )

        # Update the timing
        execution_sec = time.perf_counter() - start_sec

        sampleset = dimod.SampleSet.from_samples(samples, vartype=dimod.SPIN, energy=energies, aggregate_samples=True, sort_labels=True)
        sampleset._info = {
            "timing": {
                "execution_sec": execution_sec,
            },
        }

        sampleset = sampleset.change_vartype(self._original_bqm.vartype, inplace=True)
        if not need_stats:
            return sampleset
        return sampleset, stats

    def annealing(self, num_reads, num_sweeps, cooling_rate, initial_temperature, initial_state, reverse_options, pickup_mode):
        num_variables = self._bqm.num_variables
        if initial_state is None:
            x = ((self._rng.integers(2, size=num_variables) - 0.5) * 2).astype(int)  # -1 or +1
        else:
            x = np.ones(shape=(num_variables), dtype=int)
            for v in self._bqm.variables:
                idx = self._model._label_to_index[v]
                x[idx] = initial_state[v]

        initial_sample = dict(zip(list(self._model._index_to_label.values()), x))
        # logger.info(f"initial_spins: {initial_sample}")
        initial_energy = self._bqm.energy(initial_sample) * -1.0  # Note that the signs of original bqm is opposite from ours
        # logger.info(f"initial_energy: {initial_energy}")

        if not reverse_options:
            # Forward (normal) annealing
            temperature = initial_temperature
        else:
            # Reverse annealing
            temperature = 1e-9
            reverse_target_temperature = reverse_options["reverse_temperature"]  # The max temperature when the phase is reverse annealing

        energy = initial_energy
        reversing_phase = reverse_options is not None
        sweep = 0

        energy_hist = []
        temperature_hist = []
        acceptance_hist = []

        # Create a random values for accept beforehand for speed up
        self._accept_randoms = self._rng.random(size=num_sweeps * num_variables)
        self._accept_randoms_idx = -1

        for sweep in range(num_sweeps):  # outer loop (=sweeps)
            # Normal annealing in the last half of period if reverse annealing is performed
            if reversing_phase and (reverse_options["reverse_period"] <= sweep):
                reversing_phase = False

            # logger.info(f"sweep: {sweep + 1}/{num_sweeps}  (temperature: {temperature}, reversing_phase: {reversing_phase})")

            energy_hist.append(energy)
            temperature_hist.append(temperature)

            # Pick up a spin (variable) randomly
            if pickup_mode == constants.PICKUP_MODE_RANDOM:
                pickups = self._rng.permutation(num_variables)
            # Pick up a spin (variable) sequentially
            elif pickup_mode == constants.PICKUP_MODE_SEQUENTIAL:
                pickups = np.arange(num_variables)

            acceptances = 0
            for inner, idx in enumerate(pickups):  # inner loop
                # logger.debug(f"inner: {inner + 1}/{num_variables}  (pickuped: {idx})")

                # `diff` represents an energy value gained after flipping
                diff = self.calc_energy_diff(idx, x)

                if self.is_acceptable(diff, temperature):
                    x[idx] *= -1
                    energy += diff
                    acceptances += 1
                    # logger.debug(f"Spin {self._model._index_to_label[idx]} was flipped to {x[idx]}")
                # logger.debug(f"energy: {energy}")

            acceptance_hist.append(acceptances)

            if reversing_phase:
                reverse_target_temperature *= cooling_rate
                temperature = reverse_options["reverse_temperature"] - reverse_target_temperature
            else:
                temperature *= cooling_rate

        sample = dict(zip(list(self._model._index_to_label.values()), x))

        # Check energy if needed
        # recalc_energy = self._bqm.energy(sample) * -1.0
        # assert math.isclose(energy, recalc_energy, rel_tol=1e-9, abs_tol=1e-9)

        # Deal with offset
        energy += self._original_bqm.offset * 2

        return sample, energy, energy_hist, temperature_hist, acceptance_hist

    def calc_energy_diff(self, idx, x):
        # h_{i}
        diff = x[idx] * self._bqm_linear[idx]

        # J_{ij}
        for aidx, j in self._bqm_adj[idx].items():
            diff += x[idx] * x[aidx] * j

        # Now the calculated diff is the local energy at x[idx].
        # If the spin flips from -1 to +1 (vice versa), the diff energy will be double.
        return 2.0 * diff

    def is_acceptable(self, diff, temperature):
        """
        Returns True if the flip is acceptable, False otherwise.
        """
        if diff <= 0.0:
            return True
        p = math.exp(-diff / temperature)  # Note: np.exp is slow here
        self._accept_randoms_idx += 1
        if self._accept_randoms[self._accept_randoms_idx] < p:
            return True
        return False
