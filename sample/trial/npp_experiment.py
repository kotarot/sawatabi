#!/usr/bin/env python3
# -*- coding: utf-8 -*-

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

import random

import numpy as np

import sawatabi


def npp_experiment(window_size, window_period, repeats, numbers_range_low=1, numbers_range_high=999, local_solver_num_reads=1000, local_solver_num_sweeps=1000, use_prev_states=False, seed=None):
    if seed:
        np.random.seed(seed)
        random.seed(seed)

    # Generate all numbers firstly
    all_size = window_size + window_period * (repeats - 1)
    all_numbers = list(np.random.randint(low=numbers_range_low, high=numbers_range_high, size=all_size))
    #print("All Numbers:", all_numbers)

    prev_states = {}

    # Windowing manually
    for r in range(repeats):
        start = r * window_period
        stop = r * window_period + window_size
        numbers = all_numbers[start:stop]
        # print("Numbers:", numbers)

        model = sawatabi.model.LogicalModel(mtype="ising")
        x = model.variables(name="x", shape=(window_size,))
        for i, m in enumerate(numbers):
            for j, n in enumerate(numbers):
                if i < j:
                    coeff = -1.0 * m * n
                    model.add_interaction(target=(x[i], x[j]), coefficient=coeff)

        ################################
        # With Annealing Solver
        begin = sawatabi.utils.current_time()

        # Local
        solver = sawatabi.solver.LocalSolver(exact=False)
        if not prev_states:
            resultset = solver.solve(model.to_physical(), num_reads=local_solver_num_reads, num_sweeps=local_solver_num_sweeps)
        else:
            initial_states = {}
            for i in range(window_size - window_period):
                initial_states[f"x[{i}]"] = prev_states[f"x[{i + window_period}]"]
                # initial_states[f"x[{i}]"] = prev_states[f"x[{i + window_period}]"] * -1
            for i in range(window_size - window_period, window_size):
                initial_states[f"x[{i}]"] = random.choice([1, -1])
            # print("Initialized with previous states.")
            resultset = solver.solve(model.to_physical(), num_reads=local_solver_num_reads, num_sweeps=local_solver_num_sweeps, initial_states=initial_states)

        # D-Wave
        # solver = sawatabi.solver.DWaveSolver()
        # resultset = solver.solve(model.to_physical(), chain_strength=2, num_reads=100)

        # Optigan
        # solver = sawatabi.solver.OptiganSolver()
        # model.to_qubo()
        # resultset = solver.solve(model.to_physical(), timeout=1000, duplicate=False)

        # print(resultset)

        if use_prev_states:
            prev_states = resultset.first.sample

        diffs = []
        for sample in resultset.samples():
            s_1, s_2 = [], []
            for i, n in enumerate(numbers):
                if sample[f"x[{i}]"] == 1:
                    s_1.append(n)
                elif sample[f"x[{i}]"] in [-1, 0]:
                    s_2.append(n)
            diff = abs(sum(s_1) - sum(s_2))
            # print(f"S_1  : sum={sum(s_1)}, elements={s_1}")
            # print(f"S_2  : sum={sum(s_2)}, elements={s_2}")
            # print("diff :", diff)
            diffs.append(diff)
        diff_min = min(diffs)
        print("diff (min):", diff_min)

        end = sawatabi.utils.current_time()

        ################################
        # With Optimal Solver
        optimal = sawatabi.utils.solve_npp_with_dp(numbers)
        # print(optimal)
        opt_s_1 = [numbers[i] for i in optimal[1]]
        opt_s_2 = [numbers[i] for i in optimal[2]]
        opt_diff = abs(sum(opt_s_1) - sum(opt_s_2))
        print("diff (opt):", opt_diff)

        occurrences_min = diffs.count(diff_min)
        occurrences_opt = diffs.count(opt_diff)
        print(f"occurrences (min): {occurrences_min} / {local_solver_num_reads}")
        print(f"occurrences (opt): {occurrences_opt} / {local_solver_num_reads}")

        # print("Time :", end - begin)
        print("")


def main():
    npp_experiment(window_size=100, window_period=20, repeats=5, use_prev_states=False, seed=12345)
    npp_experiment(window_size=100, window_period=20, repeats=5, use_prev_states=True, seed=12345)


if __name__ == "__main__":
    main()
