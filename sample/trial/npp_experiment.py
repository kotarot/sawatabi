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

import datetime
import math
import random

import matplotlib.pyplot as plt
import numpy as np

import sawatabi

# Default font size is 12
plt.rcParams['font.size'] = 14


def run(window_size, window_period, batches, numbers_range_lower=1, numbers_range_upper=99, local_solver_num_reads=1000, local_solver_num_sweeps=100, use_prev_states=False, seed=None):
    if seed:
        np.random.seed(seed)
        random.seed(seed)

    # Generate all numbers firstly
    all_size = window_size + window_period * (batches - 1)
    all_numbers = list(np.random.randint(low=numbers_range_lower, high=numbers_range_upper, size=all_size))
    #all_numbers = list(range(1, all_size + 1))
    #np.random.shuffle(all_numbers)
    #print("All Numbers:", all_numbers)

    tts_history = []
    prev_states_samples = None

    # Windowing manually
    for r in range(batches):
        start = r * window_period
        stop = r * window_period + window_size
        numbers = all_numbers[start:stop]
        #print("Numbers:", numbers)

        model = sawatabi.model.LogicalModel(mtype="ising")
        x = model.variables(name="x", shape=(window_size,))
        for i, m in enumerate(numbers):
            for j, n in enumerate(numbers):
                if i < j:
                    coeff = -1.0 * m * n
                    model.add_interaction(target=(x[i], x[j]), coefficient=coeff)
        #print(model)

        ################################
        # With Annealing Solver
        begin = sawatabi.utils.current_time()

        # Local
        solver = sawatabi.solver.LocalSolver(exact=False)
        if not prev_states_samples:
            resultset = solver.solve(model.to_physical(), num_reads=local_solver_num_reads, num_sweeps=local_solver_num_sweeps, seed=1 * (r + 1))
        else:
            initial_states = []
            for sample in prev_states_samples:
                this_initial_states = {}
                for i in range(window_size - window_period):
                    this_initial_states[f"x[{i}]"] = sample[f"x[{i + window_period}]"]
                for i in range(window_size - window_period, window_size):
                    this_initial_states[f"x[{i}]"] = random.choice([1, -1])
                initial_states.append(this_initial_states)
            #print("Initialized with previous states.")
            resultset = solver.solve(model.to_physical(), num_reads=local_solver_num_reads, num_sweeps=local_solver_num_sweeps, initial_states=initial_states, initial_states_generator="none", seed=2 * (r + 1))

        # D-Wave
        #solver = sawatabi.solver.DWaveSolver()
        #resultset = solver.solve(model.to_physical(), chain_strength=2, num_reads=100)

        # Optigan
        #solver = sawatabi.solver.OptiganSolver()
        #model.to_qubo()
        #resultset = solver.solve(model.to_physical(), timeout=1000, duplicate=False)

        #print(resultset)

        if use_prev_states:
            prev_states_samples = resultset.samples()

        diffs = []
        for sample in resultset.samples():
            s_1, s_2 = [], []
            for i, n in enumerate(numbers):
                if sample[f"x[{i}]"] == 1:
                    s_1.append(n)
                elif sample[f"x[{i}]"] in [-1, 0]:
                    s_2.append(n)
            diff = abs(sum(s_1) - sum(s_2))
            #print(f"S_1  : sum={sum(s_1)}, elements={s_1}")
            #print(f"S_2  : sum={sum(s_2)}, elements={s_2}")
            #print("diff :", diff)
            diffs.append(diff)
        diff_min = min(diffs)
        #print("diff (min):", diff_min)

        end = sawatabi.utils.current_time()

        ################################
        # With Optimal Solver
        optimal = sawatabi.utils.solve_npp_with_dp(numbers)
        # print(optimal)
        opt_s_1 = [numbers[i] for i in optimal[1]]
        opt_s_2 = [numbers[i] for i in optimal[2]]
        opt_diff = abs(sum(opt_s_1) - sum(opt_s_2))
        #print("diff (opt):", opt_diff)

        occurrences_min = diffs.count(diff_min)
        occurrences_opt = diffs.count(opt_diff)
        #print(f"occurrences (min): {occurrences_min} / {local_solver_num_reads}")
        #print(f"occurrences (opt): {occurrences_opt} / {local_solver_num_reads}")

        # TTS (time to solution) in sweeps unit
        P = 0.99
        p = occurrences_opt / local_solver_num_reads
        if 0.999 < p:
            p = 0.999
        if p < 0.001:
            p = 0.001
        tts = local_solver_num_sweeps * math.log(1 - P) / math.log(1 - p)
        #print("TTS:", tts)
        tts_history.append(tts)

        # print("Time :", end - begin)
        # print("")

    return tts_history


def experiment1(window_size, window_period, batches):
    NUMBERS_RANGE_UPPER = 99
    LOCAL_SOLVER_NUM_SWEEPS = 100

    print(f"[{datetime.datetime.today()}] Experimenting window_size={window_size} window_period={window_period} batches={batches} ...")

    def calc_average_and_standard_error(histories):
        result = {
            "average": [],
            "standard_error": [],
        }
        for i, _ in enumerate(histories[0]):
            data = []
            for h in histories:
                data.append(h[i])
            result["average"].append(np.average(data))
            result["standard_error"].append(np.std(data, ddof=1) / np.sqrt(len(data)))
        return result

    # Run annealing againt 10 different problem seed (for "without previous states")
    print("  Running againt 10 different problems without previous states ...")
    tts_hists_without_states = []
    for seed in range(10, 110, 10):
        tts_hist_without_states = run(window_size=window_size, window_period=window_period, batches=batches, numbers_range_upper=NUMBERS_RANGE_UPPER, local_solver_num_sweeps=LOCAL_SOLVER_NUM_SWEEPS, use_prev_states=False, seed=seed)
        #print(tts_hist_without_states)
        tts_hists_without_states.append(tts_hist_without_states)
    tts_result_without_states = calc_average_and_standard_error(tts_hists_without_states)
    #print(tts_result_without_states)
    #print("")

    # Run annealing againt 10 different problem seed (for "with previous states")
    print("  Running againt 10 different problems with previous states ...")
    tts_hists_with_states = []
    for seed in range(10, 110, 10):
        tts_hist_with_states = run(window_size=window_size, window_period=window_period, batches=batches, numbers_range_upper=NUMBERS_RANGE_UPPER, local_solver_num_sweeps=LOCAL_SOLVER_NUM_SWEEPS, use_prev_states=True, seed=seed)
        #print(tts_hist_with_states)
        tts_hists_with_states.append(tts_hist_with_states)
    tts_result_with_states = calc_average_and_standard_error(tts_hists_with_states)
    #print(tts_result_with_states)
    #print("")

    # Plot chart for comparison
    fig = plt.figure(figsize=(10, 6))
    ax = plt.axes()
    xdata = list(range(1, batches + 1))
    plt.errorbar(xdata, tts_result_without_states["average"], yerr=tts_result_without_states["standard_error"], marker="o", linewidth=2, capsize=5)
    plt.errorbar(xdata, tts_result_with_states["average"], yerr=tts_result_with_states["standard_error"], marker="o", linewidth=2, capsize=5)
    #ax.yaxis.set_major_formatter(ScalarFormatter(useMathText=True))
    #ax.ticklabel_format(style="sci", axis="y", scilimits=(0, 0))
    plt.ylim(ymin=0)
    plt.xlabel("Iteration")
    plt.ylabel("TTS (sweeps)")
    plt.legend(["w/o previous states", "w/ previous states"])
    plt.title(f"Using previous states (window_size={window_size} window_period={window_period})")
    plt.savefig(f"use_prev_states_size{window_size}_period{window_period}_upper{NUMBERS_RANGE_UPPER}_sweeps{LOCAL_SOLVER_NUM_SWEEPS}.png")
    print("  Plot generated.")
    print("")


def experiment2(incremental_rate, window_size, batches):
    NUMBERS_RANGE_UPPER = 99
    LOCAL_SOLVER_NUM_SWEEPS = 100

    tts_result_without_states_avg = []
    tts_result_without_states_se = []
    tts_result_with_states_avg = []
    tts_result_with_states_se = []

    for rate in incremental_rate:
        window_period = int(window_size * rate)
        print(f"[{datetime.datetime.today()}] Experimenting window_size={window_size} window_period={window_period} batches={batches} ...")

        tts_without_states = run(window_size=window_size, window_period=window_period, batches=batches, numbers_range_upper=NUMBERS_RANGE_UPPER, local_solver_num_sweeps=LOCAL_SOLVER_NUM_SWEEPS, use_prev_states=False, seed=12345)
        #print(tts_without_states)
        tts_without_states_avg = np.average(tts_without_states)
        tts_without_states_se = np.std(tts_without_states, ddof=1) / np.sqrt(len(tts_without_states))
        tts_result_without_states_avg.append(tts_without_states_avg)
        tts_result_without_states_se.append(tts_without_states_se)
        print(f"  w/o: avg={tts_without_states_avg} se={tts_without_states_se}")

        tts_with_states = run(window_size=window_size, window_period=window_period, batches=batches, numbers_range_upper=NUMBERS_RANGE_UPPER, local_solver_num_sweeps=LOCAL_SOLVER_NUM_SWEEPS, use_prev_states=True, seed=12345)
        #print(tts_with_states)
        tts_with_states_avg = np.average(tts_with_states)
        tts_with_states_se = np.std(tts_with_states, ddof=1) / np.sqrt(len(tts_with_states))
        tts_result_with_states_avg.append(tts_with_states_avg)
        tts_result_with_states_se.append(tts_with_states_se)
        print(f"  w/ : avg={tts_with_states_avg} se={tts_with_states_se}")

    # Plot chart for comparison
    fig = plt.figure(figsize=(10, 6))
    ax = plt.axes()
    plt.errorbar(incremental_rate, tts_result_without_states_avg, yerr=tts_result_without_states_se, marker="o", linewidth=2, capsize=5)
    plt.errorbar(incremental_rate, tts_result_with_states_avg, yerr=tts_result_with_states_se, marker="o", linewidth=2, capsize=5)
    #ax.yaxis.set_major_formatter(ScalarFormatter(useMathText=True))
    #ax.ticklabel_format(style="sci", axis="y", scilimits=(0, 0))
    plt.ylim(ymin=0)
    #plt.yscale("log")
    #plt.ylim(ymin=LOCAL_SOLVER_NUM_SWEEPS)
    plt.xlabel("Incremental step rate")
    plt.ylabel("TTS (sweeps)")
    plt.legend(["w/o previous states", "w/ previous states"])
    plt.title(f"Using previous states (window_size={window_size} batches={batches})")
    plt.savefig(f"use_prev_states_size{window_size}_batches{batches}_upper{NUMBERS_RANGE_UPPER}_sweeps{LOCAL_SOLVER_NUM_SWEEPS}.png")
    print("  Plot generated.")
    print("")


def main():
    # Experiment 1:
    # Plot TTS for each iteration using different problems
    sizes = [10]  # [10, 20, 50, 100, 200]
    incremental_rate = [0.1]  # [0.1, 0.2, 0.5]
    for size in sizes:
        #if int(size * incremental_rate[0]) != 1:
        #    experiment(window_size=size, window_period=1, batches=10)
        for rate in incremental_rate:
            period = int(size * rate)
            experiment1(window_size=size, window_period=period, batches=10)

    # Experiment 2:
    # Compare TTS between incremental rates just using one problem
    sizes = [10, 20, 50, 100]
    incremental_rate = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    for size in sizes:
        experiment2(incremental_rate=incremental_rate, window_size=size, batches=50)


if __name__ == "__main__":
    main()
