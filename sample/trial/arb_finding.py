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

import argparse
import datetime
#import logging
import math

import numpy as np
import optuna
import pandas as pd

import sawatabi

#logging.basicConfig(level=logging.DEBUG)
#logging.basicConfig(level=logging.INFO)

pd.options.display.max_columns = None
pd.options.display.width = None
pd.options.display.max_colwidth = 80


# Generate sample problem
NUM_CURRENCIES = 5
index_to_currency = {0: "CAD", 1: "CNY", 2: "EUR", 3: "JPY", 4: "USD"}
currency_to_index = {"CAD": 0, "CNY": 1, "EUR": 2, "JPY": 3, "USD": 4}
conversion_rates = {
    "CAD": {
        "CNY": 5.10327,
        "EUR": 0.68853,
        "JPY": 78.94,
        "USD": 0.75799,
    },
    "CNY": {
        "CAD": 0.19586,
        "EUR": 0.13488,
        "JPY": 15.47,
        "USD": 0.14864,
    },
    "EUR": {
        "CAD": 1.45193,
        "CNY": 7.41088,
        "JPY": 114.65,
        "USD": 1.10185,
    },
    "JPY": {
        "CAD": 0.01266,
        "CNY": 0.06463,
        "EUR": 0.00872,
        "USD": 0.00961,
    },
    "USD": {
        "CAD": 1.31904,
        "CNY": 6.72585,
        "EUR": 0.90745,
        "JPY": 104.05,
    },
}


def create_model(log_base=100.0, M_0=1.0, M_1=1.0, M_2=1.0):
    # Create model
    model = sawatabi.model.LogicalModel(mtype="qubo")
    x = model.variables(name="x", shape=(NUM_CURRENCIES, NUM_CURRENCIES))

    # - Objective term
    for c_from, rates in conversion_rates.items():
        for c_to, rate in rates.items():
            i = currency_to_index[c_from]
            j = currency_to_index[c_to]
            coeff = M_0 * math.log(rate, log_base)
            model.add_interaction(x[i, j], coefficient=coeff, attributes={"from": c_from, "to": c_to, "rate": rate})

    # - Constraints
    for i in range(NUM_CURRENCIES):
        variables_1, variables_2 = [], []
        for j in range(NUM_CURRENCIES):
            if i != j:
                variables_1.append(x[i, j])
                variables_2.append(x[j, i])

        # - Constraint 1
        constraint = sawatabi.model.constraint.EqualityConstraint(variables_1, variables_2, label=f"Equality Constraint for {index_to_currency[i]}", strength=M_1)
        model.add_constraint(constraint)

        # - Constraint 2
        constraint = sawatabi.model.constraint.ZeroOrOneHotConstraint(variables_1, label=f"Zero-or-One-hot Constraint for {index_to_currency[i]}", strength=M_2)
        model.add_constraint(constraint)

    return model


def check_constraints(sample, record):
    for i in range(NUM_CURRENCIES):
        count_to = count_from = 0
        for j in range(NUM_CURRENCIES):
            if (i != j) and (sample[f"x[{i}][{j}]"] == 1):
                count_to += 1
            if (i != j) and (sample[f"x[{j}][{i}]"] == 1):
                count_from += 1

        # Constraint 1
        if count_to != count_from:
            return False

        # Constraint 2
        if count_to not in [0, 1]:
            return False
        if count_from not in [0, 1]:
            return False

    return True


def interpret(sample, record, satisfied, optimal_profit):
    max_gain = -1.0
    max_cycle = None
    num_feasible = num_profitable = num_optimal = 0
    for start in range(NUM_CURRENCIES):
        #print(f"starting with: {index_to_currency[start]} #{start}")
        i = start
        visited = []
        valid = False

        # Find cycle
        while True:
            succ = []
            for j in range(NUM_CURRENCIES):
                if (i != j) and (sample[f"x[{i}][{j}]"] == 1):
                    succ.append(j)
            if len(succ) != 1:  # invalid cycle
                valid = False
                break
            if succ[0] in visited:  # already visited
                valid = False
                break
            if succ[0] == start:  # cycle finished
                valid = True
                break
            i = succ[0]
            visited.append(succ[0])

        # Rearrange the cycle
        cycle = [start] + visited + [start]

        if valid:
            #print(f"  cycle #{start}: {cycle}")
            num_feasible += record.num_occurrences

            # Calc gain
            gain = 1.0
            for i in range(len(cycle) - 1):
                c_from = index_to_currency[cycle[i]]
                c_to = index_to_currency[cycle[i + 1]]
                gain *= conversion_rates[c_from][c_to]
            #print(f"  gain #{start}: {gain}")
            if max_gain < gain:
                max_cycle = cycle
            max_gain = max(max_gain, gain)

            if gain > 1.0:
                #print(f"  cycle #{start}: {cycle}")
                #print(f"  gain #{start}: {gain}")
                num_profitable += record.num_occurrences

            if not optimal_profit:
                optimal_profit = 1.0007375  # 1.0007375904861755 is the optimal gain
            if gain >= optimal_profit:
                #print(f"  cycle #{start}: {cycle}")
                #print(f"  gain #{start}: {gain}")
                #print("  energy:", record.energy)
                #print("  satisfied:", satisfied)
                num_optimal += record.num_occurrences

    # Returns:
    # - record.num_occurrences if there is at least one feasible cycle, 0 otherwise,
    # - record.num_occurrences if there is at least one cycle that can obtain a positive gain, 0 otherwise,
    # - record.num_occurrences if there is at least one cycle that can obtain the maximum gain, 0 otherwise.
    return min(num_feasible, record.num_occurrences), min(num_profitable, record.num_occurrences), min(num_optimal, record.num_occurrences), max_gain, max_cycle


def cycle_in_currency_name(cycle):
    names = []
    for c in cycle:
        names.append(index_to_currency[c])
    return " -> ".join(names)


def solve_arb_finding(log_base=100.0, M_0=1.0, M_1=1.0, M_2=1.0, num_reads=1000, num_sweeps=100, cooling_rate=0.9, initial_temperature=100.0, reverse_temperature=1.0, seed=12345, exact=False, prev_sampleset=None, optimal_profit=None):
    if exact:
        print(f"== arb finding (log_base={log_base}, M_0={M_0}, M_1={M_1}, M_2={M_2}, exact={exact}) ==")
    else:
        print(f"== arb finding (log_base={log_base}, M_0={M_0}, M_1={M_1}, M_2={M_2}, nr={num_reads}, ns={num_sweeps}, cr={cooling_rate}, it={initial_temperature}, rt={reverse_temperature}, seed={seed}, exact={exact}, prev_sampleset={prev_sampleset is not None}) ==")

    ################################
    # Create Model

    model = create_model(log_base=log_base, M_0=M_0, M_1=M_1, M_2=M_2)
    #print(model)

    ################################
    # With Annealing Solver

    # ExactSolver
    if exact:
        solver = sawatabi.solver.LocalSolver(exact=True)
        sampleset = solver.solve(model.to_physical(), num_reads=num_reads, num_sweeps=num_sweeps, seed=seed)

    else:
        # LocalSolver
        '''
        solver = sawatabi.solver.LocalSolver(exact=False)
        if not prev_sampleset:
            sampleset = solver.solve(model.to_physical(), num_reads=num_reads, num_sweeps=num_sweeps, seed=seed)
        else:
            initial_states = []

            # Set initial_states based on the previous sampleset
            record = sorted(prev_sampleset.record, key=lambda r: r.energy)  # sort by energy
            for r in record:
                sample = dict(zip(prev_sampleset.variables, r.sample))
                initial_states.append(sample)

            #print("Initialized with previous states.")
            assert len(initial_states) == num_reads

            # When using initial_states, use new beta_range to utilize the initial states
            physical = model.to_physical()
            default_beta_range = solver.default_beta_range(physical)
            beta_range = (default_beta_range[0] * 100, default_beta_range[1])  # Tighten the initial temperature than the default temperature
            sampleset = solver.solve(physical, num_reads=num_reads, num_sweeps=num_sweeps, initial_states=initial_states, initial_states_generator="none", seed=seed, beta_range=beta_range)
        '''

        # D-Wave
        '''
        solver = sawatabi.solver.DWaveSolver()
        sampleset = solver.solve(model.to_physical(), chain_strength=2, num_reads=1000)
        '''

        # Optigan
        '''
        solver = sawatabi.solver.OptiganSolver()
        sampleset = solver.solve(model.to_physical(), timeout=1000, duplicate=False)
        '''

        # Sawatabi Solver
        #'''
        solver = sawatabi.solver.SawatabiSolver()
        if not prev_sampleset:
            sampleset = solver.solve(model.to_physical(), num_reads=num_reads, num_sweeps=num_sweeps, cooling_rate=cooling_rate, initial_temperature=initial_temperature, seed=seed)
        else:
            initial_states = []

            # Set initial_states based on the previous sampleset
            record = sorted(prev_sampleset.record, key=lambda r: r.energy)  # sort by energy
            initial_state = dict(zip(prev_sampleset.variables, record[0].sample))
            for _ in range(num_reads):
                initial_states.append(initial_state)

            #print("Initialized with previous states.")
            assert len(initial_states) == num_reads

            sampleset = solver.solve(model.to_physical(), num_reads=num_reads, num_sweeps=num_sweeps, cooling_rate=cooling_rate, initial_temperature=initial_temperature, seed=seed,
                initial_states=initial_states, reverse_options={"reverse_period": int(num_sweeps / 2), "reverse_temperature": reverse_temperature})
        #'''

    #print("\n== Result ==")
    #print(sampleset)

    ################################
    # Solution

    #print("\n== Solution ==")

    energy_min = 999999.99
    max_gain = -1.0
    max_cycle = None
    num_feasible = num_profitable = num_optimal = 0
    for record in sampleset.record:
        sample = dict(zip(sampleset.variables, record.sample))
        energy_min = min(energy_min, record.energy)
        satisfied = check_constraints(sample, record)
        if satisfied:
            nf, np, no, gain, cycle = interpret(sample, record, satisfied, optimal_profit)
            num_feasible += nf
            num_profitable += np
            num_optimal += no
            if max_gain < gain:
                max_cycle = cycle
            max_gain = max(max_gain, gain)

    assert 0 <= num_feasible <= num_reads
    assert 0 <= num_profitable <= num_feasible
    assert 0 <= num_optimal <= num_profitable

    #print("energy_min:", energy_min)

    #if exact:
    #    print("profit:", max_gain)
    if not exact:
        #print("num_feasible:", num_feasible)
        #print("num_profitable:", num_profitable)
        print("num_optimal:", num_optimal)
    #    print("profit:", max_gain)
    #if 0 < max_gain:
    #    print("cycle:", cycle_in_currency_name(max_cycle))

    return sampleset, num_feasible, num_profitable, num_optimal, max_gain, max_cycle


def single_run():
    # Single run
    # Exact Solver
    solve_arb_finding(log_base=100.0, M_0=1.0, M_1=8.0, M_2=8.0, exact=True)
    # Swatabi Solver without previous state (forward annealing)
    sampleset, _, _, _, _, _ = solve_arb_finding(log_base=100.0, M_0=1.0, M_1=8.0, M_2=8.0, num_reads=1000, num_sweeps=100, cooling_rate=0.95, initial_temperature=100.0, seed=12345)
    # Sawatabi Solver with previous state (reverse annealing)
    solve_arb_finding(log_base=100.0, M_0=1.0, M_1=8.0, M_2=8.0, num_reads=1000, num_sweeps=100, cooling_rate=0.95, initial_temperature=100.0, seed=12345, prev_sampleset=sampleset, reverse_temperature=1.0)
    solve_arb_finding(log_base=100.0, M_0=1.0, M_1=8.0, M_2=8.0, num_reads=1000, num_sweeps=100, cooling_rate=0.95, initial_temperature=100.0, seed=12345, prev_sampleset=sampleset, reverse_temperature=2.0)
    solve_arb_finding(log_base=100.0, M_0=1.0, M_1=8.0, M_2=8.0, num_reads=1000, num_sweeps=100, cooling_rate=0.95, initial_temperature=100.0, seed=12345, prev_sampleset=sampleset, reverse_temperature=3.0)
    solve_arb_finding(log_base=100.0, M_0=1.0, M_1=8.0, M_2=8.0, num_reads=1000, num_sweeps=100, cooling_rate=0.95, initial_temperature=100.0, seed=12345, prev_sampleset=sampleset, reverse_temperature=4.0)
    solve_arb_finding(log_base=100.0, M_0=1.0, M_1=8.0, M_2=8.0, num_reads=1000, num_sweeps=100, cooling_rate=0.95, initial_temperature=100.0, seed=12345, prev_sampleset=sampleset, reverse_temperature=5.0)
    solve_arb_finding(log_base=100.0, M_0=1.0, M_1=8.0, M_2=8.0, num_reads=1000, num_sweeps=100, cooling_rate=0.95, initial_temperature=100.0, seed=12345, prev_sampleset=sampleset, reverse_temperature=6.0)
    solve_arb_finding(log_base=100.0, M_0=1.0, M_1=8.0, M_2=8.0, num_reads=1000, num_sweeps=100, cooling_rate=0.95, initial_temperature=100.0, seed=12345, prev_sampleset=sampleset, reverse_temperature=7.0)
    solve_arb_finding(log_base=100.0, M_0=1.0, M_1=8.0, M_2=8.0, num_reads=1000, num_sweeps=100, cooling_rate=0.95, initial_temperature=100.0, seed=12345, prev_sampleset=sampleset, reverse_temperature=8.0)
    solve_arb_finding(log_base=100.0, M_0=1.0, M_1=8.0, M_2=8.0, num_reads=1000, num_sweeps=100, cooling_rate=0.95, initial_temperature=100.0, seed=12345, prev_sampleset=sampleset, reverse_temperature=9.0)
    solve_arb_finding(log_base=100.0, M_0=1.0, M_1=8.0, M_2=8.0, num_reads=1000, num_sweeps=100, cooling_rate=0.95, initial_temperature=100.0, seed=12345, prev_sampleset=sampleset, reverse_temperature=10.0)
    solve_arb_finding(log_base=100.0, M_0=1.0, M_1=8.0, M_2=8.0, num_reads=1000, num_sweeps=100, cooling_rate=0.95, initial_temperature=100.0, seed=12345, prev_sampleset=sampleset, reverse_temperature=20.0)
    solve_arb_finding(log_base=100.0, M_0=1.0, M_1=8.0, M_2=8.0, num_reads=1000, num_sweeps=100, cooling_rate=0.95, initial_temperature=100.0, seed=12345, prev_sampleset=sampleset, reverse_temperature=50.0)


def calc_tts(num_occurrences, num_reads, num_sweeps, P=0.99):
    """
    Calculate TTS (time to solution) in sweeps unit
    """
    p = num_occurrences / num_reads
    if 0.999 < p:
        p = 0.999
    if p < 0.001:
        p = 0.001
    tts = num_sweeps * math.log(1 - P) / math.log(1 - p)
    return tts


def continuous(iterations=10, problem_seed=12345):
    # Continuous:
    # Update conversion rate one by one
    # and find the optimal solution.

    # Experiment options
    exact = False
    use_state = False

    # Initial solution
    print("[0] Initial problem")
    prev_sampleset, _, _, _, _, prev_cycle = solve_arb_finding(log_base=100.0, M_0=1.0, M_1=8.0, M_2=8.0, num_reads=1000, num_sweeps=1000, seed=12345, exact=exact)
    print("")

    # Continuous running
    rng = np.random.default_rng(problem_seed)
    for itr in range(1, iterations + 1):
        while True:
            i = rng.integers(NUM_CURRENCIES)
            j = rng.integers(NUM_CURRENCIES)
            if i != j:
                break

        change_rate = rng.normal(loc=1.0, scale=0.001)
        currency_i = index_to_currency[i]
        currency_j = index_to_currency[j]
        print(f"[{itr}] Change conversion rate {currency_i}-{currency_j} from {conversion_rates[currency_i][currency_j]}", end="")
        conversion_rates[currency_i][currency_j] *= change_rate
        print(f" to {conversion_rates[currency_i][currency_j]}")

        if not use_state:
            sampleset, _, _, _, _, current_cycle = solve_arb_finding(log_base=100.0, M_0=1.0, M_1=8.0, M_2=8.0, num_reads=1000, num_sweeps=1000, seed=12345, exact=exact)
        else:
            sampleset, _, _, _, _, current_cycle = solve_arb_finding(log_base=100.0, M_0=1.0, M_1=8.0, M_2=8.0, num_reads=1000, num_sweeps=1000, seed=12345, exact=exact, prev_sampleset=prev_sampleset)
        if prev_cycle == current_cycle:
            print("\033[32mcycle unchanged.\033[0m\n")
        else:
            print("\033[31mcycle changed!\033[0m\n")

        prev_sampleset = sampleset
        prev_cycle = current_cycle


def continuous_sawatabi(iterations=10, problem_seed=12345):
    # Continuous Sawatabi:
    # Update conversion rate one by one
    # and find the optimal solution for sawatabi.

    # Experiment options
    use_state = False
    nr = 200
    ns = 100
    cr = 0.95
    it = 100.0
    rt = 7.0
    seeds = [101, 102, 103, 104, 105]

    # Output file
    path = f"experiment-output-continuous_sawatabi_{problem_seed}.txt"
    f = open(path, mode="w")

    # Initial solution
    print(f"(0) Initial problem (problem_seed={problem_seed})")

    # Exact Solution
    _, _, _, _, optimal_profit, prev_opt_cycle = solve_arb_finding(log_base=100.0, M_0=1.0, M_1=8.0, M_2=8.0, exact=True)
    print("optimal_profit:", optimal_profit)
    print("optimal_cycle:", cycle_in_currency_name(prev_opt_cycle))
    print("")

    n_optimal = 0
    for seed in range(11, 12):
        _prev_sampleset, _, _, numo, _, _prev_cycle = solve_arb_finding(
            log_base=100.0, M_0=1.0, M_1=8.0, M_2=8.0, num_reads=nr, num_sweeps=ns, cooling_rate=cr, initial_temperature=it, seed=seed, exact=False, optimal_profit=optimal_profit)
        n_optimal += numo
        if numo > 0:
            prev_sampleset = _prev_sampleset
            prev_cycle = _prev_cycle
            #tts = calc_tts(numo, nr, ns)
            #print(f"TTS:", tts)
    #tts = calc_tts(n_optimal, len(seeds) * nr, ns)
    #print(f"n_optimal ({len(seeds)} seeds, {len(seeds) * nr} samples):", n_optimal)
    #print(f"TTS:", tts)
    print("")

    # Continuous running
    rng = np.random.default_rng(problem_seed)
    for itr in range(1, iterations + 1):
        while True:
            i = rng.integers(NUM_CURRENCIES)
            j = rng.integers(NUM_CURRENCIES)
            if i != j:
                break

        change_rate = rng.normal(loc=1.0, scale=0.001)
        currency_i = index_to_currency[i]
        currency_j = index_to_currency[j]
        print(f"[{datetime.datetime.now()}]")
        print(f"({itr}) Change conversion rate {currency_i}-{currency_j} from {conversion_rates[currency_i][currency_j]}", end="")
        conversion_rates[currency_i][currency_j] *= change_rate
        print(f" to {conversion_rates[currency_i][currency_j]} (problem_seed={problem_seed})")

        # Exact Solution
        _, _, _, _, optimal_profit, current_opt_cycle = solve_arb_finding(log_base=100.0, M_0=1.0, M_1=8.0, M_2=8.0, exact=True)
        print("optimal_profit:", optimal_profit)
        print("optimal_cycle:", cycle_in_currency_name(current_opt_cycle))
        if prev_opt_cycle == current_opt_cycle:
            print("\033[32moptimal cycle unchanged.\033[0m\n")
        else:
            print("\033[31moptimal cycle changed!\033[0m\n")
        prev_opt_cycle = current_opt_cycle

        # Sawatabi Solver for 5 different seeds WITHOUT previous state
        n_feasible = n_profitable = n_optimal = 0
        ttss = []
        for seed in range(20 * itr + 1, 20 * itr + 6):
            _, numf, nump, numo, _, _current_cycle = solve_arb_finding(
                log_base=100.0, M_0=1.0, M_1=8.0, M_2=8.0, num_reads=nr, num_sweeps=ns, cooling_rate=cr, initial_temperature=it, seed=seed, exact=False, optimal_profit=optimal_profit)
            n_feasible += numf
            n_profitable += nump
            n_optimal += numo
            if numo > 0:
                current_cycle = _current_cycle
                tts = calc_tts(num_occurrences=numo, num_reads=nr, num_sweeps=ns)
                ttss.append(str(tts))
                print(f"TTS:", tts)
        #tts = calc_tts(n_optimal, len(seeds) * nr, ns)
        #print(f"n_feasible ({len(seeds)} seeds, {len(seeds) * nr} samples):", n_feasible)
        #print(f"n_profitable ({len(seeds)} seeds, {len(seeds) * nr} samples):", n_profitable)
        print(f"n_optimal ({len(seeds)} seeds, {len(seeds) * nr} samples):", n_optimal)
        #print(f"TTS:", tts)
        #f.write(f"{tts}\n")
        f.write("forward_" + str(itr) + " " + (" ".join(ttss)) + "\n")
        print("")

        # Sawatabi Solver for 5 different seeds WITH previous state
        n_feasible = n_profitable = n_optimal = 0
        ttss = []
        for seed in range(20 * itr + 11, 20 * itr + 16):
            _sampleset, numf, nump, numo, _, _current_cycle = solve_arb_finding(
                log_base=100.0, M_0=1.0, M_1=8.0, M_2=8.0, num_reads=nr, num_sweeps=ns, cooling_rate=cr, initial_temperature=it, seed=seed, exact=False, optimal_profit=optimal_profit, prev_sampleset=prev_sampleset, reverse_temperature=rt)
            n_feasible += numf
            n_profitable += nump
            n_optimal += numo
            if numo > 0:
                sampleset = _sampleset
                current_cycle = _current_cycle
                tts = calc_tts(num_occurrences=numo, num_reads=nr, num_sweeps=ns)
                ttss.append(str(tts))
                print(f"TTS:", tts)
        #tts = calc_tts(n_optimal, len(seeds) * nr, ns)
        #print(f"n_feasible ({len(seeds)} seeds, {len(seeds) * nr} samples):", n_feasible)
        #print(f"n_profitable ({len(seeds)} seeds, {len(seeds) * nr} samples):", n_profitable)
        print(f"n_optimal ({len(seeds)} seeds, {len(seeds) * nr} samples):", n_optimal)
        #print(f"TTS:", tts)
        #f.write(f"{tts}\n")
        f.write("reverse_" + str(itr) + " " + (" ".join(ttss)) + "\n")
        print("")

        #if prev_cycle == current_cycle:
        #    print("\033[32mcycle unchanged.\033[0m\n")
        #else:
        #    print("\033[31mcycle changed!\033[0m\n")

        prev_sampleset = sampleset
        prev_cycle = current_cycle

    f.close()


def find_sawatabi_parameters():
    """
    Experiments Note:
    - Annealing parameters for Sawatabi Solver.
    - The following parameter set can get num_optimal >= 2 when num_reads = 100
    - num_sweeps  cooling_rate  initial_temperature  num_feasible/num_profitable/num_optimal
      50          0.85          10                   99 / 15 / 4
      50          0.85          500                  99 / 15 / 4
      50          0.8           10                   98 / 19 / 2
      50          0.8           500                  100 / 18 / 2
      50          0.7           10                   99 / 16 / 3
      100         0.95          100                  100 / 13 / 2
      100         0.85          100                  100 / 18 / 2
      100         0.85          1000                 100 / 17 / 3
      200         0.85          10                   100 / 18 / 4
      200         0.85          100                  99 / 21 / 4
      200         0.8           500                  99 / 15 / 2
      500         0.95          100                  99 / 18 / 3
      500         0.95          1000                 98 / 14 / 3
      500         0.9           10                   100 / 15 / 4
      500         0.8           100                  100 / 14 / 3
    """
    '''
    for sweep in [50, 100, 200, 500]:
        for rate in [0.99, 0.95, 0.9, 0.85, 0.8, 0.7]:
            for temperature in [10.0, 100.0, 500.0, 1000.0]:
                solve_arb_finding(log_base=100.0, M_0=1.0, M_1=8.0, M_2=8.0, num_reads=100, num_sweeps=sweep, cooling_rate=rate, initial_temperature=temperature, seed=12345)
    '''

    """
    Experiments Note:
    - More detail search to set num_read to 100 and for 10 different solver seed
    - The following parameter sets seem to be good:
    - num_sweeps  cooling_rate  initial_temperature  n_feasible/n_profitable/n_optimal
      50          0.85          500                  988 / 153 / 11
      100         0.95          100                  994 / 159 / 14
      200         0.85          100                  993 / 154 / 15
      500         0.9           10                   992 / 145 / 10
      500         0.8           100                  981 / 148 / 11
    """
    #'''
    parameter_set = [
        (50, 0.85, 10),
        (50, 0.85, 100),
        (50, 0.85, 500),
        (50, 0.8, 10),
        (50, 0.8, 100),
        (50, 0.8, 500),
        (50, 0.7, 10),
        (100, 0.95, 100),
        (100, 0.85, 100),
        (100, 0.85, 1000),
        (200, 0.85, 10),
        (200, 0.85, 100),
        (200, 0.8, 500),
        (500, 0.95, 100),
        (500, 0.95, 1000),
        (500, 0.9, 10),
        (500, 0.8, 100),
    ]
    for (sweep, rate, temperature) in parameter_set:
        n_feasible = n_profitable = n_optimal = 0
        for seed in [101, 102, 103, 104, 105, 106, 107, 108, 109, 110]:
            _, numf, nump, numo, _, _ = solve_arb_finding(log_base=100.0, M_0=1.0, M_1=8.0, M_2=8.0, num_reads=100, num_sweeps=sweep, cooling_rate=rate, initial_temperature=temperature, seed=seed)
            n_feasible += numf
            n_profitable += nump
            n_optimal += numo
        print("--------")
        print("n_feasible:", n_feasible)
        print("n_profitable:", n_profitable)
        print("n_optimal:", n_optimal)
        print("")
    #'''


def objective_sawatabi(trial):
    num_sweeps = trial.suggest_int("num_sweeps", 50, 500)
    cooling_rate = trial.suggest_int("cooling_rate", 80, 99)
    initial_temperature = trial.suggest_int("initial_temperature", 100, 1000)

    n_optimal = 0
    for seed in [11, 22, 33, 44, 55]:
        _, _, _, numo, _, _ = solve_arb_finding(log_base=100.0, M_0=1.0, M_1=8.0, M_2=8.0, num_reads=100, num_sweeps=num_sweeps, cooling_rate=cooling_rate/100.0, initial_temperature=initial_temperature, seed=seed)
        n_optimal += numo

    return n_optimal


def find_sawatabi_parameters_using_optuna():
    # Tuning by Optuna
    """
    The number of trials is small now...

    Accuracy: 9.0
    Best hyperparameters: {'num_sweeps': 163, 'cooling_rate': 96, 'initial_temperature': 918}
    """
    study = optuna.create_study(direction="maximize")
    study.optimize(objective_sawatabi, n_trials=20)
    trial = study.best_trial
    print("Accuracy:", trial.value)
    print("Best hyperparameters:", trial.params)


def find_parameters():
    """
    Experiments Note:
    - A bigger log_base is better, since energy decrement isn't be steep even if rate is small (near 0) if log_base is large. Base is 100 for now.
    - It's enough that M_0 = 1.
    - When M_0 = 1, good possible pairs of M_1 and M_2 are: (1.0, 2.0), (8.0, 8.0), (8.0, 16.0), (16.0, 16.0), (32.0, 32.0).
    """
    for m_0 in [1.0]:
        for m_1 in [1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0, 128.0]:
            for m_2 in [1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0, 128.0]:
                if m_0 == 1.0 or m_1 == 1.0 or m_2 == 1.0:
                    solve_arb_finding(log_base=100.0, M_0=m_0, M_1=m_1, M_2=m_2)


def objective(trial):
    log_base = trial.suggest_int("log_base", 2, 100)
    m_0 = trial.suggest_int("m_0", 1, 200)
    m_1 = trial.suggest_int("m_1", 1, 200)
    m_2 = trial.suggest_int("m_2", 1, 200)

    n_optimal = 0
    for seed in [11, 22, 33, 44, 55]:
        _, _, _, numo, _, _ = solve_arb_finding(log_base=log_base, M_0=m_0, M_1=m_1, M_2=m_2, num_reads=1000, num_sweeps=1000, seed=seed)
        n_optimal += numo

    return n_optimal


def find_parameters_using_optuna():
    # Tuning by Optuna
    """
    Environment: num_reads=1000, num_sweeps=1000

    Accuracy: 70.0
    Best hyperparameters: {'log_base': 60, 'm_0': 75, 'm_1': 48, 'm_2': 29}

    Accuracy: 70.0
    Best hyperparameters: {'log_base': 88, 'm_0': 8740, 'm_1': 5869, 'm_2': 5289}

    Accuracy: 67.0
    Best hyperparameters: {'log_base': 52, 'm_0': 93, 'm_1': 62, 'm_2': 38}
    """
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=100)
    trial = study.best_trial
    print("Accuracy:", trial.value)
    print("Best hyperparameters:", trial.params)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--iterations",
        dest="iterations",
        type=int,
        default=50,
        help="Iterations.")
    parser.add_argument(
        "--problem-seed",
        dest="problem_seed",
        type=int,
        default=12345,
        help="Problem seed.")
    args = parser.parse_args()

    #single_run()
    #continuous(iterations=20)
    continuous_sawatabi(iterations=args.iterations, problem_seed=args.problem_seed)

    #find_sawatabi_parameters()
    #find_sawatabi_parameters_using_optuna()
    #find_parameters()
    #find_parameters_using_optuna()


if __name__ == "__main__":
    main()
