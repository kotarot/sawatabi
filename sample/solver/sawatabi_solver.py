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

import logging

# fmt: off
from _solver_helper import (_create_ising_model, _create_qubo_model, _create_simple_ising_model_with_only_1_body, _create_simple_ising_model_with_only_2_body,
                            _print_resultset)

import sawatabi

# fmt: on

# logging.basicConfig(level=logging.DEBUG)
logging.basicConfig(level=logging.INFO)


def sawatabi_solver_simple_ising_with_only_1_body():
    print("\n=== solver (sawatabi simple ising with only 1-body) ===")
    physical = _create_simple_ising_model_with_only_1_body()

    solver = sawatabi.solver.SawatabiSolver()
    resultset = solver.solve(physical, num_reads=2, num_sweeps=1000, num_coolings=100, cooling_rate=0.9, initial_temperature=100.0, seed=12345)

    _print_resultset(resultset)


def sawatabi_solver_simple_ising_with_only_2_body():
    print("\n=== solver (sawatabi simple ising with only 2-body) ===")
    physical = _create_simple_ising_model_with_only_2_body()

    solver = sawatabi.solver.SawatabiSolver()
    resultset = solver.solve(physical, num_reads=1, num_sweeps=1000, num_coolings=100, cooling_rate=0.9, initial_temperature=100.0, seed=12345)

    _print_resultset(resultset)


def sawatabi_solver_ising():
    print("\n=== solver (sawatabi ising) ===")
    physical = _create_ising_model()

    solver = sawatabi.solver.SawatabiSolver()
    resultset = solver.solve(physical, num_reads=1, num_sweeps=1000, num_coolings=101, cooling_rate=0.9, initial_temperature=100.0, seed=12345)

    _print_resultset(resultset)


def sawatabi_solver_qubo():
    print("\n=== solver (sawatabi qubo) ===")
    physical = _create_qubo_model()

    solver = sawatabi.solver.SawatabiSolver()
    resultset = solver.solve(physical, num_reads=1, num_sweeps=1000, num_coolings=99, cooling_rate=0.9, initial_temperature=100.0, seed=12345)

    _print_resultset(resultset)


def sawatabi_solver_with_initial_states():
    print("\n=== solver (sawatabi with initial states) ===")
    physical = _create_simple_ising_model_with_only_1_body()

    initial_states = [
        {
            "x[0]": 1,
            "x[1]": -1,
            "x[2]": -1,
            "x[3]": -1,
            "x[4]": -1,
            "x[5]": -1,
            "x[6]": -1,
            "x[7]": -1,
            "x[8]": -1,
            "x[9]": -1,
            "x[10]": -1,
            "x[11]": -1,
        },
    ]

    solver = sawatabi.solver.SawatabiSolver()
    resultset = solver.solve(physical, num_reads=1, num_sweeps=1, num_coolings=1, pickup_mode="sequential", initial_states=initial_states)

    _print_resultset(resultset)


def main():
    sawatabi_solver_simple_ising_with_only_1_body()
    sawatabi_solver_simple_ising_with_only_2_body()
    sawatabi_solver_ising()
    sawatabi_solver_qubo()
    sawatabi_solver_with_initial_states()


if __name__ == "__main__":
    main()
