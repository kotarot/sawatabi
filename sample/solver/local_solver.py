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

# fmt: off
from _solver_helper import (_create_ising_model, _create_qubo_model, _create_simple_2x2_ising_model_without_active_var,
                            _create_simple_2x2_qubo_model_without_active_var, _print_sampleset)

import sawatabi

# fmt: on


def solver_local_ising(solver):
    print("\n=== solver (local ising) ===")
    physical = _create_ising_model()

    sampleset = solver.solve(physical, num_reads=1, num_sweeps=10000, seed=12345)

    _print_sampleset(sampleset)


def solver_local_qubo(solver):
    print("\n=== solver (local qubo) ===")
    physical = _create_qubo_model()

    sampleset = solver.solve(physical, num_reads=1, num_sweeps=10000, seed=12345)

    _print_sampleset(sampleset)


def solver_local_simple_2x2_ising_without_active_var(solver):
    print("\n=== solver (simple ising 2x2) ===")
    physical = _create_simple_2x2_ising_model_without_active_var()

    sampleset = solver.solve(physical, num_reads=1, num_sweeps=100, seed=12345)

    _print_sampleset(sampleset)


def solver_local_simple_2x2_qubo_without_active_var(solver):
    print("\n=== solver (simple qubo 2x2) ===")
    physical = _create_simple_2x2_qubo_model_without_active_var()

    sampleset = solver.solve(physical, num_reads=1, num_sweeps=100, seed=12345)

    _print_sampleset(sampleset)


def main():
    solver = sawatabi.solver.LocalSolver(exact=False)

    solver_local_ising(solver)
    solver_local_qubo(solver)
    solver_local_simple_2x2_ising_without_active_var(solver)
    solver_local_simple_2x2_qubo_without_active_var(solver)


if __name__ == "__main__":
    main()
