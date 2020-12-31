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

from _solver_helper import _create_ising_model, _create_qubo_model, _print_resultset

import sawatabi


def solver_local_ising():
    print("\n=== solver (local ising) ===")
    physical = _create_ising_model()

    solver = sawatabi.solver.LocalSolver(exact=False)
    resultset = solver.solve(physical, num_reads=1, num_sweeps=10000, seed=12345)

    _print_resultset(resultset)


def solver_local_qubo():
    print("\n=== solver (local qubo) ===")
    physical = _create_qubo_model()

    solver = sawatabi.solver.LocalSolver(exact=False)
    resultset = solver.solve(physical, num_reads=1, num_sweeps=10000, seed=12345)

    _print_resultset(resultset)


def main():
    solver_local_ising()
    solver_local_qubo()


if __name__ == "__main__":
    main()
