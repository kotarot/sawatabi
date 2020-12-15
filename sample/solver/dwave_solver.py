#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Copyright 2020 Kotaro Terada
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

from _solver_helper import _create_ising_model, _print_resultset

import sawatabi


def solver_dwave():
    print("\n=== solver (dwave) ===")
    physical = _create_ising_model()

    solver = sawatabi.solver.DWaveSolver()
    resultset = solver.solve(physical, chain_strength=2, num_reads=10)

    _print_resultset(resultset)


def solver_dwave_long_schedule():
    print("\n=== solver (dwave long schedule) ===")
    physical = _create_ising_model()

    solver = sawatabi.solver.DWaveSolver(solver="Advantage_system1.1")
    resultset = solver.solve(physical, chain_strength=2, annealing_time=1000, num_reads=1000)

    _print_resultset(resultset)


def main():
    solver_dwave()
    solver_dwave_long_schedule()


if __name__ == "__main__":
    main()
