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

from _solver_helper import _create_qubo_model, _print_sampleset

import sawatabi


def solver_optigan():
    print("\n=== solver (optigan) ===")
    physical = _create_qubo_model()

    solver = sawatabi.solver.OptiganSolver()
    sampleset = solver.solve(physical, timeout=1000, duplicate=True, gzip_request=False, gzip_response=False)

    _print_sampleset(sampleset)


def solver_optigan_gzip():
    print("\n=== solver (optigan gzip) ===")
    physical = _create_qubo_model()

    solver = sawatabi.solver.OptiganSolver()
    sampleset = solver.solve(physical, timeout=1000, duplicate=True)

    _print_sampleset(sampleset)


def main():
    solver_optigan()
    solver_optigan_gzip()


if __name__ == "__main__":
    main()
