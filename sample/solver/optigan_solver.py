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

from solver_helper import _create_qubo_model, _print_resultset

import sawatabi


def solver_optigan():
    print("\n=== solver (optigan) ===")
    physical = _create_qubo_model()

    solver = sawatabi.solver.OptiganSolver()
    resultset = solver.solve(physical, timeout=1000, duplicate=True)

    _print_resultset(resultset)


def main():
    solver_optigan()


if __name__ == "__main__":
    main()
