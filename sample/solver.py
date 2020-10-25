# Copyright 2020 Kotaro Terada, Shingo Furuyama, Junya Usui, and Kazuki Ono
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

import sawatabi


def solver_local():
    print("\n=== solver (local) ===")
    physical = _create_model()

    solver = sawatabi.solver.LocalSolver(exact=False)
    resultset = solver.solve(physical, num_reads=1, num_sweeps=10000, seed=12345)

    _show_resultset(resultset)


def solver_dwave():
    print("\n=== solver (dwave) ===")
    physical = _create_model()

    solver = sawatabi.solver.DWaveSolver()
    resultset = solver.solve(physical, chain_strength=2, num_reads=10)

    _show_resultset(resultset)


def _create_model():
    model = sawatabi.model.LogicalModel(mtype="ising")

    print("\nSet shape to (1, 2)")
    x = model.variables("x", shape=(1, 2))
    model.add_interaction(x[0, 0], coefficient=1.0)
    model.add_interaction((x[0, 0], x[0, 1]), coefficient=1.0)
    print(model)

    print("\nAdd shape by (1, 0)")
    x = model.append("x", shape=(1, 0))
    model.add_interaction((x[0, 1], x[1, 0]), coefficient=-2.0)
    model.add_interaction((x[1, 0], x[1, 1]), coefficient=3.0)
    print(model)

    print("\nAdd shape by (1, 0)")
    x = model.append("x", shape=(1, 0))
    model.add_interaction((x[1, 1], x[2, 0]), coefficient=-4.0)
    model.add_interaction((x[2, 0], x[2, 1]), coefficient=5.0)
    print(model)

    print("\nPhysical model")
    physical = model.to_physical()
    print(physical)

    return physical


def _show_resultset(resultset):
    print("\nresultset")
    print(resultset)
    print("\nresultset.info")
    print(resultset.info)
    print("\nresultset.variables")
    print(resultset.variables)
    print("\nresultset.record")
    print(resultset.record)
    print("\nresultset.vartype:")
    print(resultset.vartype)
    print("\nresultset.first:")
    print(resultset.first)
    print("\nresultset.samples():")
    print([sample for sample in resultset.samples()])


def main():
    solver_local()
    solver_dwave()


if __name__ == "__main__":
    main()
