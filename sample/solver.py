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


def solver():
    print("\n=== solver ===")
    model = sawatabi.model.LogicalModel(mtype="ising")

    print("\nSet shape to (1, 2)")
    x = model.variables("x", shape=(1, 2))
    model.add_interaction(x[0, 0], coefficient=10.0)
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
    physical = model.convert_to_physical()
    print(physical)

    solver = sawatabi.solver.LocalSolver(exact=False)
    resultset = solver.solve(physical)
    print("\nresultset")
    print(resultset)
    print("\nresultset.info")
    print(resultset.info)
    print("\nresultset.variables")
    print(resultset.variables)
    print("\nresultset.record")
    print(resultset.record)


def main():
    solver()


if __name__ == "__main__":
    main()
