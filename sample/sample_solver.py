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


def sample_neal():
    import dimod
    import neal

    print("\n=== neal (sample) ===\n")

    # BQM
    # bqm = dimod.BinaryQuadraticModel({0: -1, 1: 1}, {(0, 1): 2}, 0.0, dimod.BINARY)
    bqm = dimod.BinaryQuadraticModel({"x[0]": 1, "x[1]": 0}, {("x[0]", "x[1]"): 2}, 0.0, dimod.SPIN)

    # dimod's brute force solver
    sampleset = dimod.ExactSolver().sample(bqm)
    print(sampleset)

    # SA
    sampler = neal.SimulatedAnnealingSampler()
    sampleset = sampler.sample(bqm)
    print(sampleset)


def sample_solver():
    print("\n=== solver ===")
    model = sawatabi.model.LogicalModel(type="ising")

    x = model.variables("x", shape=(1, 2))
    model.add_interaction(x[0, 0], coefficient=10.0)
    model.add_interaction((x[0, 0], x[0, 1]), coefficient=1.0)
    print("\n")
    print(model)

    x = model.append("x", shape=(1, 0))
    model.add_interaction((x[0, 1], x[1, 0]), coefficient=-2.0)
    model.add_interaction((x[1, 0], x[1, 1]), coefficient=3.0)
    print("\n")
    print(model)

    x = model.append("x", shape=(1, 0))
    model.add_interaction((x[1, 1], x[2, 0]), coefficient=-4.0)
    model.add_interaction((x[2, 0], x[2, 1]), coefficient=5.0)
    print("\n")
    print(model)

    physical_model = model.convert_to_physical()
    print("\n")
    print(physical_model)

    solver = sawatabi.solver.LocalSolver()
    resultset = solver.solve(physical_model)
    print("\n")
    print(resultset)


def main():
    sample_neal()
    sample_solver()


if __name__ == "__main__":
    main()
