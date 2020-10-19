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


def sample_version():
    print("\n=== version ===\n")
    print("version:", sawatabi.version())
    print("version_info:", sawatabi.version_info())


def sample_current_time():
    print("\n=== current time ===\n")
    print("sec:", sawatabi.utils.current_time())
    print("ms: ", sawatabi.utils.current_time_ms())
    print("us: ", sawatabi.utils.current_time_us())
    print("ns: ", sawatabi.utils.current_time_ns())


def sample_model_pyqubo():
    import pyqubo

    print("\n=== model (pyqubo) ===")
    x = pyqubo.Array.create("x", shape=(2, 3), vartype="SPIN")
    model = sawatabi.model.LogicalModel(type="ising")
    model.variables(x)
    print("\n")
    print(model)


def sample_model_1d():
    print("\n=== model (1d) ===")
    model = sawatabi.model.LogicalModel(type="ising")
    x = model.variables("x", shape=(2,))
    print("\n")
    print(model)
    print("\n--- Return Value of variables (x) ---")
    print(x)

    y = model.variables("y", shape=(2,))
    print("\n")
    print(model)
    print("\n--- Return Value of variables (y) ---")
    print(y)

    model.add_interaction(x[0], coefficient=1.0)
    model.add_interaction(x[1], coefficient=2.0, attributes={"foo": "bar"})
    model.add_interaction((x[0], x[1]), coefficient=-3.0)

    x = model.append("x", shape=(1,))
    print("\n")
    print(model)
    print("\n--- Return Value of append (x) ---")
    print(x)

    model.add_interaction(
        x[2],
        coefficient=4.0,
        attributes={"myattr1": "foo", "myattr2": "bar"},
        timestamp=1234567890123,
    )
    model.update_interaction(x[0], coefficient=1000.0)
    print("\n")
    print(model)


def sample_model_2d():
    print("\n=== model (2d) ===")
    model = sawatabi.model.LogicalModel(type="ising")
    model.variables("y", shape=(2, 2))
    print("\n")
    print(model)

    model.append("y", shape=(1, 1))
    print("\n")
    print(model)


def sample_model_constraints():
    print("\n=== model (constraints) ===")
    model = sawatabi.model.LogicalModel(type="qubo")
    a = model.variables("a", shape=(3,))
    print("\n")
    print(model)

    model.n_hot_constraint(a[(slice(0, 2),)], n=1, label="my constraint 1")
    print("\n")
    print(model)

    model.n_hot_constraint(a[2], n=1, label="my constraint 1")
    print("\n")
    print(model)


def sample_model_convert():
    print("\n=== convert ===")
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

    physical_model = model.convert_to_physical()
    print("\n")
    print(physical_model)


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


def main():
    sample_version()
    sample_current_time()
    sample_model_pyqubo()
    sample_model_1d()
    sample_model_2d()
    sample_model_constraints()

    sample_model_convert()
    # sample_neal()
    # sample_solver()


if __name__ == "__main__":
    main()
