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

import itertools
import random

from dwave.cloud import Client
from dwave.system.composites import EmbeddingComposite
from dwave.system.samplers import DWaveSampler


def dwave_sample():
    print("\n=== dwave_sample ===\n")

    # Connect using the default or environment connection information
    with Client.from_config() as client:

        # Load the default solver
        # solvers = client.get_solvers()
        # print(solvers)
        # Load the specific solver
        solver = client.get_solver(name="Advantage_system1.1")
        print(solver)
        # Solver properties
        print("num_active_qubits:", solver.num_active_qubits)
        # print(solver.nodes)
        # print(solver.undirected_edges)

        # Build a random Ising model to exactly fit the graph the solver supports
        # linear = {index: random.choice([-1, 1]) for index in solver.nodes}
        # quad = {key: random.choice([-1, 1]) for key in solver.undirected_edges}

        # A small problem instead of a random model
        linear, quad = {}, {}
        for index in itertools.islice(solver.nodes, 10):
            linear[index] = random.choice([-1, 1])
        print("linear:", linear)

        # Send the problem for sampling, include solver-specific parameter 'num_reads'
        if solver.check_problem(linear, quad):
            computation = solver.sample_ising(linear, quad, offset=0, num_reads=10)
            print("Solving...")
            computation.wait(timeout=10)

            print("id:", computation.id)
            print("done:", computation.done())
            print("remote_status:", computation.remote_status)

            print("variable -> sample:")
            print({index: computation.samples[0][index] for index in computation.variables})
            print("energy:", computation.energies[0])
            print("num_occurrences:", computation.num_occurrences[0])
            print("sampleset:")
            print(computation.sampleset)
            print("problem_type:", computation.problem_type)
            print("timing:", computation.timing)


def _create_ising_model():
    # Optimal solution of this ising model:
    #   - Spins a, z: +1
    #   - The others: -1
    #   - Energy = -320.0
    ising_linear = {"a": -20}
    ising_quadratic = {
        ("a", "b"): 10,
        ("a", "c"): 11,
        ("a", "d"): 12,
        ("a", "e"): 13,
        ("a", "f"): 14,
        ("a", "g"): 10,
        ("a", "h"): 11,
        ("a", "i"): 12,
        ("a", "j"): 13,
        ("a", "k"): 14,
        ("a", "l"): 10,
        ("a", "m"): 11,
        ("a", "n"): 12,
        ("a", "o"): 13,
        ("a", "p"): 14,
        ("a", "q"): 10,
        ("a", "r"): 11,
        ("a", "s"): 12,
        ("a", "t"): 13,
        ("a", "u"): 14,
        ("a", "v"): 10,
        ("a", "w"): 11,
        ("a", "x"): 12,
        ("a", "y"): 13,
        ("a", "z"): -14,
    }
    return ising_linear, ising_quadratic


def _print_sampleset(sampleset):
    print("sampleset:")
    print(sampleset)
    print("sampleset.record:")
    print(sampleset.record)
    print("sampleset.variables:")
    print(sampleset.variables)
    print("sampleset.info:")
    print(sampleset.info)
    print("sampleset.vartype:")
    print(sampleset.vartype)
    print("sampleset.first:")
    print(sampleset.first)
    print("sampleset.samples():")
    print([sample for sample in sampleset.samples()])
    print("")


def dwave_with_embedding():
    print("\n=== dwave_with_embedding ===\n")

    solver = EmbeddingComposite(DWaveSampler(solver="Advantage_system1.1"))
    ising_linear, ising_quadratic = _create_ising_model()
    sampleset = solver.sample_ising(ising_linear, ising_quadratic)
    _print_sampleset(sampleset)


def dwave_scheduling_options():
    print("\n=== dwave_scheduling_options ===\n")

    solver = EmbeddingComposite(DWaveSampler(solver="Advantage_system1.1"))
    ising_linear, ising_quadratic = _create_ising_model()

    # Normal Annealing
    sampleset = solver.sample_ising(
        ising_linear, ising_quadratic, chain_strength=2.0, answer_mode="histogram", num_reads=10, annealing_time=320
    )
    _print_sampleset(sampleset)

    # Reverse Annealing
    initial_state = dict(zip(sampleset.variables, sampleset.record[0].sample))
    sampleset = solver.sample_ising(
        ising_linear,
        ising_quadratic,
        chain_strength=2.0,
        answer_mode="histogram",
        num_reads=10,
        anneal_schedule=[(0, 1), (160, 0.2), (320, 1)],
        initial_state=initial_state,
        reinitialize_state=False,
    )
    _print_sampleset(sampleset)


def main():
    dwave_sample()
    dwave_with_embedding()
    dwave_scheduling_options()


if __name__ == "__main__":
    main()
