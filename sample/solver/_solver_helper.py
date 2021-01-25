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

import sawatabi


def _create_ising_model():
    # Optimal solution of this ising model:
    #   - x[1][0] and x[1][1]: -1
    #   - The others: +1
    #   - Energy = -10.0
    model = sawatabi.model.LogicalModel(mtype="ising")

    # print("\nSet shape to (1, 2)")
    x = model.variables("x", shape=(1, 2))
    model.add_interaction(x[0, 0], coefficient=1.0)
    model.add_interaction((x[0, 0], x[0, 1]), coefficient=1.0)
    # print(model)

    # print("\nAdd shape by (1, 0)")
    x = model.append("x", shape=(1, 0))
    model.add_interaction((x[0, 1], x[1, 0]), coefficient=-2.0)
    model.add_interaction((x[1, 0], x[1, 1]), coefficient=3.0)
    # print(model)

    # print("\nAdd shape by (1, 0)")
    x = model.append("x", shape=(1, 0))
    model.add_interaction((x[1, 1], x[2, 0]), coefficient=-4.0)
    model.add_interaction((x[2, 0], x[2, 1]), coefficient=5.0)
    # print(model)

    # Add offset so that the final energy becomes -10.0
    model._offset = 6.0

    # print("\nPhysical model")
    physical = model.to_physical()
    # print(physical)

    return physical


def _create_qubo_model():
    # Optimal solution of this qubo model:
    #   - Only one of the variables: 1
    #   - The others: 0
    #   - Energy = 0.0
    model = sawatabi.model.LogicalModel(mtype="qubo")

    # print("\nOne-hot constraint for an array of (4,)")
    a = model.variables("a", shape=(4,))
    constraint = sawatabi.model.constraint.NHotConstraint(variables=a, n=1)
    model.add_constraint(constraint)
    # print(model)

    # Add offset so that the final energy becomes 0.0
    model._offset = 1.0

    # print("\nPhysical model")
    physical = model.to_physical()
    # print(physical)

    return physical


def _create_simple_ising_model_with_only_1_body():
    # Optimal solution of this ising model:
    #   - All spins: -1
    #   - Energy = -12.0
    model = sawatabi.model.LogicalModel(mtype="ising")

    x = model.variables("x", shape=(12,))
    for i in range(12):
        model.add_interaction(x[i], coefficient=-1.0)

    return model.to_physical()


def _create_simple_ising_model_with_only_2_body():
    # Optimal solution of this ising model:
    #   - All spins: +1 or -1
    #   - Energy = -11.0
    model = sawatabi.model.LogicalModel(mtype="ising")

    x = model.variables("x", shape=(12,))
    for i in range(11):
        model.add_interaction((x[i], x[i + 1]), coefficient=1.0)

    return model.to_physical()


def _create_simple_2x2_ising_model_without_active_var():
    # Optimal solution of this ising model:
    #   - All spins (except a[0][0]): +1
    #   - Energy = 0.0
    model = sawatabi.model.LogicalModel(mtype="ising")

    a = model.variables("a", shape=(2, 2))
    model.add_interaction(a[0, 1], coefficient=2.0)
    model.add_interaction(a[1, 0], coefficient=2.0)
    model.add_interaction(a[1, 1], coefficient=2.0)
    model._offset = 6.0

    return model.to_physical()


def _create_simple_2x2_qubo_model_without_active_var():
    # Optimal solution of this qubo model:
    #   - All spins (except a[0][0]): +1
    #   - Energy = 0.0
    model = sawatabi.model.LogicalModel(mtype="qubo")

    a = model.variables("b", shape=(2, 2))
    model.add_interaction(a[0, 1], coefficient=2.0)
    model.add_interaction(a[1, 0], coefficient=2.0)
    model.add_interaction(a[1, 1], coefficient=2.0)
    model._offset = 6.0

    return model.to_physical()


def _print_resultset(resultset):
    print("\nresultset")
    print(resultset)
    print("\nresultset.info")
    print(resultset.info)
    print("\nresultset.variables")
    print(resultset.variables)
    print("\nresultset.record")
    print(resultset.record)
    print("\nresultset.record[0]")
    print(resultset.record[0])
    print(resultset.record[0].sample)
    print(resultset.record[0].energy)
    print(resultset.record[0].num_occurrences)
    print("\nresultset.vartype:")
    print(resultset.vartype)
    print("\nresultset.first:")
    print(resultset.first)
    print("\nresultset.samples():")
    print([sample for sample in resultset.samples()])
