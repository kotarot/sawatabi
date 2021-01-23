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

import platform
import sys

import pandas as pd
from pyqubo import Array, Placeholder

import sawatabi

pd.options.display.max_columns = None
pd.options.display.width = None
pd.options.display.max_colwidth = 80


def _print_utf8(model):
    """
    Util function for Windows (GitHub Actions)
    """
    if platform.system() == "Windows":
        sys.stdout.buffer.write(str(model).encode("utf-8"))
    else:
        print(model)


def model_variables_pyqubo():
    print("\n=== model (variables via pyqubo) ===")
    x = Array.create("x", shape=(2, 3), vartype="SPIN")
    model = sawatabi.model.LogicalModel(mtype="ising")
    model.variables(x)

    print("\nCheck the variables below.")
    _print_utf8(model)


def model_1d():
    print("\n=== model (1d) ===")

    print("\nSet variables x to shape (2,)")
    model = sawatabi.model.LogicalModel(mtype="ising")
    x = model.variables("x", shape=(2,))
    _print_utf8(model)
    print("\n--- Return Value of variables (x) ---")
    print(x)

    print("\nSet variables y to shape (2,)")
    y = model.variables("y", shape=(2,))
    _print_utf8(model)
    print("\n--- Return Value of variables (y) ---")
    print(y)

    model.add_interaction(x[0], coefficient=1.0)
    model.add_interaction(x[1], coefficient=2.0, attributes={"foo": "bar"})
    model.add_interaction((x[0], x[1]), coefficient=-3.0)

    print("\nExpand variables x by (1,)")
    x = model.append("x", shape=(1,))
    _print_utf8(model)
    print("\n--- Return Value of append (x) ---")
    print(x)

    print("\nSet interactions.")
    model.add_interaction(
        x[2],
        coefficient=4.0,
        attributes={"myattr1": "foo", "myattr2": "bar"},
        timestamp=1234567890.123,
    )
    model.update_interaction(x[0], coefficient=1000.0)
    model.update_interaction(name="x[1]", coefficient=2000.0)
    _print_utf8(model)

    print("\nDelete x[0].")
    model.delete_variable(x[0])
    _print_utf8(model)


def model_2d():
    print("\n=== model (2d) ===")

    print("\nSet variables x to shape (2, 2)")
    model = sawatabi.model.LogicalModel(mtype="ising")
    model.variables("y", shape=(2, 2))
    _print_utf8(model)

    print("\nExpand variables x by (1, 1)")
    model.append("y", shape=(1, 1))
    _print_utf8(model)


def model_constraints():
    print("\n=== model (constraints) ===")

    print("\nSet variables x to shape (4,)")
    model = sawatabi.model.LogicalModel(mtype="qubo")
    a = model.variables("a", shape=(4,))
    _print_utf8(model)

    print("\nSet a one-hot constraint to a[0] and a[1]")
    vset = set([a[0], a[1]])
    constraint = sawatabi.model.constraint.NHotConstraint(variables=vset, n=1, label="my constraint 1", strength=1.0)
    model.add_constraint(constraint=constraint)
    _print_utf8(model)

    print("\nSet the one-hot constraint to a[2]")
    vlist = [a[2]]
    constraint.add_variable(variables=vlist)
    _print_utf8(model)

    print("\nSet the one-hot constraint to a[3]")
    constraint.add_variable(variables=a[3])
    _print_utf8(model)

    print("\nDelete a[0].")
    constraint.delete_variable(variables=a[0])
    _print_utf8(model)

    print("\nMerge the constraint model.")
    constraint_model = constraint.to_model()
    model.merge(constraint_model)
    _print_utf8(model)


def model_select():
    print("\n=== select ===")
    model = sawatabi.model.LogicalModel(mtype="ising")

    x = model.variables("x", shape=(2,))
    model.add_interaction(x[0], coefficient=1.0)
    model.add_interaction(x[1], coefficient=2.0, attributes={"foo": "bar", "my attribute": "my my my"})
    model.add_interaction((x[0], x[1]), coefficient=3.0, attributes={"foo": "bar"})
    _print_utf8(model)

    print("\nSelect x[0].")
    res = model.select_interaction("name == 'x[0]'")
    _print_utf8(res)

    print("\nSelect interactions whose attributes.foo is bar.")
    res = model.select_interaction("`attributes.foo` == 'bar'", fmt="dict")
    _print_utf8(res)


def model_from_pyqubo_expression():
    print("\n=== from pyqubo expression ===")
    model = sawatabi.model.LogicalModel(mtype="qubo")
    x = model.variables("x", shape=(4,))
    y = model.variables("y", shape=(4,))

    sum_x = sum(x[i] for i in range(4))
    sum_y = sum(y[i] for i in range(4))
    hamiltonian = (sum_x - sum_y) ** 2

    model.from_pyqubo(hamiltonian)

    print("\nInteractions are set via pyqubo expression.")
    _print_utf8(model)


def model_from_pyqubo_model():
    print("\n=== from pyqubo compiled model ===")
    model = sawatabi.model.LogicalModel(mtype="qubo")
    x = model.variables("x", shape=(4, 2))
    y = model.variables("y", shape=(4, 2))

    sum_x = sum(x[i, 0] for i in range(4))
    sum_y = sum(y[i, 0] for i in range(4))
    hamiltonian = Placeholder("A") * (sum_x - sum_y) ** 2 + 10.0
    pyqubo_model = hamiltonian.compile()

    model.from_pyqubo(pyqubo_model)

    print("\nInteractions are set with pyqubo expression.")
    _print_utf8(model)

    print("\nCheck also the physical model after placeholder resolves.")
    physical = model.to_physical({"A": 2.0})
    _print_utf8(physical)


def model_convert():
    print("\n=== convert ===")
    model = sawatabi.model.LogicalModel(mtype="ising")

    print("\nSet variables x to shape (1, 2) and add interactions.")
    x = model.variables("x", shape=(1, 2))
    model.add_interaction(x[0, 0], coefficient=10.0)
    model.add_interaction(x[0, 1], coefficient=11.0)
    model.add_interaction((x[0, 0], x[0, 1]), coefficient=1.0)
    _print_utf8(model)

    print("\nExpand variables x by shape (1, 0) and add interactions.")
    x = model.append("x", shape=(1, 0))
    model.add_interaction((x[0, 1], x[1, 0]), coefficient=-2.0)
    model.add_interaction((x[1, 0], x[1, 1]), coefficient=3.0)
    _print_utf8(model)

    print("\nx[0, 1] is removed.")
    model.remove_interaction(x[0, 1])
    _print_utf8(model)

    print("\nConvert to Physical model.")
    physical_model = model.to_physical()
    _print_utf8(physical_model)

    print("\nLogical Model.")
    _print_utf8(model)


def model_convert_to_ising():
    print("\n=== convert mtype from qubo to ising ===")
    qubo = sawatabi.model.LogicalModel(mtype="qubo")

    print("\nPrepare a QUBO model and set variable x of shape (2,), and add interactions.")
    x = qubo.variables("x", shape=(2,))
    qubo.add_interaction(x[0], coefficient=10.0)
    qubo.add_interaction(x[1], coefficient=11.0)
    qubo.add_interaction((x[0], x[1]), coefficient=12.0)
    _print_utf8(qubo)

    print("\nConvert the model to Ising.")
    qubo.to_ising()
    _print_utf8(qubo)

    print("\nRe-convert to QUBO.")
    qubo.to_qubo()
    _print_utf8(qubo)

    print("\nAfter to_physical.")
    physical = qubo.to_physical()
    _print_utf8(qubo)
    _print_utf8(physical)


def model_convert_to_qubo():
    print("\n=== convert mtype from ising to qubo ===")
    ising = sawatabi.model.LogicalModel(mtype="ising")

    print("\nPrepare an Ising model and set variables s of shape (3,) and t of shape (2, 2), and add interactions.")
    s = ising.variables("s", shape=(3,))
    t = ising.variables("t", shape=(2, 2))
    ising.add_interaction(s[0], coefficient=10.0)
    ising.add_interaction(s[1], coefficient=11.0)
    ising.add_interaction((s[1], s[2]), coefficient=12.0)
    ising.add_interaction(t[0, 0], coefficient=-20.0)
    ising.add_interaction(t[1, 1], coefficient=-21.0)
    ising.remove_interaction(target=t[1, 1])
    _print_utf8(ising)

    print("\nConvert the model to QUBO.")
    ising.to_qubo()
    _print_utf8(ising)

    print("\nRe-convert to Ising.")
    ising.to_ising()
    _print_utf8(ising)

    print("\nAfter to_physical.")
    physical = ising.to_physical()
    _print_utf8(ising)
    _print_utf8(physical)


def model_convert_with_placeholder():
    print("\n=== convert with placeholder ===")
    model = sawatabi.model.LogicalModel(mtype="ising")

    print("\nSet variables x to shape (7,) and add interactions.")
    x = model.variables("x", shape=(7,))
    model.add_interaction(x[0], coefficient=Placeholder("a"))
    model.add_interaction(x[1], coefficient=Placeholder("b") + 1.0)
    model.add_interaction(x[2], coefficient=2.0)
    model.add_interaction(x[2], name="x[2]-2", coefficient=Placeholder("c"))
    model.add_interaction(x[3], coefficient=2 * Placeholder("d") + 3 * Placeholder("e"))
    model.add_interaction(x[4], coefficient=Placeholder("f"), scale=3.0)
    model.add_interaction(x[5], coefficient=4.0, scale=Placeholder("g"))
    model.add_interaction(x[6], coefficient=Placeholder("h"), scale=Placeholder("i") * 5.0)
    _print_utf8(model)

    print("\nConvert to Physical model with placeholders.")
    physical_model = model.to_physical(placeholder={"a": 1.0, "b": 2.0, "c": 3.0, "d": 4.0, "e": -5.0, "f": 6.0, "g": -7.0, "h": 8.0, "i": 9.0})
    _print_utf8(physical_model)


def main():
    model_variables_pyqubo()
    model_1d()
    model_2d()
    model_constraints()
    model_select()
    model_from_pyqubo_expression()
    model_from_pyqubo_model()
    model_convert()
    model_convert_to_ising()
    model_convert_to_qubo()
    model_convert_with_placeholder()


if __name__ == "__main__":
    main()
