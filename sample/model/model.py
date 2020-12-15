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

import sawatabi


def model_pyqubo():
    import pyqubo

    print("\n=== model (pyqubo) ===")
    x = pyqubo.Array.create("x", shape=(2, 3), vartype="SPIN")
    model = sawatabi.model.LogicalModel(mtype="ising")
    model.variables(x)
    print("\nCheck the variables below.")
    print(model)


def model_1d():
    print("\n=== model (1d) ===")
    model = sawatabi.model.LogicalModel(mtype="ising")
    x = model.variables("x", shape=(2,))
    print("\nSet variables x to shape (2,)")
    print(model)
    print("\n--- Return Value of variables (x) ---")
    print(x)

    y = model.variables("y", shape=(2,))
    print("\nSet variables y to shape (2,)")
    print(model)
    print("\n--- Return Value of variables (y) ---")
    print(y)

    model.add_interaction(x[0], coefficient=1.0)
    model.add_interaction(x[1], coefficient=2.0, attributes={"foo": "bar"})
    model.add_interaction((x[0], x[1]), coefficient=-3.0)

    x = model.append("x", shape=(1,))
    print("\nExpand variables x by (1,)")
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
    model.update_interaction(name="x[1]", coefficient=2000.0)
    print("\nSet interactions.")
    print(model)

    model.delete_variable(x[0])
    print("\nDeleted x[0].")
    print(model)


def model_2d():
    print("\n=== model (2d) ===")
    model = sawatabi.model.LogicalModel(mtype="ising")
    model.variables("y", shape=(2, 2))
    print("\nSet variables x to shape (2, 2)")
    print(model)

    model.append("y", shape=(1, 1))
    print("\nExpand variables x by (1, 1)")
    print(model)


def model_constraints():
    print("\n=== model (constraints) ===")
    model = sawatabi.model.LogicalModel(mtype="qubo")
    a = model.variables("a", shape=(3,))
    print("\nSet variables x to shape (3,)")
    print(model)

    model.n_hot_constraint(a[(slice(0, 2),)], n=1, label="my constraint 1")
    print("\nSet a one-hot constraint to a[0] and a[1]")
    print(model)

    model.n_hot_constraint(a[2], n=1, label="my constraint 1")
    print("\nSet the one-hot constraint to a[2]")
    print(model)

    model.delete_variable(a[0])
    print("\nErased a[0].")
    print(model)


def model_convert():
    print("\n=== convert ===")
    model = sawatabi.model.LogicalModel(mtype="ising")

    x = model.variables("x", shape=(1, 2))
    model.add_interaction(x[0, 0], coefficient=10.0)
    model.add_interaction(x[0, 1], coefficient=11.0)
    model.add_interaction((x[0, 0], x[0, 1]), coefficient=1.0)
    print("\nSet variables x to shape (1, 2) and add interactions.")
    print(model)

    x = model.append("x", shape=(1, 0))
    model.add_interaction((x[0, 1], x[1, 0]), coefficient=-2.0)
    model.add_interaction((x[1, 0], x[1, 1]), coefficient=3.0)
    print("\nExpand variables x by shape (1, 0) and add interactions.")
    print(model)

    model.remove_interaction(x[0, 1])
    print("\nx[0, 1] is removed.")
    print(model)

    physical_model = model.to_physical()
    print("\nConvert to Physical model.")
    print(physical_model)

    print("\nLogical Model.")
    print(model)


def model_select():
    print("\n=== select ===")
    model = sawatabi.model.LogicalModel(mtype="ising")

    x = model.variables("x", shape=(2,))
    model.add_interaction(x[0], coefficient=1.0)
    model.add_interaction(x[1], coefficient=2.0, attributes={"foo": "bar", "my attribute": "my my my"})
    model.add_interaction((x[0], x[1]), coefficient=3.0, attributes={"foo": "bar"})
    print(model)

    res = model.select_interaction("name == 'x[0]'")
    print("\nSelected x[0].")
    print(res)

    res = model.select_interaction("`attributes.foo` == 'bar'", fmt="dict")
    print("\nSelected interactions whose attributes.foo is bar.")
    print(res)


def main():
    model_pyqubo()
    model_1d()
    model_2d()
    model_constraints()
    model_convert()
    model_select()


if __name__ == "__main__":
    main()
