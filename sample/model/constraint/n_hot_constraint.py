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

import sawatabi


def n_hot_s_of_n_ising(s, n):
    print("\n=== N-hot (s of n) ising ===")
    print("s =", s)
    print("n =", n)

    model = sawatabi.model.LogicalModel(mtype="ising")
    x = model.variables("x", shape=(s,))

    constraint = sawatabi.model.constraint.NHotConstraint(variables=x, n=n, label="n-hot")
    model.add_constraint(constraint)

    print(model)
    print(model.to_physical())


def n_hot_s_of_n_qubo(s, n):
    print("\n=== N-hot (s of n) qubo ===")
    print("s =", s)
    print("n =", n)

    model = sawatabi.model.LogicalModel(mtype="qubo")
    x = model.variables("x", shape=(s,))

    constraint = sawatabi.model.constraint.NHotConstraint(variables=x, n=n, label="n-hot")
    model.add_constraint(constraint)

    print(model)
    print(model.to_physical())


def main():
    n_hot_s_of_n_ising(s=100, n=1)
    n_hot_s_of_n_qubo(s=100, n=1)

    n_hot_s_of_n_ising(s=4, n=1)
    n_hot_s_of_n_qubo(s=4, n=1)

    n_hot_s_of_n_ising(s=4, n=2)
    n_hot_s_of_n_qubo(s=4, n=2)


if __name__ == "__main__":
    main()
