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


def main():
    numbers = [1, 1, 2, 3, 5, 8, 13, 21]

    # import numpy as np
    # np.random.seed(12345)
    # numbers = list(np.random.randint(low=1, high=99, size=100))

    s2 = sum(numbers)
    s = int(s2 / 2)

    print("Problem:", numbers)
    print("Sum:", s2)
    print("Half of Sum:", s)
    print("")

    ans = sawatabi.utils.solve_npp_with_dp(numbers)

    print("Answer:", ans)
    solution1 = [numbers[i] for i in ans[1]]
    solution2 = [numbers[i] for i in ans[2]]
    print("Solution 1:", solution1)
    print("Sum 1:", sum(solution1))
    print("Solution 2:", solution2)
    print("Sum 2:", sum(solution2))


if __name__ == "__main__":
    main()
