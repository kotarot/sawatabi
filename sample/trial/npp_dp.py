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

import numpy as np


"""
Solve Number Partitioning Problem (NPP) with Dynamic Programming (DP).
If a given numbers can be exactly partitioned, return (True, partitions).
If not, return (False, partitions) such that the diff of partitions is minimized.
Note that `partitions` represents a pair of lists of indices of the input numbers list.
"""

def solve_npp(numbers, print_dp=False):
    n = len(numbers)
    s2 = sum(numbers)
    s = int(s2 / 2)

    # DP table of (s + 1) * (n + 1)
    # Each element is a tupple:
    #   - 1st value: True/False
    #   - 2nd value: Indices of values which compose the current sum
    dp = [[(None, []) for i in range(n + 1)] for j in range(s + 1)]

    # Initialize the top row with True
    for j in range(n + 1):
        dp[0][j] = (True, [])

    # Initialize the left column with False (except 0)
    for i in range(1, s + 1):
        dp[i][0] = (False, [])

    # Fill the DP table in bottom up manner
    for i in range(1, s + 1):
        for j in range(1, n + 1):
            # Rule 1
            dp[i][j] = dp[i][j - 1]

            # Rule 2
            current_val = numbers[j - 1]
            if (numbers[j - 1] <= i) and dp[i - current_val][j - 1][0]:
                dp[i][j] = (True, dp[i - current_val][j - 1][1] + [j - 1])

    if print_dp:
        print("dp:")
        print("    0", end="")
        for j in range(n):
            print(f" {numbers[j]:20}", end="")
        print("")
        for i in range(s + 1):
            print(f"{i:3}", end="")
            for j in range(n + 1):
                print(f" {str(dp[i][j]):20}", end="")
            print("")

    def find_another_partition(p, n):
        a = set([i for i in range(n)])
        return list(a - set(p))

    # The bottom right element (dp[s][n]) is the exact answer
    if (s2 % 2 == 0) and dp[s][n][0]:
        return (True, dp[s][n][1], find_another_partition(dp[s][n][1], n))
    else:
        for i in range(s, -1, -1):
            if dp[i][n][0]:
                return (False, dp[i][n][1], find_another_partition(dp[i][n][1], n))


def main():
    numbers = [1, 1, 2, 3, 5, 8, 13, 21]

    # np.random.seed(12345)
    # numbers = list(np.random.randint(low=1, high=99, size=100))

    n = len(numbers)
    s2 = sum(numbers)
    s = int(s2 / 2)

    print("Problem:", numbers)
    print("Sum:", s2)
    print("Half of Sum:", s)
    print("")

    ans = solve_npp(numbers)

    print("Answer:", ans)
    solution1 = [numbers[i] for i in ans[1]]
    solution2 = [numbers[i] for i in ans[2]]
    print("Solution 1:", solution1)
    print("Sum 1:", sum(solution1))
    print("Solution 2:", solution2)
    print("Sum 2:", sum(solution2))


if __name__ == "__main__":
    main()
