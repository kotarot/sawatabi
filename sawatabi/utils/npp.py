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


"""
Solve Number Partitioning Problem (NPP) with Dynamic Programming (DP).
If a given numbers can be exactly partitioned, return (True, partitions).
If not, return (False, partitions) such that the diff of partitions is minimized.
Note that `partitions` represents a pair of lists of indices of the input numbers list.
"""


def solve_npp_with_dp(numbers, enumerate_all=False, print_dp_table=False):
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
            # Rule 1: Copy its left element
            dp[i][j] = dp[i][j - 1]

            # Rule 2: Add current number to the sub-problem answers
            current_val = numbers[j - 1]
            if (current_val <= i) and dp[i - current_val][j - 1][0]:
                if not enumerate_all:
                    dp[i][j] = (True, dp[i - current_val][j - 1][1] + [j - 1])
                else:
                    if i == current_val:
                        dp[i][j] = (True, dp[i][j][1] + [[j - 1]])
                    else:
                        solution = []
                        for prev in dp[i - current_val][j - 1][1]:
                            solution.append(prev + [j - 1])
                        if len(dp[i][j][1]) != 0:
                            solution.extend(dp[i][j][1])
                        dp[i][j] = (True, solution)

    if print_dp_table:
        print("dp:")
        print("    0        ", end="")
        for j in range(n):
            print(f" {numbers[j]:27} (idx={j})", end="")
        print("")
        for i in range(s + 1):
            print(f"{i:3}", end="")
            for j in range(n + 1):
                print(f" {str(dp[i][j]):35}", end="")
            print("")

    def find_the_other_partition(p, n):
        a = set(list(range(n)))
        return list(a - set(p))

    # The bottom right element (dp[s][n]) is the exact answer
    the_others = []
    if (s2 % 2 == 0) and dp[s][n][0]:
        if not enumerate_all:
            dp[s][n] = (dp[s][n][0], [dp[s][n][1]])
        for p in dp[s][n][1]:
            the_others.append(find_the_other_partition(p, n))
        assert len(dp[s][n][1]) == len(the_others)
        return (True, dp[s][n][1], the_others)
    for i in range(s, -1, -1):
        if dp[i][n][0]:
            if not enumerate_all:
                dp[i][n] = (dp[i][n][0], [dp[i][n][1]])
            for p in dp[i][n][1]:
                the_others.append(find_the_other_partition(p, n))
            assert len(dp[i][n][1]) == len(the_others)
            return (False, dp[i][n][1], the_others)
