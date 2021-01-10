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
import pytest

from sawatabi.utils.npp import solve_npp_with_dp


@pytest.mark.parametrize(
    "numbers,ans,s1,s2",
    [
        ([1, 2, 4, 5], True, [0, 3], [1, 2]),  # sum: even, exactly partitioned: True
        ([1, 2, 4, 9], False, [0, 1, 2], [3]),  # sum: even, exactly partitioned: False
        ([1, 2, 4, 6], False, [3], [0, 1, 2]),  # sum: odd,  exactly partitioned: False
    ],
)
def test_solve_npp_with_dp(numbers, ans, s1, s2):
    solution = solve_npp_with_dp(numbers)
    assert solution[0] == ans
    assert solution[1] == s1
    assert solution[2] == s2


@pytest.mark.parametrize("seed", [1, 2, 3, 4, 5])
def test_solve_npp_with_dp_random(seed):
    np.random.seed(seed)
    numbers = list(np.random.randint(low=1, high=99, size=100))

    solution = solve_npp_with_dp(numbers)
    s1 = sum(numbers[i] for i in solution[1])
    s2 = sum(numbers[i] for i in solution[2])
    if solution[0]:
        assert s1 == s2
    else:
        assert s1 + 1 == s2


def test_solve_npp_with_dp_with_dp_table(capfd):
    solution = solve_npp_with_dp(numbers=[1, 1, 2, 3, 5, 8, 13, 21], print_dp_table=True)
    assert solution[0]
    assert solution[1] == [1, 4, 7]
    assert solution[2] == [0, 2, 3, 5, 6]

    out, err = capfd.readouterr()

    assert "dp:" in out
    assert "(True, [1, 4, 7])" in out
