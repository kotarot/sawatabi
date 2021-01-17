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

import copy
import datetime

import numpy as np
import Levenshtein

import sawatabi


def experiment_hamming(window_size, window_period, batches, numbers_range_lower=1, numbers_range_upper=99, seed=12345):
    if seed:
        np.random.seed(seed)

    print(f"[{datetime.datetime.today()}] Experimenting window_size={window_size} window_period={window_period} batches={batches} ...")

    all_size = window_size + window_period * (batches - 1)
    all_numbers = list(np.random.randint(low=numbers_range_lower, high=numbers_range_upper, size=all_size))
    # print(all_numbers)

    # Windowing manually
    prev_bitarrays = None
    for r in range(batches):
        start = r * window_period
        stop = r * window_period + window_size
        numbers = all_numbers[start:stop]
        optimal = sawatabi.utils.solve_npp_with_dp(numbers, enumerate_all=True)

        # Convert optimal solutions to bit arrays
        optimal_bitarrays = set()
        for i in range(len(optimal[1])):
            bits = ""
            for n in range(len(numbers)):
                if n in optimal[1][i]:
                    bits += "0"
                elif n in optimal[2][i]:
                    bits += "1"
            optimal_bitarrays.add(bits)

        # Add inverse arrays
        new_optimal_bitarrays = copy.copy(optimal_bitarrays)
        for a in new_optimal_bitarrays:
            inv = a.replace('0', 't').replace('1', '0').replace('t', '1')
            optimal_bitarrays.add(inv)

        print("")
        print("Numbers:", numbers)
        print("Optimal:", optimal)
        print("Optimal (bit-arrays):", optimal_bitarrays)
        print("Prev (bit-arrays):   ", prev_bitarrays)

        print("Hamming distance between previous and current optimal solutions:")
        if prev_bitarrays:
            for p in prev_bitarrays:
                for o in optimal_bitarrays:
                    dist = Levenshtein.hamming(p, o)
                    print(p, o, dist)
                print("----")

        prev_bitarrays = copy.copy(optimal_bitarrays)


def main():
    # Experiment Hamming:
    # Search similarity between continuous solutions by calculating hamming distance
    sizes = [10]  # Note: We cannot list up solutions of an input size more than 20
    incremental_rate = [0.1]
    for size in sizes:
        for rate in incremental_rate:
            period = int(size * rate)
            experiment_hamming(window_size=size, window_period=period, batches=10)


if __name__ == "__main__":
    main()
