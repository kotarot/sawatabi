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

import glob
import os

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

# Default font size is 12
plt.rcParams['font.size'] = 12


def calc_average_and_standard_error(ttslist):
    result = {
        "average": [],
        "standard_error": [],
    }
    for data in ttslist.values():
        #data = []
        #for h in histories:
        #    data.append(h[i])
        result["average"].append(np.average(data))
        result["standard_error"].append(np.std(data, ddof=1) / np.sqrt(len(data)))
    return result


def main():
    tts_without = {}
    tts_with = {}

    abspath = os.path.dirname(os.path.abspath(__file__))
    outputs = glob.glob(f"{abspath}/experiment-output-continuous_sawatabi_*.txt")
    print(outputs)
    for output in outputs:
        with open(output, mode="r") as f:
            lines = f.read().splitlines()
            for i, line in enumerate(lines):
                index = int(i / 2)
                print(index, line)
                if i % 2 == 0:  # even -> without (forward)
                    if index in tts_without:
                        tts_without[index].extend([float(s) for s in line.split()[1:]])
                    else:
                        tts_without[index] = [float(s) for s in line.split()[1:]]
                else:  # odd -> with (reverse)
                    if index in tts_with:
                        tts_with[index].extend([float(s) for s in line.split()[1:]])
                    else:
                        tts_with[index] = [float(s) for s in line.split()[1:]]

    #print("tts_without:", tts_without)
    #print("tts_with:", tts_with)

    tts_result_without = calc_average_and_standard_error(tts_without)
    tts_result_with = calc_average_and_standard_error(tts_with)
    #print("tts_result_without:", tts_result_without)
    #print("tts_result_with:", tts_result_with)

    # Plot chart for comparison
    fig = plt.figure(figsize=(10, 6))
    ax = plt.axes()

    xdata = list(range(1, len(tts_with) + 1))
    # With errorbars
    #plt.errorbar(xdata, tts_result_without["average"], yerr=tts_result_without["standard_error"], marker="o", linewidth=2, capsize=5)
    #plt.errorbar(xdata, tts_result_with["average"], yerr=tts_result_with["standard_error"], marker="o", linewidth=2, capsize=5)
    # Without
    plt.errorbar(xdata, tts_result_without["average"], yerr=0, marker="o", linewidth=2, capsize=1)
    plt.errorbar(xdata, tts_result_with["average"], yerr=0, marker="o", linewidth=2, capsize=1)

    # Print only integers in x axis
    plt.gca().get_xaxis().set_major_locator(ticker.MaxNLocator(integer=True))

    # Format of y axis
    #ax.yaxis.set_major_formatter(ScalarFormatter(useMathText=True))
    #ax.ticklabel_format(style="sci", axis="y", scilimits=(0, 0))

    plt.ylim(ymin=0)
    plt.xlabel("Iteration")
    plt.ylabel("Time-to-Solution (in sweeps)")
    plt.legend(["forward", "reverse"])
    plt.title(f"Finding arbitrage opportunities by sawatabi solver\nwithout previous state (forward) and with (reverse annealing).")
    plt.savefig(f"experiment-output-continuous-sawatabi.png")

    print("Plot generated.")


if __name__ == "__main__":
    main()
