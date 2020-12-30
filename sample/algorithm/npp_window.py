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

import argparse

import sawatabi


def npp_window(project, topic, subscription, path, output):
    def npp_mapping(prev_model, elements, incoming, outgoing):
        """
        Mapping -- Update the model based on the input data elements
        """

        model = prev_model
        if len(incoming) > 0:
            # Max index of the incoming elements
            max_index = max([i[1][0] for i in incoming])
            # Get current array size
            x_size = model.get_all_size()
            # Update variables
            x = model.append(name="x", shape=(max_index - x_size + 1,))
        else:
            x = model.get_variables_by_name(name="x")

        # print("x:", x)
        # print("elements:", elements)
        # print("incoming:", incoming)
        # print("outgoing:", outgoing)
        for i in incoming:
            for j in elements:
                if i[0] > j[0]:
                    idx_i = i[1][0]
                    idx_j = j[1][0]
                    coeff = -1.0 * i[1][1] * j[1][1]
                    model.add_interaction(target=(x[idx_i], x[idx_j]), coefficient=coeff)

        for o in outgoing:
            idx = o[1][0]
            model.delete_variable(target=x[idx])

        return model

    # end of user-defined function "mapping"

    def npp_unmapping(resultset, elements, incoming, outgoing):
        """
        Unmapping -- Decode spins to a problem solution
        """

        outputs = []
        outputs.append("")
        outputs.append("INPUT -->")
        outputs.append("  " + str([e[1][1] for e in elements]))
        outputs.append("SOLUTION ==>")

        # Decode spins to solution
        spins = resultset.samples()[0]

        set_p, set_n = [], []
        n_set_p = n_set_n = 0
        for e in elements:
            if spins[f"x[{e[1][0]}]"] == 1:
                set_p.append(e[1][1])
                n_set_p += e[1][1]
            elif spins[f"x[{e[1][0]}]"] == -1:
                set_n.append(e[1][1])
                n_set_n += e[1][1]
        outputs.append(f"  Set(+) : sum={n_set_p}, elements={set_p}")
        outputs.append(f"  Set(-) : sum={n_set_n}, elements={set_n}")
        outputs.append(f"  diff   : {abs(n_set_p - n_set_n)}")

        return "\n".join(outputs)

    # end of user-defined function "unmapping"

    def npp_solving(physical_model, elements, incoming, outgoing):
        # Solver instance
        # - LocalSolver
        solver = sawatabi.solver.LocalSolver(exact=False)
        # Solver options as a dict
        SOLVER_OPTIONS = {
            "num_reads": 1,
            "num_sweeps": 10000,
            "seed": 12345,
        }
        # The main solve.
        resultset = solver.solve(physical_model, **SOLVER_OPTIONS)

        # Set a fallback solver if needed here.
        pass

        return resultset

    # end of user-defined function "solving"

    pipeline_args = ["--runner=DirectRunner"]
    if project is not None:
        pipeline_args.append("--streaming")

    algorithm_options = {"window.size": 30, "window.period": 5, "output.with_timestamp": True, "output.prefix": "<<<\n", "output.suffix": "\n>>>\n"}

    if topic is not None:
        input_fn = sawatabi.algorithm.IO.read_from_pubsub_as_number(project=project, topic=topic)
    elif subscription is not None:
        input_fn = sawatabi.algorithm.IO.read_from_pubsub_as_number(project=project, subscription=subscription)
    elif path is not None:
        input_fn = sawatabi.algorithm.IO.read_from_text_as_json(path=path)
        algorithm_options["input.reassign_timestamp"] = True

    if output is not None:
        output_fn = sawatabi.algorithm.IO.write_to_text(path=output)
    else:
        output_fn = sawatabi.algorithm.IO.write_to_stdout()

    # Pipeline creation with Sawatabi
    pipeline = sawatabi.algorithm.Window.create_pipeline(
        algorithm_options=algorithm_options,
        input_fn=input_fn,
        map_fn=npp_mapping,
        solve_fn=npp_solving,
        unmap_fn=npp_unmapping,
        output_fn=output_fn,
        pipeline_args=pipeline_args,
    )

    # Run the pipeline
    result = pipeline.run()
    result.wait_until_finish()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", dest="project", help="Google Cloud Pub/Sub project name.")
    parser.add_argument("--topic", dest="topic", help="Google Cloud Pub/Sub topic name to subscribe messages from.")
    parser.add_argument("--subscription", dest="subscription", help="Google Cloud Pub/Sub subscription name.")
    parser.add_argument("--path", dest="path", help="Path to the local file or the GCS object to read from.")
    parser.add_argument("--output", dest="output", help="Path (prefix) to the output file or the object to write to.")
    args = parser.parse_args()

    npp_window(args.project, args.topic, args.subscription, args.path, args.output)


if __name__ == "__main__":
    main()
