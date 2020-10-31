# Copyright 2020 Kotaro Terada, Shingo Furuyama, Junya Usui, and Kazuki Ono
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

import gzip
import io
import json
import os

import requests
import yaml

import sawatabi.constants as constants
from sawatabi.model.physical_model import PhysicalModel
from sawatabi.solver.abstract_solver import AbstractSolver
from sawatabi.solver.sawatabi_sample_set import SawatabiSampleSet


class OptiganSolver(AbstractSolver):
    def __init__(self, config=None):
        super().__init__()
        if config:
            self.config_filename = config
        else:
            self.config_filename = "{}/.optigan.yml".format(os.environ["HOME"])

    def get_config(self):
        with open(self.config_filename, "r") as f:
            config = yaml.load(f, Loader=yaml.SafeLoader)
        return config

    def solve(self, model, num_unit_steps=10, timeout=10000, duplicate=False, gzip_request=True, gzip_response=True):
        self._check_argument_type("model", model, PhysicalModel)

        if (
            len(model._interactions[constants.INTERACTION_LINEAR]) == 0
            and len(model._interactions[constants.INTERACTION_QUADRATIC]) == 0
        ):
            raise ValueError("Model cannot be empty.")

        if model.get_mtype() == constants.MODEL_ISING:
            raise ValueError("Ising model is not supported yet.")

        # For optigan, a variable identifier must be an integer.
        # Names for variables in the physical model is string, we need to convert them.
        #
        # Signs for Optigan are opposite from our definition.
        # - Optigan:  H =   sum( Q_{ij} * x_i * x_j ) + sum( Q_{i, i} * x_i )
        # - Ours:     H = - sum( J_{ij} * x_i * x_j ) - sum( h_{i} * x_i )
        polynomial = []
        map_label_to_index = {}
        map_index_to_label = []
        current_max_index = 0
        for k, v in model._interactions[constants.INTERACTION_LINEAR].items():
            if k in map_label_to_index:
                index = map_label_to_index[k]
            else:
                map_label_to_index[k] = current_max_index
                map_index_to_label.append(k)
                index = current_max_index
                current_max_index += 1
            polynomial.append([index, index, -1.0 * v])
        for k, v in model._interactions[constants.INTERACTION_QUADRATIC].items():
            index = [None, None]
            for i in range(2):
                if k[i] in map_label_to_index:
                    index[i] = map_label_to_index[k[i]]
                else:
                    map_label_to_index[k[i]] = current_max_index
                    map_index_to_label.append(k[i])
                    index[i] = current_max_index
                    current_max_index += 1
            polynomial.append([index[0], index[1], -1.0 * v])

        config = self.get_config()
        endpoint = "http://{}/solve".format(config["api"]["host"])
        headers = {
            "Authorization": "Bearer {}".format(config["api"]["token"]),
            "X-Accept": "application/json",
        }
        payload = {
            "num_unit_steps": num_unit_steps,
            "timeout": timeout,  # in milli seconds
            "polynomial": polynomial,
        }
        if duplicate:
            payload["outputs"] = {
                "duplicate": True,
                "num_outputs": 0,
            }

        if gzip_response:
            headers["Accept-Encoding"] = "gzip"
            # Note: Decompress will be performed by the library.

        if gzip_request:
            # Compress request body
            buf = io.BytesIO()
            with gzip.GzipFile(fileobj=buf, mode="wb") as f:
                f.write(json.dumps(payload).encode("utf-8"))
            headers["Content-Encoding"] = "gzip"
            response = requests.post(endpoint, headers=headers, data=buf.getvalue())
        else:
            # Don't comress request body
            headers["Content-Type"] = "application/json; charset=UTF-8"
            response = requests.post(endpoint, headers=headers, json=payload)

        if response.status_code != 200:
            raise ValueError("Cannot get a valid response.")

        # Interpret the response as JSON
        result = response.json()

        # Create a sampleset object for return
        sampleset = SawatabiSampleSet()
        sampleset.info = result
        sampleset.variables = map_index_to_label
        for i, spins in enumerate(result["spins"]):
            sampleset.add_record(spins, result["energies"][i])

        return sampleset
