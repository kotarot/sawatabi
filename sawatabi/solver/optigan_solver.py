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

import gzip
import io
import json
import os

import dimod
import requests
import yaml

import sawatabi.constants as constants
from sawatabi.model.physical_model import PhysicalModel
from sawatabi.solver.abstract_solver import AbstractSolver


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

        if len(model._raw_interactions[constants.INTERACTION_LINEAR]) == 0 and len(model._raw_interactions[constants.INTERACTION_QUADRATIC]) == 0:
            raise ValueError("Model cannot be empty.")

        if model.get_mtype() == constants.MODEL_ISING:
            raise ValueError("Ising model is not supported yet.")

        # Converts to polynomial (model representation for Optigan)
        polynomial = model.to_polynomial()

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
            raise ValueError(f"Cannot get a valid response (status_code: {response.status_code}).")

        # Interpret the response as JSON
        result = response.json()

        # Create a sampleset object for return
        samples = []
        for spins in result["spins"]:
            sample = dict(zip(list(model._index_to_label.values()), spins))
            samples.append(sample)
        sampleset = dimod.SampleSet.from_samples(samples, dimod.BINARY, energy=result["energies"])
        sampleset._info = result

        return sampleset
