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

import gzip
import io
import json
import os.path

import dimod
import requests
import yaml

import sawatabi.constants as constants
from sawatabi.model.physical_model import PhysicalModel
from sawatabi.solver.abstract_solver import AbstractSolver


class OptiganSolver(AbstractSolver):
    def __init__(self, config=None, endpoint=None, token=None):
        super().__init__()
        home_dir = os.path.expanduser("~")
        home_config_filename = f"{home_dir}/.optigan.yml"
        self._config_filename = None
        if config:
            self._config_filename = config
        elif os.path.exists(home_config_filename):
            self._config_filename = home_config_filename
        self._endpoint = endpoint
        self._token = token

    def get_config(self):
        with open(self._config_filename, "r") as f:
            config = yaml.load(f, Loader=yaml.SafeLoader)
        return config

    def solve(self, model, num_unit_steps=10, timeout=10000, duplicate=False, gzip_request=True, gzip_response=True):
        self._check_argument_type("model", model, PhysicalModel)

        if len(model._raw_interactions[constants.INTERACTION_LINEAR]) == 0 and len(model._raw_interactions[constants.INTERACTION_QUADRATIC]) == 0:
            raise ValueError("Model cannot be empty.")

        if model.get_mtype() == constants.MODEL_ISING:
            raise ValueError("Ising model is not supported yet. Please try to convert the logical model to QUBO beforehand.")

        # Converts to polynomial (model representation for Optigan)
        polynomial = model.to_polynomial()

        if self._endpoint and self._token:
            endpoint = self._endpoint
            token = self._token
        elif self._config_filename:
            config = self.get_config()
            endpoint = config["api"]["endpoint"]
            token = config["api"]["token"]

        headers = {
            "Authorization": "Bearer {}".format(token),
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
            # Don't compress request body
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
        sampleset = dimod.SampleSet.from_samples(samples, vartype=dimod.BINARY, energy=result["energies"], aggregate_samples=True, sort_labels=True)
        sampleset._info = result

        return sampleset
