![sawatabi-logo](./figs/sawatabi-logo.gif)

# sawatabi

[![PyPI](https://img.shields.io/pypi/v/sawatabi?style=flat-square)](https://pypi.org/project/sawatabi/)
[![PyPI - Python Version](https://img.shields.io/pypi/pyversions/sawatabi?style=flat-square)](https://pypi.org/project/sawatabi/)
[![GitHub repo size](https://img.shields.io/github/repo-size/kotarot/sawatabi?style=flat-square)](https://github.com/kotarot/sawatabi)
[![GitHub Workflow Status](https://img.shields.io/github/workflow/status/kotarot/sawatabi/ci?style=flat-square)](https://github.com/kotarot/sawatabi/actions?query=workflow%3Aci)
[![Codecov branch](https://img.shields.io/codecov/c/gh/kotarot/sawatabi/main?flag=unittests&style=flat-square&token=SKXOS0VKOA)](https://codecov.io/gh/kotarot/sawatabi)
[![GitHub](https://img.shields.io/github/license/kotarot/sawatabi?style=flat-square)](https://github.com/kotarot/sawatabi/blob/main/LICENSE)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg?style=flat-square)](https://github.com/psf/black)
[![FOSSA Status](https://app.fossa.com/api/projects/git%2Bgithub.com%2Fkotarot%2Fsawatabi.svg?type=shield)](https://app.fossa.com/projects/git%2Bgithub.com%2Fkotarot%2Fsawatabi?ref=badge_shield)

Sawatabi is an application framework to develop and run stream-data-oriented Ising applications with quantum annealing.

## Visualization of a Sample Sawatabi Application

The animation shows visualization of Outlier Detection Problem and its solutions obtained by a Sawatabi application.
The Ising fomulation for the Outlier Detection Problem proposed in [1] is used.
The left figure represents continuous input event data (stream-data) where blue points mean normal values and red points mean abnormal values (outliers).
The right figure represents solutions for Outlier Detection for each window of the input stream-data.
The boxes try to cover only normal points as much as possible.

![Outlier_Detection_JMA_Tokyo_202010_WINDOW](./figs/Outlier_Detection_JMA_Tokyo_202010_WINDOW.gif)

[1] V. N. Smelyanskiy, E. G. Rieffel, S. I. Knysh, C. P. Williams, M. W. Johnson, M. C. Thom, W. G. Macready, and K. L. Pudenz, "A near-term quantum computing approach for hard computational problems in space exploration," arXiv:1204.2821 [quant-ph], 2012. Available: https://arxiv.org/abs/1204.2821

## Usage

### Installation

```
pip install sawatabi
```

### Sample Applications

This section only describes a sample application of NPP (Number Partition Problem), for other sample applications please see:  
https://github.com/kotarot/sawatabi/tree/main/sample/algorithm

#### To run a sample NPP (Number Partition Problem) Sawatabi application on local environment

The following application reads numbers from a local file, run continuous annealing computations to solve NPP on local environment, and writes solutions to the stdout:
```
python sample/algorithm/npp_window.py --input="tests/algorithm/numbers_100.txt"
```

#### To run a sample NPP (Number Partition Problem) Sawatabi application on Google Cloud Dataflow using Google Cloud Pub/Sub

Please prepare your GCP service account credentials as `./gcp-key.json` and open three terminals.

**[1st terminal]** The Pub/Sub publisher continuously publishes numbers to the specified Pub/Sub topic:
```
GOOGLE_APPLICATION_CREDENTIALS="./gcp-key.json" \
    python sample/pubsub/publish_pubsub.py \
        --project=<PROJECT> \
        --topic=<TOPIC> \
        --interval=1.0 \
        --random-number
```
where
- `<PROJECT>` is your GCP project name, and
- `<TOPIC>` is your Google Cloud Pub/Sub topic name to publish messages (numbers) to.

**[2nd terminal]** The Pub/Sub subscriber continuously subscribes solutions from the specified Pub/Sub subscription:
```
GOOGLE_APPLICATION_CREDENTIALS="./gcp-key.json" \
    python sample/pubsub/subscribe_pubsub.py \
        --project=<PROJECT> \
        --subscription=<SUBSCRIPTION>
```
where
- `<PROJECT>` is your GCP project name, and
- `<SUBSCRIPTION>` is your Google Cloud Pub/Sub subscription name to subscribe messages (solutions) from.

**[3rd terminal]** The following application reads numbers from the given Pub/Sub topic, run continuous annealing computations to solve NPP on Google Cloud Dataflow, and writes solutions to the given Pub/Sub topic:
```
GOOGLE_APPLICATION_CREDENTIALS="./gcp-key.json" \
    python sample/algorithm/npp_window.py \
        --project=<PROJECT> \
        --input-topic=<INPUT_TOPIC> \
        --output-topic=<OUTPUT_TOPIC> \
        --dataflow \
        --dataflow-bucket=<DATAFLOW_BUCKET>
```
where
- `<PROJECT>` is your GCP project name,
- `<INPUT_TOPIC>` is your Google Cloud Pub/Sub topic name of input,
- `<OUTPUT_TOPIC>` is your Google Cloud Pub/Sub topic name of output, and
- `<DATAFLOW_BUCKET`> is your GCS bucket name for Dataflow temporary files.

### Solvers

#### If you would like to use the D-Wave solver

Please give credentials directly to the `sawatabi.solver.DWaveSolver()` constructor arguments, or set up a config using dwave-cloud-client:
```
$ dwave config create
Configuration file not found; the default location is: /path/to/your/location/dwave.conf
Configuration file path [/path/to/your/location/dwave.conf]:
Configuration file path does not exist. Create it? [y/N]: y
Profile (create new) [prod]: dev
API endpoint URL [skip]: xxxxxxxxxxxxxxxx
Authentication token [skip]: xxxxxxxxxxxxxxxx
Default client class [skip]:
Default solver [skip]: Advantage_system1.1
Configuration saved.
```

#### If you would like to use the Fixstars GPU solver (Optigan)

Please give credentials directly to the `sawatabi.solver.OptiganSolver()` constructor arguments, or set up a API Token in `~/.optigan.yml`:
```
api:
    endpoint: http://optigan.example.com/method
    token: XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX
```

### For Contributions to the Sawatabi Framework

Please set up a development environment as follows:
```
python -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install --editable ".[dev]"
```

## Acknowledgement

This work is supported by the MITOU Target program from Information-technology Promotion Agency, Japan (IPA).


## License
[![FOSSA Status](https://app.fossa.com/api/projects/git%2Bgithub.com%2Fkotarot%2Fsawatabi.svg?type=large)](https://app.fossa.com/projects/git%2Bgithub.com%2Fkotarot%2Fsawatabi?ref=badge_large)