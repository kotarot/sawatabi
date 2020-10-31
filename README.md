![sawatabi-logo](./figs/sawatabi-logo.gif)

# sawatabi

[![PyPI](https://img.shields.io/pypi/v/sawatabi?style=flat-square)](https://pypi.org/project/sawatabi/)
[![PyPI - Python Version](https://img.shields.io/pypi/pyversions/sawatabi?style=flat-square)](https://pypi.org/project/sawatabi/)
[![GitHub repo size](https://img.shields.io/github/repo-size/kotarot/sawatabi?style=flat-square)](https://github.com/kotarot/sawatabi)
[![GitHub Workflow Status](https://img.shields.io/github/workflow/status/kotarot/sawatabi/ci?style=flat-square)](https://github.com/kotarot/sawatabi/actions?query=workflow%3Aci)
[![Codecov branch](https://img.shields.io/codecov/c/gh/kotarot/sawatabi/main?flag=unittests&style=flat-square&token=SKXOS0VKOA)](https://codecov.io/gh/kotarot/sawatabi)
[![GitHub](https://img.shields.io/github/license/kotarot/sawatabi?style=flat-square)](https://github.com/kotarot/sawatabi/blob/main/LICENSE)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg?style=flat-square)](https://github.com/psf/black)

:warning: **This project is work in progress** :warning:

## Usage

### For Users

```
pip install sawatabi
```

### If you use the D-Wave solver

Set up a config using dwave-cloud-client:
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

### If you use the Fixstars GPU solver (Optigan)

Set up a API Token in `~/.optigan.yml`:
```
api:
    host: optigan.example.com
    token: XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX
```

### For Developers

```
python -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install -e ".[dev]"
```

## Acknowledgement

This work is supported by the MITOU Target program from Information-technology Promotion Agency, Japan (IPA).
