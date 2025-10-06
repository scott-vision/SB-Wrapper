# SB Wrapper

Utilities and metadata structures for working with SlideBook's remote control
and acquisition APIs from Python.

## Installation

The project now uses a modern `pyproject.toml` configuration with a `src`
layout. Install in editable mode while developing:

```bash
pip install -e .
```

This installs the `sbwrapper` package and its dependencies (`numpy` and
`pyyaml`) as well as a small helper CLI.

## Command Line Entry Point

After installation a ``sbwrapper`` console script becomes available. Use it to
inspect the available APIs directly from a shell:

```bash
sbwrapper --doc            # Render the SBAccess documentation
sbwrapper --list-metadata  # List metadata record classes
```

## Python Usage

```python
from sbwrapper import SBAccess, MicroscopeStates

client = SBAccess()
# client.Connect("127.0.0.1", 60000)  # Example connection
print(MicroscopeStates.CurrentObjective)
```

The package exposes the networking client (`SBAccess`), enumerations for
hardware state queries, helper utilities in `sbwrapper.byte_util`, and the
generated metadata structures under `sbwrapper.c_metadata_lib`.

## Examples and Tests

Demonstration scripts such as `examples/test_sb_access.py` show how to interact
with a running SlideBook instance. Automated tests live in the `tests/`
directory and always import from the `sbwrapper` package.

Always ensure the source files in `src/sbwrapper/` track the latest upstream
versions from the official SlideBook interoperability repository.

