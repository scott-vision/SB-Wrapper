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
from sbwrapper import MicroscopeConnection, MicroscopeStates

with MicroscopeConnection("127.0.0.1", 60000, keep_alive_interval=30.0) as client:
    # Interact with the remote SlideBook instance via the SBAccess API.
    print(MicroscopeStates.CurrentObjective)
```

The package exposes the low-level networking client (`SBAccess`), the
high-level `MicroscopeConnection` context manager, enumerations for hardware
state queries, helper utilities in `sbwrapper.byte_util`, and the generated
metadata structures under `sbwrapper.c_metadata_lib`.

### Remote Connection Configuration

The connection helper expects the SlideBook remote-control service to be
enabled. Typical deployments expose the service on TCP port ``60000`` and use
the hostname or IP address of the workstation running SlideBook. When
credentials are required, authenticate through the returned `SBAccess` instance
inside the context manager (for example via `client.Login(user, password)`).

For repeatable automation workflows you may prefer setting environment
variables and passing them into ``MicroscopeConnection``:

```bash
export SBWRAPPER_HOST=192.168.1.20
export SBWRAPPER_PORT=60000
```

```python
from sbwrapper import MicroscopeConnection
import os

conn = MicroscopeConnection(
    os.environ["SBWRAPPER_HOST"],
    int(os.environ.get("SBWRAPPER_PORT", "60000")),
    keep_alive_interval=10.0,
    keep_alive_message="PING\n",
)

with conn as client:
    # Perform initialization commands here. The connection automatically
    # retries transient network failures and keeps the socket alive.
    ...
```

The connection object can be reused across threads, and calling ``close()`` or
exiting the context manager ensures the socket is shut down and the keep-alive
worker terminates.

## Examples and Tests

Demonstration scripts such as `examples/test_sb_access.py` show how to interact
with a running SlideBook instance. Automated tests live in the `tests/`
directory and always import from the `sbwrapper` package.

Always ensure the source files in `src/sbwrapper/` track the latest upstream
versions from the official SlideBook interoperability repository.

