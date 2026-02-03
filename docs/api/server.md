# Server API Reference

The Server module provides the backend infrastructure for running LeanFlow as a service. It wraps the Lean REPL in a HTTP server (FastAPI), managing a pool of worker processes to handle concurrent requests.

## Server Class

The `Server` class is the main entry point. It initializes the worker pool, handles the request queue, and manages the lifecycle of persistent Lean environments.

::: leanflow.Server
    options:
      show_root_heading: true
      show_source: false
      members:
        - __init__
        - create_from_yaml
        - run
        - delete_environment
        - shutdown

---

## Starting the Server

### Command Line

```bash
leanflow-serve --config server.yaml
```

### Configuration File (`server.yaml`)

The configuration file is split into two sections:

- `server`: Controls HTTP server and worker settings.

- `repl`: Passed directly to the underlying Lean REPL instances.

```yaml
server:
  host: 0.0.0.0
  port: 8000
  workers: 10
  stateless: false

repl:
  lean_version: "4.24.0"
  require_mathlib: true
  timeout: 300
  # Optional: Shared header to run on startup for every worker
  header: |
    import Mathlib
```

### Configuration Options

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `server.host` | string | `localhost` | Host address to bind to |
| `server.port` | int | `8000` | Port number |
| `server.workers` | int | `10` | Number of REPL workers |
| `server.stateless` | bool | `false` | Stateless mode |
| `repl.lean_version` | string | required | Lean version to use |
| `repl.require_mathlib` | bool | `true` | Whether to include Mathlib |
| `repl.timeout` | int | `300` | Command timeout in seconds |
| `repl.header` | string | `null` | Commands to run at startup |

---

## REST API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/status` | Check server status |
| `POST` | `/run` | Execute a Lean command |

### POST /run

**Request:**
```json
{"command": "#eval 1 + 1", "env": null}
```

**Response:**
```json
{
  "result": {
    "env": 1,   // ID of the new resulting environment
    "messages": [
      {
        "severity": "info",
        "pos": {"line": 1, "column": 0},
        "data": "2"
      }
    ],
    "sorries": [],
    "goals": []
  }
}
```
