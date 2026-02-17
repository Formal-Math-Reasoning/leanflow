# Client-Server Architecture

LeanFlow also supports a **client–server** model:

In this architecture, a centralized server manages the heavy lifting (Lean runtime, Mathlib compilation, and state), while lightweight clients send commands and receive results over HTTP.

---

## 1. Server Configuration

The server hosts a persistent Lean environment behind a simple API. You configure it using a YAML file.

### Configuration (`server.yaml`)

```yaml
server:
  host: 0.0.0.0
  port: 8000
  stateless: false    # Set to 'true' to disable persistent environments
  # max_workers: 20   # optional 

repl:
  lean_version: "v4.24.0"
  require_mathlib: true
  timeout: 300
  # project_path: /path/to/lean-project  # optional

  # "Warm Start" Header: These commands run once at startup. All new environments inherit this state.
  header: |
    import Mathlib
```

### Starting the Server

```bash
leanflow-serve --config server.yaml
```

#### Stateless Mode

By setting stateless: true, you configure the server to discard environments immediately after execution. The `env` field in the response will always be `None`.

#### Warm start with `header` field:

The header block allows you to define a "base state" that is shared by everyone. 

- **Efficiency:** Imports like `Mathlib` take time to load. By putting them in the header, they are loaded only once when the server starts.

- **Consistency:** New commands are executed with the exact same imports and namespaces opened, ensuring consistent behavior across experiments.


## 2. Client Usage

The `Client` class provides a Pythonic wrapper for the server's API. It mirrors the REPL interface, making it easy to switch between local and remote execution.

### Basic Connection

=== "Synchronous"
    **Best for:** Scripts, Jupyter notebooks, and simple integrations.

    ```python
    from leanflow import SyncClient

    # Connect to the running server
    with SyncClient(base_url="http://localhost:8000") as client:
        
        if client.status():
            print("Server is connected!")

        result = client.run("#check 1 + 1")
        print(result)
    ```

=== "Asynchronous"
    **Best for:** High-performance applications and web services.

    ```python
    import asyncio
    from leanflow import Client

    async def main():
        async with Client(base_url="http://localhost:8000") as client:
            if await client.status():
                print("Server is connected!")

            result = await client.run("#check 1 + 1")
            print(result)

    if __name__ == "__main__":
        asyncio.run(main())
    ```

## 3. Environment Management

Environments are persistent on the server when it's configured as stateful. You manage the states using Environment IDs.

=== "Synchronous"
    ```python
    from leanflow import SyncClient

    with SyncClient(base_url="http://localhost:8000") as client:
        # 1. Define in a fresh environment
        env1 = client.run("def x : Nat := 42")

        # 2. Extend that environment (pass env ID)
        env2 = client.run("def y : Nat := x + 1", env=env1)

        # 3. Evaluate in the extended environment
        result = client.run("#eval y", env=env2)
        print(result.messages[-1].data)  # "43"
    ```

=== "Asynchronous"
    ```python
    import asyncio
    from leanflow import Client

    async def main():
        async with Client(base_url="http://localhost:8000") as client:
            # 1. Define in a fresh environment
            env1 = await client.run("def x : Nat := 42")

            # 2. Extend that environment (pass env ID)
            env2 = await client.run("def y : Nat := x + 1", env=env1)

            # 3. Evaluate in the extended environment
            result = await client.run("#eval y", env=env2)
            print(result.messages[-1].data)  # "43"

    if __name__ == "__main__":
        asyncio.run(main())
    ```

## 4. API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/status` | Check server status (returns `{"status": "ok"}`) |
| `POST` | `/run` | Execute a Lean command |

### POST /run

Execute a Lean command.

**Request Body:**
```json
{
  "command": "#check 1 + 1",
  "env": null
}
```

**Response:**
```json
{
  "result": {
    "env": 1,
    "goals": [],
    "messages": [],
    "sorries": [],
  }
}
```

---
<!-- 
## 4. Troubleshooting

??? question "Connection refused"
    Ensure the server is running and the port is accessible:
    
    ```bash
    curl http://localhost:8000/status
    ```

??? question "Timeout on first request"
    The first request may trigger Mathlib compilation. Increase client timeout:
    
    ```python
    SyncClient(base_url="...", timeout=600)
    ```

??? question "Environment not found"
    Environment IDs are server-specific. After server restart, all environments are cleared. -->
