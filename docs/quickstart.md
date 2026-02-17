# Quickstart Guide

Welcome to LeanFlow! This guide will walk you through installation and running your first Lean 4 commands from Python.

---

## 1. Installation

### Install LeanFlow

```bash
pip install leanflow
```

### Install Lean 4

LeanFlow requires Lean 4 to be installed on your system. Check out the [official Lean 4 installation guide](https://lean-lang.org/install/) for installation details.

```bash
curl https://elan.lean-lang.org/elan-init.sh -sSf | sh
```

---

## 2. Your First Command

LeanFlow provides both Synchronous and Asynchronous interfaces. Use the toggle below to see the pattern that fits your project.

!!! tip "First Run Note"
    LeanFlow will automatically fetch the requested version (e.g., `v4.24.0`) and configure the environment the first time you run code.


=== "Synchronous"
    **Best for:** Scripts, Jupyter notebooks, and interactive exploration.

    ```python
    from leanflow import SyncREPL

    # A simple theorem with a missing proof ('sorry')
    theorem = "theorem add_zero_nat (n : Nat) : n + 0 = n := by sorry"

    with SyncREPL(lean_version="4.24.0") as repl:
        env = repl.run(theorem)
        print(env)
    ```

=== "Asynchronous"
    **Best for:** High-throughput research and production environments.

    ```python
    import asyncio
    from leanflow import REPL

    async def main():
        
        # A simple theorem with a missing proof ('sorry')
        theorem = "theorem add_zero_nat (n : Nat) : n + 0 = n := by sorry"

        async with REPL(lean_version="4.24.0") as repl:
            result = await repl.run(theorem)
            print(result)

    if __name__ == "__main__":
        asyncio.run(main())
    ```

---

## 3. Understanding Results

When you run a command, LeanFlow returns an `Environment` object. This object tells you everything about the current state of Lean, including what is left to prove and any errors that occurred.

- **`env`**: Environment ID
- **`messages`**: List of Lean messages (errors, warnings, info)

```python
result = repl.run("theorem add_zero_nat (n : Nat) : n + 0 = n := by sorry")

print(result.goals) 
# Output: ['n : Nat\n⊢ n + 0 = n']

for msg in result.messages:
    print(f"[{msg.severity}] {msg.data}")
# Output: [warning] declaration uses 'sorry'

print(result.sorries)
# Output: [Sorry(pos=..., goal='n : Nat\n⊢ n + 0 = n')]
```

| Component | Description |
| :--- | :--- |
| **goals** | A list of current tactical goals (what is left to prove). |
| **messages** | Compiler output, including errors, warnings, and results. |
| **sorries** | Specific data on sorry placeholders, including their exact position and the goal they were meant to solve. |

---

## 4. Using a Server

While running LeanFlow locally is great for testing, you may want to use the Server Mode for larger projects.

### 4.1. Why use a Server?

- **Instant Startup:** Lean takes time to start up and import libraries. A server keeps the environment "warm" in the background, making your scripts respond instantly.
- **Consistency:** Ensure your entire team is running against the exact same Lean version and configuration.
- **Offload Computation:** Run the Lean server on a powerful machine (e.g., a cloud instance).

### 4.2. Create a config file (`server.yaml`):

```yaml
server:
  host: localhost
  port: 8000
repl:
  lean_version: "4.24.0"
```

### 4.3. Start the server:

```bash
leanflow-serve --config server.yaml
```

### 4.4. Connect from Python:

Now, instead of starting a local REPL, connect to your running server:

=== "Synchronous"

    ```python
    from leanflow import SyncClient

    client = SyncClient(base_url="http://localhost:8000")
    result = client.run("theorem add_zero_nat (n : Nat) : n + 0 = n := by sorry")
    print(result)
    ```

=== "Asynchronous"

    ```python
    import asyncio
    from leanflow import Client

    async def main():
        async with Client(base_url="http://localhost:8000") as client:
            result = await client.run("theorem add_zero_nat (n : Nat) : n + 0 = n := by sorry")
            print(result)

    if __name__ == "__main__":
        asyncio.run(main())
    ```

---

## Next Steps

<div class="grid cards" markdown>

-   :material-book-open-variant:{ .lg .middle } **REPL Guide**

    ---

    Configuration, state management, custom projects, and troubleshooting

    [:octicons-arrow-right-24: REPL Documentation](repl.md)

-   :material-server:{ .lg .middle } **Server Mode**

    ---

    Running LeanFlow as a service for shared environments

    [:octicons-arrow-right-24: Server Guide](server.md)

-   :material-chart-line:{ .lg .middle } **Evaluation Metrics**

    ---

    TypeCheck, BEq+, LLM Grader for autoformalization research

    [:octicons-arrow-right-24: Metrics](metrics.md)

-   :material-api:{ .lg .middle } **API Reference**

    ---

    Complete reference for all classes and methods

    [:octicons-arrow-right-24: API Docs](api/repl.md)

</div>

---
<!-- 
## 8. Common Issues

??? question "Lean version not found"
    LeanFlow auto-downloads Lean versions. Make sure `elan` is installed:
    
    ```bash
    which elan  # Should show a path
    ```

??? question "Import errors"
    Make sure LeanFlow is installed:
    
    ```bash
    pip install leanflow
    ```

??? question "Server connection refused"
    Ensure the server is running:
    
    ```bash
    leanflow-serve --config server.yaml
    ```

??? question "Timeout errors"
    Increase the timeout:
    
    ```python
    repl = SyncREPL(lean_version="4.21.0", timeout=600)
    ```
??? question "Could not deserialize ATN"
    Either downgrade antlr4-python3-runtime to version 4.9 or omegaconf to version 2.0. -->