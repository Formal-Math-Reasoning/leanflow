# Interactive REPL

The **Interactive REPL** provides a persistent interface for executing Lean 4 code.

It allows you to run Lean commands, inspect the proof state, and maintain a session where definitions and theorems persist between calls. This is ideal for scripting Lean interactions, testing tactics, or building tools that require fine-grained control over the Lean compiler.


## 1. Basic Usage

Choose the API that fits your needs:

=== "Synchronous"
    **Best for:** Debugging, scripts, and data exploration.
    
    The synchronous API provides a simple, convenient way to interact with Lean without `async`/`await` boilerplate.
    
    ```python
    from leanflow import SyncREPL
    
    with SyncREPL(lean_version="4.24.0") as repl:
        result = repl.run("#check 1 + 1")
        print(result)
    ```

=== "Asynchronous"
    **Best for:** High-performance applications and web services.
    
    The asynchronous API uses Python's `asyncio` to handle multiple Lean operations concurrently, essential for performance at scale.
    
    ```python
    import asyncio
    from leanflow import REPL
    
    async def main():
        async with REPL(lean_version="4.24.0") as repl:
            result = await repl.run("#check 1 + 1")
            print(result)
            
    if __name__ == "__main__":
        asyncio.run(main())
    ```


## 2. Understanding the Environment

Every time you run a command, LeanFlow returns an `Environment` object. This object captures the state of the compiler after your code runs, including any errors, warnings, or remaining proof goals.

### 2.1 Inspecting Unfinished Proofs

When a proof uses `sorry`, Lean accepts the syntax but reports the missing logic as a "sorry" placeholder. You can inspect this to see exactly what remains to be proved.

=== "Synchronous"
    ```python
    from leanflow import SyncREPL
    
    code = "theorem add_zero_nat (n : Nat) : n + 0 = n := by sorry"

    with SyncREPL(lean_version="4.24.0") as repl:
        env = repl.run(code)
        print(f"Goals:   {env.goals}")
        print(f"Sorries: {env.sorries}")
    ```

=== "Asynchronous"
    ```python
    import asyncio
    from leanflow import REPL
    
    async def main():
        code = "theorem add_zero_nat (n : Nat) : n + 0 = n := by sorry"

        async with REPL(lean_version="4.24.0") as repl:
            env = await repl.run(code)
            print(f"Goals:   {env.goals}")
            print(f"Sorries: {env.sorries}")
            
    asyncio.run(main())
    ```


### 2.2 Verifying Complete Proofs

When a proof is valid and complete (e.g., using `rfl`), the environment does not contain any error messages or open goal states.

=== "Synchronous"
    ```python
    from leanflow import SyncREPL
    
    code = "theorem add_zero_nat (n : Nat) : n + 0 = n := by rfl"

    with SyncREPL(lean_version="4.24.0") as repl:
        env = repl.run(code)
        
        # A successful proof has empty lists
        assert not env.goals
        assert not env.sorries
        print("Proof complete!")
    ```

=== "Asynchronous"
    ```python
    import asyncio
    from leanflow import REPL
    
    async def main():
        code = "theorem add_zero_nat (n : Nat) : n + 0 = n := by rfl"

        async with REPL(lean_version="4.24.0") as repl:
            env = await repl.run(code)

            # A successful proof has empty lists
            assert not env.goals
            assert not env.sorries
            print("Proof complete!")
            
    asyncio.run(main())
    ```

---

## 3. State Management

The REPL is stateful. Definitions and theorems you execute are remembered for the duration of the session, allowing you to build up complex environments step-by-step.

### 3.1 Explicit Chaining

If you are running commands individually (e.g., inside a loop or conditional logic), you can manually pass the `env` to the next run command.

=== "Synchronous"
    ```python
    from leanflow import SyncREPL
    
    with SyncREPL(lean_version="4.24.0") as repl:
        env1 = repl.run("def double (n : Nat) := 2 * n")
        result = repl.run("#eval double 21", env=env1)
        print(result.messages[-1].data)  # 42
    ```

=== "Asynchronous"
    ```python
    import asyncio
    from leanflow import REPL
    
    async def main():
        async with REPL(lean_version="4.24.0") as repl:
            env1 = await repl.run("def double (n : Nat) := 2 * n")
            result = await repl.run("#eval double 21", env=env1)
            print(result.messages[-1].data) # 42
            
    asyncio.run(main())
    ```


### 3.2 Implicit Chaining

You can pass a list of commands to `run_list`. The REPL automatically propagates the environment state from one command to the next.

=== "Synchronous"
    ```python
    from leanflow import SyncREPL
    
    with SyncREPL(lean_version="4.24.0") as repl:
        results = repl.run_list([
            "def double (n : Nat) := 2 * n",
            "#eval double 21"
        ])
        print(results[-1].messages[-1].data)  # 42
    ```

=== "Asynchronous"
    ```python
    import asyncio
    from leanflow import REPL
    
    async def main():
        async with REPL(lean_version="4.24.0") as repl:
            results = await repl.run_list([
                "def double (n : Nat) := 2 * n",
                "#eval double 21"
            ])
        print(results[-1].messages[-1].data)  # 42
            
    asyncio.run(main())
    ```

---

## 4. Configuration

You can customize the Lean environment by passing arguments to `REPL`.

Typical options include:

- `lean_version`: Lean version to use, e.g. `"v4.24.0"`.
- `require_mathlib`: whether to load Mathlib (default may be `True`).
- `project_path`: path to an existing Lean project.
- `timeout`: optional timeout (seconds) for each command.

Example:

```python
import asyncio
from leanflow import REPL

async def main():
    async with REPL(
        lean_version="4.24.0",
        require_mathlib=True,
        timeout=300,
    ) as repl:
        env = await repl.run("import Mathlib\n#check Nat")
        print(env)

if __name__ == "__main__":
    asyncio.run(main())
```


## 5. Using Custom Lean Projects

By default, LeanFlow handles everything for you. It automatically downloads Lean and Mathlib into the `$HOME/.leanflow` directory. 

**How to Load a Custom Project**

Simply provide the absolute path to your project's root directory using the `project_path` argument.

If you are working on a specific theorem proving repository, you likely have a local folder with its own dependencies and configuration file. You can configure LeanFlow to run inside this existing project context. This allows you to:

- Import local modules (e.g., `import MyProject.Chapter1`).
- Use custom dependencies defined in your lakefile.
- Ensure consistency with your local development environment.


---

## 5. Environment Diagnostics

If you encounter issues with toolchains or paths, you can inspect the internal configuration using the `EnvironmentManager`.

```python
from leanflow import EnvironmentManager

manager = EnvironmentManager({"lean_version": "4.24.0"})
print(manager.diagnose_pretty())
```

This outputs:
```
=== LeanFlow Environment Diagnostics ===

Python: 3.11.0
Platform: linux
CPU count: 64

--- Paths ---
Home: /home/user
Base path: /home/user/.leanflow (exists: True, writable: True)

--- Tools ---
Lake: Lake version 5.0.0-src+797c613 (Lean version 4.24.0)
Git: git version 2.52.0
Elan: elan 4.1.2

--- Existing Environments ---
  - lean-v4.24.0_mathlib
```

---

<!-- ## 6. Troubleshooting

??? question "Lean version not found"
    LeanFlow automatically downloads Lean versions via `elan`. Ensure `elan` is installed:
    
    ```bash
    which elan  # Should show path
    elan show   # List installed toolchains
    ```

??? question "Build takes too long or hangs"
    Mathlib builds can take 30+ minutes on first run. Use cached builds when possible:
    
    ```python
    # Set require_mathlib=False for faster startup (if you don't need it)
    async with REPL(lean_version="4.24.0", require_mathlib=False) as repl:
        ...
    ```

??? question "Memory issues with large proofs"
    `REPL` automatically monitors memory and restarts when needed. You can also manually configure limits or restart:
    
    ```python
    # REPL handles this automatically, but you can force a restart
    await repl._restart_async()
    ```

??? question "Permission errors on shared filesystems"
    On NFS/Lustre, file locks may behave differently. Set a custom base path:
    
    ```python
    async with REPL(
        lean_version="4.24.0",
        base_path="/local/scratch/.leanflow"  # Local disk
    ) as repl:
        ...
    ``` -->