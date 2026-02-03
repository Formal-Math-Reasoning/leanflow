# REPL API Reference

The REPL module is the core of LeanFlow. It manages the lifecycle of local Lean 4 processes, handles toolchain management (via elan), and provides methods to execute code and query the proof state.

It offers two interfaces:

- `SyncREPL`: A blocking wrapper, ideal for scripts and notebooks.

- `REPL`: The native asynchronous implementation, ideal for high-concurrency workloads.

=== "Synchronous"
    `SyncREPL` is a wrapper around `REPL` abstracting the asyncio event loop.

    ::: leanflow.SyncREPL
        options:
          show_root_heading: true
          show_source: false
          heading_level: 3
          members:
            - __init__
            - start
            - run
            - run_list
            - run_file
            - run_tactic
            - close

=== "Asynchronous"

    ::: leanflow.REPL
        options:
          show_root_heading: true
          show_source: false
          members:
            - __init__
            - load_from_yaml
            - run
            - run_list
            - run_file_async
            - run_tac_async
            - pickle_env_async
            - unpickle_env_async
            - pickle_state_async
            - unpickle_state_async
            - close_async