# Client API Reference

The `Client` module provides an interface for interacting with a LeanFlow Server.

It abstracts the HTTP communication, allowing you to treat a remote Lean instance almost exactly like a local `REPL` instance. It automatically handles connection pooling, request serialization, and error parsing.

=== "Synchronous"
    The `SyncClient` class is a blocking wrapper around the asynchronous client.

    ::: leanflow.SyncClient
        options:
          show_root_heading: true
          show_source: false
          members:
            - __init__
            - run
            - run_list
            - status
            - close

=== "Asynchronous"
    The `Client` class is the native asynchronous implementation, built on top of `httpx`.

    ::: leanflow.Client
        options:
          show_root_heading: true
          show_source: false
          members:
            - __init__
            - run
            - run_list
            - status
            - close