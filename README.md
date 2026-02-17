# LeanFlow

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python Version](https://img.shields.io/badge/python-3.9%2B-blue)](https://www.python.org/downloads/)
[![Documentation](https://img.shields.io/badge/docs-live-brightgreen)](https://formal-math-reasoning.github.io/leanflow)

**A fast, scalable, and easy-to-use Python interface to [Lean 4](https://lean-lang.org/).**

LeanFlow lets you run Lean code, interact with proofs, and evaluate formal statements directly from Python. 

<p align="center">
  <img src="docs/images/demo.gif" alt="LeanFlow Server Demo" width="100%">
</p>

---

### Why LeanFlow?
* **🚀 Fast & Efficient:** Built on `asyncio` for high-throughput, parallel execution.
* **🔄 Flexible Deployment:** Seamlessly switch between **Local Mode** for development and **Server Mode** for scalable production workloads.
* **🎯 Evaluation Ready:** Includes built-in metrics for autoformalization like `TypeCheck` and `BEq+` (semantic equivalence).

## Installation

Install LeanFlow via pip:

```bash
pip install leanflow
```

### Install Lean 4

LeanFlow requires Lean 4 to be installed on your system. Check out the [official Lean 4 installation guide](https://lean-lang.org/install/) for installation details.

```bash
curl https://elan.lean-lang.org/elan-init.sh -sSf | sh
```

## Quickstart

### 1. Run Locally (Development)

The simplest way to start. LeanFlow automatically manages the Lean environment for you.

```python
from leanflow import SyncREPL

repl = SyncREPL(lean_version="4.24.0", require_mathlib=True)

result = repl.run("theorem add_zero_nat (n : Nat) : n + 0 = n := by sorry")
print(result)
```

### 2. Run as a Server (Scalable)

For heavier workloads or shared environments, connect to a remote LeanFlow server.

Create a config file (`server.yaml`):

```yaml
server:
  host: localhost
  port: 8000
repl:
  lean_version: "4.24.0"
```

Then, start the server:

```bash
leanflow-serve --config server.yaml
```

And connect from Python:

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

## 🌟 Inspiration & Acknowledgements

This project builds upon the incredible work of the Lean community. We are deeply grateful to the authors of the following projects, which directly inspired LeanFlow:

- **[LeanInteract](https://github.com/augustepoiroux/LeanInteract)** by Auguste Poiroux
- **[Rethinking and Improving Autoformalization](https://github.com/Purewhite2019/rethinking_autoformalization)** by Qi Liu
- **[Kimina](https://github.com/project-numina/kimina-lean-server)** by Project Numina

We highly recommend checking out these projects!

Special thanks to the **[Lean community](https://github.com/leanprover-community/mathlib4)**, the contributors to **Mathlib**, and the authors of the **[Lean REPL](https://github.com/leanprover-community/repl)**, whose tools make this ecosystem possible.


---

## License

LeanFlow is released under the [MIT License](https://github.com/Formal-Math-Reasoning/leanflow/blob/main/LICENSE).

Disclaimer: This open source project is not an official Huawei product, Huawei is not expected to provide support for this project.