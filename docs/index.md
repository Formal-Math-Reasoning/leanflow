<div class="hero" markdown>

# A Python Interface to Lean 4

<div class="typing-wrapper"><span class="typing-effect">Fast, scalable, and easy-to-use</span></div>

[Get Started](quickstart.md#1-installation){ .md-button .md-button--primary }
[View on GitHub](https://github.com/Formal-Math-Reasoning/leanflow){ .md-button }

</div>

---

## What is LeanFlow?

LeanFlow is an easy-to-use, scalable Python interface to [Lean 4](https://lean-lang.org/).  
It lets you run Lean code, interact with proofs, and evaluate formal statements from Python. Ideal for researchers in autoformalization, theorem proving, and experimentation.

<div class="grid" markdown>

<div class="feature-card" markdown>
### <span class="feature-icon">🚀</span> Fast & Efficient

LeanFlow is built for efficiency, allowing you to execute large-scale workloads simultaneously with minimal overhead.

</div>

<div class="feature-card" markdown>
### <span class="feature-icon">😊</span> Easy to Use

A small, Pythonic API that feels intuitive. Start running Lean in just a few lines of code.

</div>

<div class="feature-card" markdown>
### <span class="feature-icon">🔄</span> Flexible Deployment

Use **interactive mode** for local development or **server mode** for scalable experiments. Switch between them with a small config change.

</div>

<div class="feature-card" markdown>
### <span class="feature-icon">🎯</span> Evaluation Tools

Built-in metrics for autoformalization evaluation: TypeCheck, BEq+, LLM Grader, and more.

</div>

</div>

---

## Installation

```bash
pip install leanflow
```

---

## Get Started in 3 Lines

The simplest way to run Lean from Python:

```python
from leanflow import SyncREPL

repl = SyncREPL(lean_version="4.24.0", require_mathlib=True)
print(repl.run("#eval 1 + 1"))
```

That's it! LeanFlow automatically downloads Lean and sets up the environment.

!!! tip "Built for Performance"
    LeanFlow is built on `asyncio` for high-throughput parallel execution. The synchronous API shown above is perfect for getting started, scripts, and notebooks. For production workloads with parallel evaluation, see the [async API](repl.md#1-basic-usage).

---

## Two Ways to Run Lean

<div class="api-toggle">
  <div class="toggle-controls">
    <div class="toggle-slider"></div>
    <button class="toggle-btn active">
      <span class="toggle-icon">💻</span> Local
    </button>
    <button class="toggle-btn">
      <span class="toggle-icon">🌐</span> Server
    </button>
  </div>

  <div class="toggle-content active" markdown>

Run Lean directly on your machine. Perfect for development and testing.

```python
from leanflow import SyncREPL

repl = SyncREPL(lean_version="4.21.0")
result = repl.run("def double (n : Nat) := 2 * n\n#eval double 21")
print(result)  # Environment(messages=..., data='42')
```
  </div>

  <div class="toggle-content" markdown>
Connect to a remote LeanFlow server for shared environments and scalable evaluation.

```python
from leanflow import SyncClient

client = SyncClient(base_url="http://localhost:8000")
result = client.run("def double (n : Nat) := 2 * n\n#eval double 21")
print(result) # Environment(messages=..., data='42')
```

Start a server with: `leanflow-serve --config server.yaml`
  </div>
</div>

---

## Evaluation Metrics

LeanFlow includes built-in metrics for autoformalization evaluation, including type checking and semantic equivalence checks:

```python
from leanflow import TypeCheck, BEqPlus

header = "import Mathlib"
thm1   = "theorem infinite_primes (n : Nat) : ∃ p, n < p ∧ Nat.Prime p := sorry"
thm2   = "theorem infinite_primes_alt (k : Nat) : ∃ q, k < q ∧ Nat.Prime q := sorry"

# Type checking
metric = TypeCheck(repl_config={"lean_version": "4.24.0"})
result = metric.compute(thm1, header=header)  
print(result) # True

# Semantic equivalence (BEq+)
metric = BEqPlus(repl_config={"lean_version": "4.24.0"})
result = metric.compute(thm1, thm2, header=header)
print(result) # True
```

Run evaluations at scale with the CLI:

```bash
leanflow-eval --config config.yaml
```

[:octicons-arrow-right-24: Learn more about metrics](metrics.md)

---

## What's Next?

<div class="grid cards" markdown>

-   :material-clock-fast:{ .lg .middle } **Quickstart**

    ---

    Get up and running in 5 minutes with our step-by-step guide

    [:octicons-arrow-right-24: Quickstart](quickstart.md)

-   :material-book-open-variant:{ .lg .middle } **User Guide**

    ---

    Learn about REPL, Server, and CLI in depth

    [:octicons-arrow-right-24: Documentation](repl.md)

-   :material-code-braces:{ .lg .middle } **API Reference**

    ---

    Complete reference for all classes and methods

    [:octicons-arrow-right-24: API Docs](api/client.md)

</div>

---

## Inspiration & Acknowledgements

This project builds upon the incredible work of the Lean community. We are deeply grateful to the authors of the following projects, which directly inspired LeanFlow:

- **[LeanInteract](https://github.com/augustepoiroux/LeanInteract)** by Auguste Poiroux
- **[Rethinking and Improving Autoformalization](https://github.com/Purewhite2019/rethinking_autoformalization)** by Qi Liu
- **[Kimina](https://github.com/project-numina/kimina-lean-server)** by Project Numina

We highly recommend checking out these projects!

Special thanks to the **[Lean community](https://github.com/leanprover-community/mathlib4)**, the contributors to **Mathlib**, and the authors of the **[Lean REPL](https://github.com/leanprover-community/repl)**, whose tools make this ecosystem possible.

---

## Community & Support

- **GitHub**: [Report issues](https://github.com/Formal-Math-Reasoning/leanflow/issues) and contribute
- **Discussions**: [Ask questions](https://github.com/Formal-Math-Reasoning/leanflow/discussions)
<!-- - **Examples**: [Sample projects](https://github.com/Formal-Math-Reasoning/leanflow-examples) -->

---
