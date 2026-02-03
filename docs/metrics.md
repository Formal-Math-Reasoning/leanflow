# Evaluation Metrics

LeanFlow provides a suite of metrics to evaluate the correctness of LLM generated Lean code focusing on autoformalisation. 

## Metrics Overview

| Metric | Description | Key Method |
| :--- | :--- | :--- |
| [TypeCheck](#21-typecheck) | Verifies if code compiles without errors. | checks for `repl` errors |
| [BEqPlus](#22-beqplus) | Strong bidirectional equivalence check using a suite of tactics. | `exact?`, `simp`, `tauto`, `ring` |
| [BEqL](#23-beql) | Lightweight bidirectional equivalence check. | `exact?` only |
| [EquivRfl](#24-equivrfl) | Checks for definitional equality. | `rfl` |
| [LLMGrader](#31-llmgrader) | Semantic equivalence via backtranslation. | LLM (Backtranslation + Judge) |
| [BEq](#32-beq) | Equivalence check augmented with LLM-generated tactics. | `exact?` + LLM Generation |
| [ConJudge](#33-conjudge) | Verifies if a formal statement matches a formal conjecture. | LLM (Judge) |

---

## 1. General Settings

Metrics are available under `leanflow.metrics` and follow a consistent pattern:

- Instantiate the metric with a configuration (e.g., `repl_config` for local execution or `base_url` for server execution).

- Compute the metric on your statements.

There are two categories of metrics:

- Interactive Metrics: Run on individual examples (strings).

- Batch Metrics: Run on lists of examples, often requiring LLM API access.

## 2. Interactive Metrics


### 2.1. TypeCheck

TypeCheck verifies whether a Lean statement is syntactically valid and compiles successfully in the given environment.

=== "Synchronous"
    ```python
    from leanflow import TypeCheck

    statement = "theorem t : 1 + 1 = 2 := by rfl"

    metric = TypeCheck(repl_config={"lean_version": "4.24.0"})
    result = metric.compute(statement)
    print(result)
    ```

=== "Asynchronous"
    ```python
    import asyncio
    from leanflow import TypeCheck

    statement = "theorem t : 1 + 1 = 2 := by rfl"

    async def main():
        metric = TypeCheck(repl_config={"lean_version": "4.24.0"})
        result = await metric.run_check_async(statement)
        print(result)

    if __name__ == "__main__":
        asyncio.run(main())
    ```

---

### 2.2. BEqPlus

BEq+ checks whether two Lean statements are bidirectionally equivalent. It attempts to prove `A ↔ B` using a suite of tactics (`simp`, `tauto`, `ring`, `exact?`). These tactics require Mathlib.

> Source: [Reliable Evaluation and Benchmarks for Statement Autoformalization](https://arxiv.org/abs/2406.07222) (Poiroux et al., EMNLP 2025)

=== "Synchronous"
    ```python
    from leanflow import BEqPlus

    thm1 = "theorem t1 (a b c : Prop) : a ∧ b → c := by sorry"
    thm2 = "theorem t2 (a b c : Prop) : a → b → c := by sorry"

    metric = BEqPlus(repl_config={"lean_version": "4.24.0"})
    result = metric.compute(thm1, thm2, header="import Mathlib")
    print(result)
    ```

=== "Asynchronous"
    ```python
    import asyncio
    from leanflow import BEqPlus

    thm1 = "theorem t1 (a b c : Prop) : a ∧ b → c := by sorry"
    thm2 = "theorem t2 (a b c : Prop) : a → b → c := by sorry"

    async def main():
        metric = BEqPlus(repl_config={"lean_version": "4.24.0"})
        result = await metric.run_check_async(thm1, thm2, header="import Mathlib")
        print(result)

    if __name__ == "__main__":
        asyncio.run(main())
    ```

---

### 2.3. BEqL

BEqL is a lightweight variant of BEq+. It only uses the library search tactic (`exact?`) to check equivalence.

> Source: [Reliable Evaluation and Benchmarks for Statement Autoformalization](https://arxiv.org/abs/2406.07222) (Poiroux et al., EMNLP 2025)

=== "Synchronous"
    ```python
    from leanflow import BEqL

    thm1 = "theorem t1 (p q : Prop) : ¬(p ∨ q) ↔ ¬p ∧ ¬q := by sorry"
    thm2 = "theorem t2 (p q : Prop) : ¬p ∧ ¬q ↔ ¬(q ∨ p) := by sorry"

    metric = BEqL(repl_config={"lean_version": "4.24.0"})
    result = metric.compute(thm1, thm2)
    print(result)
    ```

=== "Asynchronous"
    ```python
    import asyncio
    from leanflow import BEqL

    thm1 = "theorem t1 (p q : Prop) : ¬(p ∨ q) ↔ ¬p ∧ ¬q := by sorry"
    thm2 = "theorem t2 (p q : Prop) : ¬p ∧ ¬q ↔ ¬(q ∨ p) := by sorry"

    async def main():
        metric = BEqL(repl_config={"lean_version": "4.24.0"})
        result = await metric.run_check_async(thm1, thm2)
        print(result)

    if __name__ == "__main__":
        asyncio.run(main())
    ```

---

### 2.4. EquivRfl

EquivRfl checks whether two statements are definitionally equal.

> Source: [Conjecturing: An Overlooked Step in Formal Mathematical Reasoning](https://arxiv.org/abs/2510.11986) (Sivakumar et al., 2025)

=== "Synchronous"
    ```python
    from leanflow import EquivRfl

    conjecture_1 = "abbrev foo : Nat := 2"
    conjecture_2 = "abbrev bar : Nat := 1 + 1"

    metric = EquivRfl(repl_config={"lean_version": "4.24.0"})
    result = metric.compute(conjecture_1, conjecture_2)
    print(result)
    ```

=== "Asynchronous"
    ```python
    import asyncio
    from leanflow import EquivRfl

    conjecture_1 = "abbrev foo : Nat := 2"
    conjecture_2 = "abbrev bar : Nat := 1 + 1"

    async def main():
        metric = EquivRfl(repl_config={"lean_version": "4.24.0"})
        result = await metric.run_check_async(conjecture_1, conjecture_2)
        print(result)

    if __name__ == "__main__":
        asyncio.run(main())
    ```

---

## 3. LLM‑as‑a‑Judge Metrics

These metrics rely on external LLM APIs to judge correctness or semantic equivalence. They are useful when formal proof evaluation fails or is too strict.

---

### 3.1. LLMGrader

LLMGrader performs semantic comparison via Back-Translation:

- Translate the Ground Truth Lean code back to Natural Language.
- Translate the Generated Lean code back to Natural Language.
- Ask an LLM Judge if the two Natural Language statements have the same meaning.

> Source: [FormalMATH: Benchmarking Formal Mathematical Reasoning of Large Language Models
](https://arxiv.org/abs/2505.02735) (Yu et al., 2025)

```python
from leanflow import LLMGrader

data = {
    "formal_statement": "theorem t1 (a b c : Prop) : a ∧ b → c := by sorry",
    "formal_statement_generated": "theorem t2 (a b c : Prop) : a → b → c := by sorry"
}

metric = LLMGrader(
    api_config={"base_url": "<URL>", "api_key": "<KEY>"},
    backtranslation={"model": "deepseek-math"},
    comparison={"model": "gpt-4"},
)

result = metric.compute_batch([data])
print(result)
```

### 3.2. BEq

BEq enhances bidirectional equivalence checking by using `exact?` and an LLM to generate proof tactics.

> Source: [Rethinking and Improving Autoformalization: Towards a Faithful Metric and a Dependency Retrieval-based Approach](https://openreview.net/forum?id=hUb2At2DsQ) (Liu et al., 2024)

```python
from leanflow import BEq

data = {
    "header": "import Mathlib"
    "formal_statement": "theorem t1 (a b c : Prop) : a ∧ b → c := by sorry",
    "formal_statement_generated": "theorem t2 (a b c : Prop) : a → b → c := by sorry",
}

metric = BEq(
    api_config={"base_url": "<URL>", "api_key": "<KEY>"},
    tactic_generator={"model": "deepseek-math"},
    repl_config={"lean_version": "v4.24.0"}
)

result = metric.compute_batch([data])
print(result)
```

---

### 3.3. ConJudge

ConJudge evaluates whether a generated formal statement correctly captures the semantics of a formal conjecture. It uses an LLM as a judge.

> Source: [Conjecturing: An Overlooked Step in Formal Mathematical Reasoning](https://arxiv.org/abs/2510.11986) (Sivakumar et al., 2025)

```python
from leanflow import ConJudge

data = {
    "header": "import Mathlib",
    "formal_conjecture": "abbrev conjecture : ℕ : 13",
    "formal_statement": "theorem hackmath_4 : IsLeast {n | ∀ f : Fin n → Fin 12, ∃ a b, f a = f b} ((conjecture) : ℕ ) := by sorry",
    "formal_statement_generated": "theorem hackmath_4 : IsLeast {n | ∀ f : Fin n → Fin 12, ∃ a b, f a = f b} (26 / 2 : ℕ ) := by sorry",
}

metric = ConJudge(
    api_config={"base_url": "<URL>", "api_key": "<KEY>"},
    comparison={"model": "gpt-4"}
)

result = metric.compute_batch([data])
print(result)
```

---

## 4. Batch Evaluation

For evaluating large datasets using these metrics, use the Evaluation CLI (`leanflow-eval`). It handles parallel execution, error logging, and result aggregation automatically.

See the [Evaluation CLI Guide](cli.md) for configuration details and usage examples.

```bash
leanflow-eval --config config.yaml
```