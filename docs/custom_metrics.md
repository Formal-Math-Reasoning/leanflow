# Custom Metrics

LeanFlow is designed to be extensible. You can create your own evaluation metrics using `leanflow.metrics.Metric`.

---

## 1. Creating a Logic-Based Metric

Let's build a `NoSorryMetric`. This metric will:

- Check if the generated code contains the string sorry.
- If it doesn't, verify that the code actually compiles (typechecks).

### 1.1. Implementation

Create a file named `my_metrics/no_sorry.py`

```python
from typing import Any, Dict
from leanflow.metrics import Metric

class NoSorryMetric(Metric):
    """Checks that generated code doesn't contain 'sorry' and optionally typechecks."""

    def __init__(self, metric_config: dict[str, Any] = {}, **shared_dependencies):
        super().__init__(metric_config, **shared_dependencies)
        self.also_check_typecheck = metric_config.get("check_typecheck", True)

    async def run_check_async(self, example: dict[str, Any]) -> dict[str, Any]:
        """Check if the generated statement has no 'sorry' and optionally typechecks.

        Returns:
            Dict with 'has_sorry' and optionally 'typechecks' fields.
        """
        runner, context = self.get_runner()

        async with context:
            statement = example.get("formal_statement_generated", "")

            # Text-based check
            has_sorry = "sorry" in statement.lower()
            result: Dict[str, Any] = {"has_sorry": has_sorry}

            # Optionally also typecheck when there is no 'sorry'
            if self.also_check_typecheck and not has_sorry:
                env = await runner.run(statement)
                has_error = any(m.severity == "error" for m in env.messages)
                result["typecheck"] = not has_error

            return result
```

### 1.2. Testing

```python
import asyncio
from my_metrics.no_sorry import NoSorryMetric  # adjust path

async def main():
    metric = NoSorryMetric(
        config={"check_typecheck": True},
        repl_config={"lean_version": "4.24.0"},
    )

    example = {
        "formal_statement_generated": "theorem test : True := by sorry"
    }

    result = await metric.run_check_async(example)
    print(result)  # e.g. {'has_sorry': True}

if __name__ == "__main__":
    asyncio.run(main())
```

---

## 2. Creating an LLM-as-a-Judge Metric

You can also use an LLM to judge the output. This is common for "soft" evaluations, like judging the readability of proofs.

```python
from typing import Any
from leanflow import Metric
import openai  # or another async client

class LLMNoSorryMetric(Metric):
    """Use an LLM to review code that may contain 'sorry'."""

    def __init__(self, metric_config: dict[str, Any] = {}, **shared_dependencies):
        super().__init__(metric_config, **shared_dependencies)
        self.model = metric_config.get("model", "gpt-4")

        # Pull API configuration from shared dependencies
        api_config = shared_dependencies.get("api_config", {})
        self.client = openai.AsyncOpenAI(api_key=api_config.get("api_key"))

    async def run_check_async(self, example: dict[str, Any]) -> dict[str, Any]:
        code = example.get("formal_statement_generated", "")
        has_sorry = "sorry" in code.lower()

        # If there is no 'sorry', we can consider this trivially passing
        if not has_sorry:
            return {"has_sorry": False, "llm_accepts": True}

        prompt = f"""Review the following Lean code that uses 'sorry':

{code}

Is this an acceptable use of 'sorry' in the context of a partially completed proof?
Reply with 'YES' if you consider it acceptable, otherwise 'NO'."""

        response = await self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
        )

        answer = response.choices[0].message.content
        llm_accepts = "YES" in answer.upper()

        return {
            "has_sorry": True,
            "llm_accepts": llm_accepts,
        }
```

---


### 3. Integration with Evaluation CLI

To use these metrics with `leanflow-eval`, you just need to update your YAML configuration file. You don't need to modify LeanFlow's source code.

`config.yaml`:

```yaml
data_path: ./data/examples.json
output_path: ./results

repl_config:
  lean_version: "4.24.0"

api_config:
  api_key:  <KEY>
  base_url: <URL>

metrics:
  - no_sorry
  - sorry_judge

no_sorry:
  source_file: ./my_metrics/no_sorry.py
  class_name: NoSorryMetric
  check_typecheck: true

sorry_judge:
  source_file: ./my_metrics/sorry_judge.py
  class_name: LLMJudgeMetric
  model: "deepseek-math"
```

Run the evaluation:

```bash
leanflow-eval --config config.yaml
```
