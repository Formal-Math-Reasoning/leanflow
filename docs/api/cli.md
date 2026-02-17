# Evaluation CLI Reference

The `leanflow-eval` command-line tool runs evaluation metrics on autoformalization datasets.

## Command Syntax

```bash
leanflow-eval --config <path_to_config.yaml>
```

Or using the Python module:

```bash
python -m leanflow.evaluate_cli --config <path_to_config.yaml>
```

## Configuration Reference

### Top-Level Options

| Option | Type | Required | Description |
|--------|------|----------|-------------|
| `data_path` | string | Yes | Path to input data (JSON file or HuggingFace dataset) |
| `output_path` | string | Yes | Directory for output results |
| `overwrite_all` | bool | No | Overwrite existing results (default: `false`) |
| `log_level` | string | No | Logging level: `debug`, `info`, `warning`, `error` |
| `use_multiprocessing` | bool | No | Enable parallel processing (default: `true`) |
| `devices` | string | No | CUDA devices, e.g., `"0,1"` |

### REPL Configuration

Configure how LeanFlow connects to Lean:

```yaml
# Local mode
repl_config:
  lean_version: "4.21.0"
  require_mathlib: true
  timeout: 300
  # project_path: /path/to/project  # Optional

# OR Server mode
repl_config:
  base_url: "http://localhost:8000"
  timeout: 300
```

| Option | Type | Description |
|--------|------|-------------|
| `lean_version` | string | Lean version (local mode) |
| `require_mathlib` | bool | Include Mathlib (local mode) |
| `project_path` | string | Path to Lean project (local mode) |
| `base_url` | string | Server URL (server mode) |
| `timeout` | int | Command timeout in seconds |

### Metrics Configuration

List metrics to run and their individual configurations:

```yaml
metrics:
  - typecheck
  - beq_plus
  - my_custom_metric

typecheck:
  overwrite: true

my_custom_metric:
  source_file: ./my_metrics/custom.py
  class_name: MyCustomMetric
  custom_param: 42
```

### Built-in Metrics

| Metric | Description | Required Fields |
|--------|-------------|-----------------|
| `typecheck` | Checks if statement typechecks | `formal_statement` |
| `beq_plus` | Behavioral equivalence (BEq+) | `formal_statement`, `formal_statement_generated` |
| `beq_l` | Logical equivalence (BEqL) | `formal_statement`, `formal_statement_generated` |
| `equiv_rfl` | Reflexive equivalence | `formal_conjecture`, `formal_conjecture_generated` |

### Custom Metrics

Load custom metrics from Python files:

```yaml
my_metric:
  source_file: /absolute/path/to/metric.py
  class_name: MyMetricClass
  # Additional params passed to __init__
  param1: value1
```

### Preprocessing Options

Control how Lean code is preprocessed:

```yaml
# Replace all headers
set_header: |
  import Mathlib
  open Nat

# OR remove headers
remove_header: true

# OR extract headers from code
extract_header: true
```

### LLM Configuration

For LLM-based metrics:

```yaml
api_config:
  base_url: "https://api.openai.com/v1"
  api_key: "sk-..."

sampling_params:
  temperature: 0.7
  max_tokens: 1024
```

## Complete Example

```yaml
# config.yaml
data_path: ./data/test_set.json
output_path: ./results
log_level: info
use_multiprocessing: true

repl_config:
  lean_version: "4.21.0"
  require_mathlib: true
  timeout: 300

metrics:
  - typecheck
  - beq_plus

typecheck:
  overwrite: false

beq_plus:
  overwrite: false
```

## Output Format

Results are saved as JSON in `output_path`:

```json
[
  {
    "formal_statement": "theorem t : 1 + 1 = 2 := rfl",
    "formal_statement_generated": "theorem t : 1 + 1 = 2 := by rfl",
    "typecheck": true,
    "beq_plus": true
  }
]
```

## See Also

- [Evaluation Metrics](../metrics.md) – User guide for metrics
- [Custom Metrics](../custom_metrics.md) – Creating custom metrics
