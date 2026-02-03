# Evaluation CLI

The LeanFlow Evaluation CLI (`leanflow-eval`) is a command‑line tool for running evaluation on autoformalization datasets. It's primary goal is facilitating research workloads.

It allows you to run metrics like typecheck over large datasets of Lean statements. It handles parallelization, error logging, and results aggregation automatically.

---

## 1. Usage

```bash
leanflow-eval --config config.yaml
```

---

## 2. Configuration File

The CLI is controlled entirely by a YAML configuration file.

### 2.1 General Settings

```yaml
# Input: Path to JSON file or HuggingFace dataset directory
data_path: ./data/examples.json

# Output: Directory where results will be saved
output_path: ./results

# Overwrite existing results (default: false)
overwrite_all: false

# Logging verbosity (debug, info, warning, error)
log_level: info

# Devices to use (comma-separated list of device IDs)
devices: "0,1"
```

### 2.2 Execution Backend

You must define how LeanFlow executes the code. You can either run Lean **locally** (spawning new processes) or connect to a running **server**.

=== "Local"

    ```yaml
    repl_config:
      # Lean version to use (auto-downloaded if needed)
      lean_version: "4.24.0"

      # Optional: Path to a local Lean project
      # project_path: /home/user/my-lean-project

      # Timeout per statement (seconds)
      timeout: 300

    use_multiprocessing: true
    ```

=== "Server"

    ```yaml
    repl:
      # URL of the LeanFlow server
      base_url: "http://localhost:8000"
      
      # Client-side timeout (seconds)
      timeout: 300
    ```
    - You must start a LeanFlow server separately (see [Client–Server Architecture](server.md)).
    - The CLI sends all Lean code to that server via the `Client`.

### 2.3 Metrics

Define which metrics to compute. You can run multiple metrics in a single pass.

```yaml
metrics:
  - typecheck
  - beq_plus

# Optional: Configuration for individual metrics 
typecheck:
  overwrite: true  # Overwrite results even if they exist
```

### 2.4 Preprocessing Options

LLM output often lacks necessary imports. You can inject them globally using set_header.

```yaml
# Option 1: Replace all headers with a specific import block
set_header: |
  import Mathlib
  open Nat

# Option 2: Remove all headers
# remove_header: true

# Option 3: Extract headers from generated code
# extract_header: true
```

- `set_header`: ensures all instances share the same imports and setup.
- `remove_header`: strips all user‑provided headers.
- `extract_header`: attempts to separate user code into header + body.

---

## 3. Data Format

The CLI expects either:

- a JSON file containing a list of objects, or
- a Hugging Face dataset directory.

Each metric requires certain fields to be present in each data item.

### Required Fields by Metric

| Metric      | Required Fields                                | Description                                  |
|------------|-----------------------------------------------|----------------------------------------------|
| `typecheck` | `formal_statement`                            | Lean statement to type‑check                 |
| `beq_plus`  | `formal_statement`, `formal_statement_generated` | Ground truth and generated statement         |
| `beq_l`     | `formal_statement`, `formal_statement_generated` | Ground truth and generated statement         |
| `equiv_rfl` | `formal_conjecture`, `formal_conjecture_generated` | Ground truth and generated conjecture   |

*(Adjust names as needed to match your actual metric implementations.)*

### Example JSON

```json
[
  {
    "formal_statement": "theorem add_zero_nat (n : Nat) : n + 0 = n := by rfl",
    "formal_statement_generated": "theorem add_zero_nat (n : Nat) : 0 + n - 0 = n := by rfl",
    "header": "import Mathlib"
  }
]
```

---

## 4. Output Format

Results are saved as JSON files in the `output_path`. The tool adds a boolean field for each metric run.

Example output:

```json
[
  {
    "formal_statement": "...",
    "formal_statement_generated": "...",
    "typecheck": true,
    "beq_plus": false,
  }
]
```

---

## 5. Custom Metrics

You can plug in your own evaluation logic without modifying the LeanFlow source code.

- Create a Python file (e.g., my_metric.py) with your metric class.

- Register it in the YAML config.


```yaml
metrics:
  - my_custom_metric

my_metric:
  source_file: ./metrics/my_metric.py
  class_name: MyCustomMetric
  my_parameter: 42
```

Your `MyCustomMetric` class should follow the expected interface used by the CLI (e.g. a `compute(...)` or similar method).  
See [Custom Metrics](custom_metrics.md) for a full walkthrough.