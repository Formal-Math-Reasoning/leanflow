import regex as re
from typing import Optional, Protocol, Union

from ..utils import LeanError, Environment
from ..errors import LeanValueError

BEQ_DEFAULT = {
    "normal": {
        "allowed_tactics": [
            "apply",
            "by_contra",
            "cases\'",
            "constructor",
            "exact",
            "exact?",
            "ext",
            "have",
            "intro",
            "intros",
            "rw",
            "use",
        ],
       "banned_tokens": [
            "sorry",
            "admit",
            "by_contra"
        ],
        "template":"""Given two Lean 4 theorems, please prove `thm_Q` with `thm_P`.
You can only use the following tactics: {ALLOWED_TACTICS}
`thm_P` should be used at least once in the proof.
DO NOT add any extra explanation.
Here are some examples:

Input:
```
import Mathlib

open Topology Filter Real Complex TopologicalSpace Finset
open scoped BigOperators
noncomputable section


theorem thm_P : ¬ ∃ (x : ℚ), ( x ^ 2 = 12 ) :=
sorry

theorem thm_Q (q : ℚ ) :q ^ 2 ≠ 12 := by
```
Output:
```
exact (not_exists.mp thm_P) q
```

---

Input:
```
import Mathlib

open Fintype Subgroup Set Polynomial Ideal
open scoped BigOperators
noncomputable section


theorem thm_P {p q r : ℕ} {G : Type*} [Group G]
  [Fintype G]  (hpqr : p < q ∧ q < r)
  (hpqr1 : p.Prime ∧ q.Prime ∧ r.Prime)(hG : card G = p*q*r) :
  Nonempty (Sylow p G) ∨ Nonempty (Sylow q G) ∨ Nonempty (Sylow r G) :=
sorry

theorem thm_Q {p : ℕ } {q : ℕ } {r : ℕ } {G : Type u_1} [Group G] [Fintype G] (hp : Nat.Prime p) (hq : Nat.Prime q) (hr : Nat.Prime r) (hpq : p < q) (hqr : q < r) (hG : Fintype.card G = p * q * r) :Nonempty (Sylow p G) ∨ Nonempty (Sylow q G) ∨ Nonempty (Sylow r G) := by
```
Output:
```
exact thm_P (And.intro hpq hqr) (And.intro hp (And.intro hq hr)) hG
```

---

Input:
```
import Mathlib

open Fintype Complex Polynomial LinearMap FiniteDimensional Module Module.End
open scoped BigOperators


theorem thm_P {F V : Type*} [AddCommGroup V] [Field F]
  [Module F V] (S T : End F V) :
  (S * T).Eigenvalues = (T * S).Eigenvalues :=
sorry

theorem thm_Q {K : Type v} {V : Type w} [Field K] [AddCommGroup V] [Module K V] (S : Module.End K V) (T : Module.End K V) :Module.End.Eigenvalues (S * T) = Module.End.Eigenvalues (T * S) := by
```
Output:
```
exact @thm_P K V _ _ _ S T
```

---

Input:
```
import Mathlib

open Function Fintype Subgroup Ideal Polynomial Submodule Zsqrtd
open scoped BigOperators
noncomputable section


theorem thm_P
    {p : ℕ} {hp : Nat.Prime p} (h : ∃ r : ℕ, p = 2 ^ r + 1) :
    ∃ (k : ℕ), p = 2 ^ (2 ^ k) + 1 :=
sorry

theorem thm_Q {p : ℕ } (hp : Nat.Prime p) (h : ∃ (r : ℕ ), p = 2 ^ r + 1) :∃ (k : ℕ ), p = 2 ^ 2 ^ k + 1 := by
```
Output:
```
exact @thm_P p hp h
```

---

Input:
```
import Mathlib

open Fintype Set Real Ideal Polynomial
open scoped BigOperators
noncomputable section


theorem thm_P {G : Type*} [Group G]
  [Fintype G] (hG2 : Even (card G)) :
  ∃ (a : G), a ≠ 1 ∧ a = a⁻¹ :=
sorry

theorem thm_Q {G : Type*} [Group G] [Fintype G] (h : Fintype.card G % 2 = 0) :
    ∃ a : G, a ≠ 1 ∧ a = a⁻¹ := by
```
Output:
```
have hG : Even (card G) := by exact?
exact thm_P hG
```

---

According to the task description and examples, given the following two Lean 4 theorems, please prove `thm_Q` with `thm_P`.

Input:
```
{autoformalization_result}
```
Output:
"""
    },
    "all": {
        "allowed_tactics": [],
        "banned_tokens": [],  # No restrictions in this mode
        "template": """Given two Lean 4 theorems, please prove `thm_Q` with `thm_P`.
DO NOT add any extra explanation.
Here are some examples:

Input:
```
import Mathlib

open Topology Filter Real Complex TopologicalSpace Finset
open scoped BigOperators
noncomputable section


theorem thm_P : ¬ ∃ (x : ℚ), ( x ^ 2 = 12 ) :=
sorry

theorem thm_Q (q : ℚ ) :q ^ 2 ≠ 12 := by
```
Output:
```
exact (not_exists.mp thm_P) q
```

---

Input:
```
import Mathlib

open Fintype Subgroup Set Polynomial Ideal
open scoped BigOperators
noncomputable section


theorem thm_P {p q r : ℕ} {G : Type*} [Group G]
  [Fintype G]  (hpqr : p < q ∧ q < r)
  (hpqr1 : p.Prime ∧ q.Prime ∧ r.Prime)(hG : card G = p*q*r) :
  Nonempty (Sylow p G) ∨ Nonempty (Sylow q G) ∨ Nonempty (Sylow r G) :=
sorry

theorem thm_Q {p : ℕ } {q : ℕ } {r : ℕ } {G : Type u_1} [Group G] [Fintype G] (hp : Nat.Prime p) (hq : Nat.Prime q) (hr : Nat.Prime r) (hpq : p < q) (hqr : q < r) (hG : Fintype.card G = p * q * r) :Nonempty (Sylow p G) ∨ Nonempty (Sylow q G) ∨ Nonempty (Sylow r G) := by
```
Output:
```
exact thm_P (And.intro hpq hqr) (And.intro hp (And.intro hq hr)) hG
```

---

Input:
```
import Mathlib

open Fintype Complex Polynomial LinearMap FiniteDimensional Module Module.End
open scoped BigOperators


theorem thm_P {F V : Type*} [AddCommGroup V] [Field F]
  [Module F V] (S T : End F V) :
  (S * T).Eigenvalues = (T * S).Eigenvalues :=
sorry

theorem thm_Q {K : Type v} {V : Type w} [Field K] [AddCommGroup V] [Module K V] (S : Module.End K V) (T : Module.End K V) :Module.End.Eigenvalues (S * T) = Module.End.Eigenvalues (T * S) := by
```
Output:
```
exact @thm_P K V _ _ _ S T
```

---

Input:
```
import Mathlib

open Function Fintype Subgroup Ideal Polynomial Submodule Zsqrtd
open scoped BigOperators
noncomputable section


theorem thm_P
    {p : ℕ} {hp : Nat.Prime p} (h : ∃ r : ℕ, p = 2 ^ r + 1) :
    ∃ (k : ℕ), p = 2 ^ (2 ^ k) + 1 :=
sorry

theorem thm_Q {p : ℕ } (hp : Nat.Prime p) (h : ∃ (r : ℕ ), p = 2 ^ r + 1) :∃ (k : ℕ ), p = 2 ^ 2 ^ k + 1 := by
```
Output:
```
exact @thm_P p hp h
```

---

Input:
```
import Mathlib

open Fintype Set Real Ideal Polynomial
open scoped BigOperators
noncomputable section


theorem thm_P {G : Type*} [Group G]
  [Fintype G] (hG2 : Even (card G)) :
  ∃ (a : G), a ≠ 1 ∧ a = a⁻¹ :=
sorry

theorem thm_Q {G : Type*} [Group G] [Fintype G] (h : Fintype.card G % 2 = 0) :
    ∃ a : G, a ≠ 1 ∧ a = a⁻¹ := by
```
Output:
```
have hG : Even (card G) := by exact?
exact thm_P hG
```

---

According to the task description and examples, given the following two Lean 4 theorems, please prove `thm_Q` with `thm_P`.

Input:
```
{autoformalization_result}
```
Output:
"""
    },
}

def is_valid_lean(
    env_state,
    start_line: Optional[int] = None, 
    end_line: Optional[int] = None, 
    allow_sorry: bool = True,  
):
    """Given an environment state, checks if there are any errors in the Lean output.

    Args:
        env_state: The environment state to check.
        start_line: The start line of the code range to check.
        end_line: The end line of the code range to check.
        allow_sorry: Whether to allow sorries in the code range.
    
    Returns:
        (bool): True if there are no errors in the code range, False otherwise.
    """
    
    if isinstance(env_state, LeanError):
        return False

    if env_state is None:
        return False
    
    # Filter messages and sorries that intersect with the specified code range
    errors = [m for m in env_state.messages if message_intersects_code(m, start_line, end_line) and m.severity == "error"]
    sorries = [m for m in env_state.sorries if message_intersects_code(m, start_line, end_line)]
    
    return not errors and (allow_sorry or not sorries)

def message_intersects_code(
        message,
        start_line: Optional[int],
        end_line: Optional[int]
    ) -> bool:
    """Check if the message intersects with the specified code range.

    Args:
        message: The message to check.
        start_line: The start line of the code range to check.
        end_line: The end line of the code range to check.
    
    Returns:
        (bool): True if the message intersects with the specified code range, False otherwise.
    """
    res = True
    if start_line is not None and message.endPos:
        res = res and message.endPos.line >= start_line
    if end_line is not None and message.startPos:
        res = res and message.startPos.line <= end_line
    return res

def split_conclusion(declaration: str, start: int = 0) -> Optional[int]:
    """Split the conclusion of a declaration.

    Args:
        declaration: The declaration to split.
        start: The start index of the declaration.
    
    Returns:
        int | None: The index of the conclusion, or None if the declaration does not have a conclusion.
    """
    counters = {"(": 0, "{": 0, "[": 0}
    closing = {")": "(", "}": "{", "]": "["}
    for i, c in enumerate(declaration[start:]):
        if c in counters:
            counters[c] += 1
        elif c in [")", "}", "]"]:
            counters[closing[c]] -= 1
        if all([v == 0 for v in counters.values()]) and c == ":":
            return i + start
    return None

def indent_code(code: str, nb_spaces: int = 2) -> str:
    """Indent the code by a given number of spaces.

    Args:
        code: The code to indent.
        nb_spaces: The number of spaces to indent.
    
    Returns:
        (str): The indented code.
    """
    return "\n".join(" " * nb_spaces + line for line in code.split("\n"))

def split_implementation(declaration: str, start: int = 0) -> Optional[str]:
    """Split the implementation of a declaration.

    Args:
        declaration: The declaration to split.
        start: The start index of the declaration.
    
    Returns:
        (Optional[str]): The implementation of the declaration.
    """
    if ":=" in declaration:
        indices = set([m.start() for m in re.finditer(r":=", declaration)])

        for keyword in ["let", "haveI"]:
            regex = rf"{keyword}\s+\S*?\s*(:=)"
            decl_indices = set([m.start(1) for m in re.finditer(regex, declaration)])
            indices = indices - decl_indices

        counters = {"(": 0, "{": 0, "[": 0}
        closing = {")": "(", "}": "{", "]": "["}
        for i, c in enumerate(declaration[start:]):
            if c in counters:
                counters[c] += 1
            elif c in [")", "}", "]"]:
                counters[closing[c]] -= 1
            if all([v == 0 for v in counters.values()]) and (i + start) in indices:
                return i + start
    return None

def remove_lean_comments(lean_code: str) -> Optional[str]:
    """Remove the comments from a Lean code.

    Args:
        lean_code: The Lean code to remove comments from.
    
    Returns:
        (Optional[str]): The Lean code without comments, or None if an error occurs.
    """
    try:
        comment_ranges = lean_comments_ranges(lean_code)

        new_lean_code = ""
        prev_start = 0
        for start, end in comment_ranges:
            new_lean_code += lean_code[prev_start:start]
            prev_start = end

        new_lean_code += lean_code[prev_start:]
        return new_lean_code

    except Exception:
        return None

def clean_theorem_string(theorem_string: str, new_theorem_name: str = "dummy", add_sorry: bool = True) -> Optional[str]:
    """Clean a theorem string by removing the proof, comments, and updating the theorem name.
    This method assumes that no other declarations are present in the theorem string.
    
    Args:
        theorem_string: The theorem string to clean.
        new_theorem_name: The new name for the theorem.
        add_sorry: Whether to add a sorry at the end of the theorem.
    
    Returns:
        (Optional[str]): The cleaned theorem string, or None if an error occurs.
    """
    try:
        clean_formal = remove_lean_comments(theorem_string)
        if clean_formal is None:
            raise LeanValueError("Comment removal failed.")
        clean_formal = clean_formal.strip()

        theorem_decl_keywords = "|".join(["theorem", "lemma", "example"])
        re_match = re.search(rf"\b{theorem_decl_keywords}\s", clean_formal)
        if re_match is None:
            raise LeanValueError("Theorem declaration keyword not found.")
        idx_theorem = re_match.start()
        clean_formal = clean_formal[idx_theorem:]

        idx_implement = split_implementation(clean_formal)
        if idx_implement is not None:
            clean_formal = clean_formal[:idx_implement].strip()
        if clean_formal.strip().startswith("example"):
            clean_formal = re.sub(r"^[^\s]+", "", clean_formal).strip()
        else:
            clean_formal = re.sub(r"^[^\s]+", "", clean_formal).strip()
            clean_formal = re.sub(r"^[^\s:({\[]+", "", clean_formal).strip()
        clean_formal = f"theorem {new_theorem_name} " + clean_formal
        if add_sorry:
            clean_formal += " := sorry"
        return clean_formal
    except Exception:
        return None

def lean_comments_ranges(
    lean_code: str, multiline_comment_suffix: str = "", remove_single_line_comments: bool = True
) -> list[tuple[int, int]]:
    """Extract the ranges of Lean comments from a Lean code snippet.

    Args:
        lean_code: The Lean code to extract comments from.
        multiline_comment_suffix: The suffix of multiline comments.
        remove_single_line_comments: Whether to remove single line comments.
    
    Returns:
        (list[tuple[int, int]]): The ranges of Lean comments.
    """
    open_comment_indices = [m.start() for m in re.finditer(r"/-" + multiline_comment_suffix, lean_code)]
    close_comment_indices = [m.start() + len(multiline_comment_suffix) + 2 for m in re.finditer(multiline_comment_suffix + r"-/", lean_code)]

    if len(open_comment_indices) == len(close_comment_indices) + 1:
        close_comment_indices.append(len(lean_code))

    elif len(open_comment_indices) + 1 == len(close_comment_indices):
        open_comment_indices.insert(0, 0)

    elif len(open_comment_indices) != len(close_comment_indices):
        raise LeanValueError("Mismatched open and close comment indices.")

    multiline_comment_ranges = list(zip(open_comment_indices, close_comment_indices))

    if remove_single_line_comments:
        single_line_comment_ranges = [(m.start(), lean_code.find("\n", m.start())) for m in re.finditer(r"--", lean_code)]
        multiline_comment_ranges += single_line_comment_ranges

    comment_ranges = sorted(multiline_comment_ranges, key=lambda x: x[0])
    merged_comment_ranges = []
    for start, end in comment_ranges:
        if merged_comment_ranges and start <= merged_comment_ranges[-1][1]:
            merged_comment_ranges[-1] = (merged_comment_ranges[-1][0], max(merged_comment_ranges[-1][1], end))
        else:
            merged_comment_ranges.append((start, end))

    return merged_comment_ranges

def extract_last_theorem(lean_code: str) -> int:
    """Extract the last theorem from a Lean code snippet. It assumes that the Lean code snippet ends with a theorem.

    Args:
        lean_code: The Lean code to extract the last theorem from.
    
    Returns:
        (int): The index of the last theorem.
    """
    comments_ranges = lean_comments_ranges(lean_code)

    # find last theorem by looking for `theorem` keyword surrounded by whitespaces, or by being at the beginning of the string
    theorem_decl_keywords = ["theorem", "lemma", "example"]
    theorem_indices = []
    for keyword in theorem_decl_keywords:
        theorem_indices += [m.start() for m in re.finditer(rf"\b{keyword}\s", lean_code)]

    # remove matches that are inside comments
    theorem_indices = [i for i in theorem_indices if not any(start <= i <= end for start, end in comments_ranges)]

    if not theorem_indices:
        raise LeanValueError(f"No theorem found in the provided Lean code:\n{lean_code}")

    return theorem_indices[-1]

def clean_last_theorem_string(lean_code: str, new_theorem_name: str = "dummy", add_sorry: bool = False) -> str:
    """Clean the last theorem string from a Lean code snippet. It assumes that the Lean code snippet ends with a theorem.

    Args:
        lean_code: The Lean code to clean the last theorem from.
        new_theorem_name: The new name for the theorem.
        add_sorry: Whether to add a sorry at the end of the theorem.
    
    Returns:
        (str): The cleaned Lean code.
    """
    idx_last_theorem = extract_last_theorem(lean_code)
    clean_thm = clean_theorem_string(lean_code[idx_last_theorem:], new_theorem_name, add_sorry=add_sorry)
    if clean_thm is not None:
        return lean_code[:idx_last_theorem] + clean_thm

    raise LeanValueError(f"Theorem extraction failed for the following Lean code:\n{lean_code}")

class LeanRunner(Protocol):
    async def run(
        self, commands: Union[str, list[str]], env: Optional[Union[Environment, int]] = None
    ) -> Union[Environment, LeanError, list[Union[Environment, LeanError]]]:
        """Wrapper for REPL or Client run method.

        Args:
            commands: The Lean commands to run.
            env: The environment to run the commands in.
        
        Returns:
            Union[Environment, LeanError, list[Union[Environment, LeanError]]]: The result of the Lean commands.
        """
        ...


def _extract_exact_proof(
    env_state: Environment,
    proof_start_line: Optional[int] = None,
    proof_end_line: Optional[int] = None,
) -> Optional[str]:
    """Extracts a proof tactic suggested by 'Try this:'.

    Args:
        env_state: The environment state.
        proof_start_line: The start line of the proof.
        proof_end_line: The end line of the proof.
    
    Returns:
        (Optional[str]): The extracted proof tactic.
    """
    if not isinstance(env_state, Environment):
        return None
    for message in env_state.messages:
        if message_intersects_code(message, proof_start_line, proof_end_line):
            if message.severity == "info" and message.data.startswith("Try this:"):
                return message.data.split("Try this:")[1].strip()
    return None

async def check_proof_sub(
    runner: LeanRunner,
    formal_code: str,
    context_env: Optional[int],
    formal_2_start_line: int,
    proof: str,
    indent_level: int = 2,
) -> Optional[str]:
    """Runs Lean code appended with a given proof and checks its validity.

    Args:
        runner: The Lean runner.
        formal_code: The formal code.
        context_env: The context environment.
        formal_2_start_line: The start line of the formal code.
        proof: The proof to check.
        indent_level: The indentation level.
    
    Returns:
        (Optional[str]): The proof tactic if the proof is valid, otherwise None.
    """

    command = formal_code + indent_code("\nsymm_saturate\n", indent_level) + indent_code("\nintros\n" + proof, indent_level)

    try:
        lean_output = await runner.run(command, env=context_env)

        if proof == "sorry":
            if is_valid_lean(lean_output, start_line=formal_2_start_line):
                return proof
            return None

        if is_valid_lean(lean_output, start_line=formal_2_start_line, allow_sorry=False):
            if proof == "exact?":
                return _extract_exact_proof(lean_output, proof_start_line=formal_2_start_line)
            return proof
    except Exception as e:
        from ..utils import logger
        logger.error(f"check_proof_sub failed: {e}")
        return None