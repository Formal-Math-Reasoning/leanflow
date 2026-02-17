# Metrics API Reference

The Metrics module provides a suite of tools for evaluating formal mathematics. These classes allow you to assess the syntactic correctness, equivalence, and quality of Lean 4 code, ranging from simple typechecking to semantic comparisons using LLMs.

## Base Metric

=== "Metric"
    ::: leanflow.Metric
        options:
          show_root_heading: true
          show_source: false

=== "BatchMetric"
    ::: leanflow.BatchMetric
        options:
          show_root_heading: true
          show_source: false

---

## Interactive Metrics

These metrics run on individual examples and typically rely on the Lean REPL to verify logic or proofs.

=== "TypeCheck"
    ::: leanflow.TypeCheck
        options:
          show_root_heading: true
          show_source: false

=== "BEqPlus"
    ::: leanflow.BEqPlus
        options:
          show_root_heading: true
          show_source: false

=== "BEqL"
    ::: leanflow.BEqL
        options:
          show_root_heading: true
          show_source: false

=== "EquivRfl"
    ::: leanflow.EquivRfl
        options:
          show_root_heading: true
          show_source: false

---

## LLM & Batch Metrics

These metrics leverage LLMs to perform semantic evaluations. They are designed to process inputs in batches.

=== "LLMGrader"
    ::: leanflow.LLMGrader
        options:
          show_root_heading: true
          show_source: false
          
=== "BEq"
    ::: leanflow.BEq
        options:
          show_root_heading: true
          show_source: false

=== "ConJudge"
    ::: leanflow.ConJudge
        options:
          show_root_heading: true
          show_source: false
