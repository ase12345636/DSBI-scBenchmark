# Single-Cell Analysis Benchmark

This repository contains independent benchmarking frameworks for single-cell analysis. Each suite provides method-specific wrappers, repeated execution, resource monitoring, and a shared downstream evaluation workflow.

Environment setup, input requirements, and command-line interfaces differ between the two suites. Select the benchmark corresponding to your task and follow its dedicated documentation.

## Benchmark Suites

| Suite | Scope |
|---|---|
| [Batch Correction](batch_effect/README.md) | Batch-effect correction and data-integration methods. |
| [Cell-Type Annotation](celltype_annotation/README.md) | Supervised reference-to-query annotation methods. |

## Repository Structure

```text
scBenchmark/
├── batch_effect/
│   ├── README.md
│   └── run_script.sh
└── celltype_annotation/
    ├── README.md
    └── run_script.sh
```

## Usage

1. Choose the benchmark suite matching your task.
2. Open its documentation:
   - [Batch correction setup and usage](batch_effect/README.md)
   - [Cell-type annotation setup and usage](celltype_annotation/README.md)
3. Install the method-specific environments described in that document.
4. Run the benchmark using the corresponding suite entry point.

