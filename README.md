# Genomic Foundationless Models

## Structure

The repository is organized into two main directories:

1. `biotype`: For gene biotype classification experiments.
2. `nt-bench`: For Nucleotide Transformer experiments.

We plan to release more code upon full release, including snesitivity and ancestry experiments.

## Main Entry Points

### Biotype Classification

- `biotype/main_biotype_gencode.py`: Main script for gene biotype classification.

### Nucleotide Transformer Benchmarking

- `nt-bench/main_nt.py`: Main script for running Nucleotide Transformer benchmarks.

## How to Run

1. For biotype classification:
   ```
   python biotype/main_biotype_gencode.py
   ```

2. For nucleotide transformer benchmarking:
   ```
   python nt-bench/main_nt.py
   ```

Both scripts support various configuration options. You can modify the parameters in the entry points.


## Models

The repository supports various transformer-based models for DNA sequence analysis, including:

- Nucleotide Transformer (50M and 500M variants)
- DNABERT-2
- HyenaDNA
- GenALM
- Caduceus

You can select and configure these models in the respective scripts. Mistral model weights will be available upon release.
