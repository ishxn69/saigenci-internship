# What I did in my internship at SAiGENCI - AMP Prediction with Graph Neural Networks

Here are the details about my internship at the South Australian immunoGENomics Cancer Institute.

## My Experience

Antimicrobial peptides are short proteins with natural antibiotic properties and are a promising alternative to conventional antibiotics as drug-resistant pathogens become more prevalent. Identifying new AMPs through wet-lab experiments is slow and expensive; computational classifiers can pre-screen large candidate libraries before synthesis.

This project converts amino acid sequences into molecular graphs (atoms = nodes, bonds = edges) using RDKit and trains three GNN architectures to perform binary classification: AMP or Non-AMP. The dataset is imbalanced (~5:1 Non-AMP:AMP), so **PR-AUC** (Precision-Recall AUC) and **Balanced Accuracy** are the primary evaluation metrics.

## Table of Contents

1. [My Internship Experience](#my-internship-experience)
2. [Model Architecture Diagram](#architecture-diagram)
3. [Results](#results)
4. [Coding Practices](#coding-practices)
5. [Libraries Used](#libraries-used)
6. [Quickstart](#quickstart)
7. [HPC/SLURM Notes](#hpc--slurm-notes)

---

## My Internship Experience

### Internship: South Australian Immunogenomics Cancer Institute (SAiGENCI)

**Project:** Graph-based deep learning to classify short peptides as antimicrobial (AMP) or non-antimicrobial.

### Process

1. **Literature review:** Read research papers on current approaches. Held regular meetings with my supervisor throughout.
2. **Tool exploration:** Studied the lab's in-house tools (structure, usage, how they approach similar problems) and related computational biology tools such as Google DeepMind's AlphaFold.
3. **CNN baseline work:** Learned how transformers and attention work. Implemented a CNN on a protein dataset from my supervisor, then tuned hyperparameters and loss functions. Learned to evaluate with AUC-ROC and to benchmark new models against prior work.
4. **Graph neural networks:** Experimented with GCN, GIN and TransformerConv. Used AI-assisted coding for the first time, while keeping a full understanding of the codebase. Wrote SLURM scripts to run jobs on the lab's HPC GPUs.
5. **Building the graph models:**
   - Converted highly imbalanced peptide sequences into **SMILES** strings (text notation for molecular structure), then into molecular graphs with **RDKit**.
   - Used 33 node features, one-hot encoded: atomic symbol, degree, attached hydrogens, chirality and others.
   - Added global descriptors (molecular weight, logP) to give broader structural context.
   - Trained with **BCE with logits loss**, using `pos_weight` (Non-AMP : AMP ratio) to handle class imbalance.
   - Used **JK (Jumping Knowledge) Readout**, which combines node representations from several GNN layers instead of only the last one. This keeps both local and wider structural information and reduces over-smoothing.

### Result

My models did not beat the baseline, **AMPLiT** (a multi-head attention model trained only on AMP amino-acid sequences). The main reasons were limited time alongside academic commitments, and limited domain knowledge and research experience.

### Key learnings

- Reading papers and benchmarking against baselines
- Working with imbalanced data and choosing evaluation metrics
- Molecular graph construction with RDKit
- GNN architectures (GCN, GIN, TransformerConv, JK readout)
- Running GPU jobs on an HPC cluster with SLURM
- Using AI coding tools without losing understanding of the code

**Tech:** Python, PyTorch, PyTorch Geometric, RDKit, SLURM

---

## Architecture Diagram:

<img width="1377" height="924" alt="model_architecture" src="https://github.com/user-attachments/assets/65b62aba-2ded-4c5d-a6d1-e6d43458b650" />

---

## Results

| Model | Test PR-AUC |
|-------|-------------|
| GCNConv (`globaldescriptors_v3`) | 0.7959 |
| GIN (`globaldescriptors_v2`) | 0.7877 |
| TransformerConv (`v1`) | 0.7186 |
| AMPLiT (reference baseline) | 0.9486 |

---

## Coding Practices

This section outlines the engineering standards, collaboration protocols, and security practices maintained throughout this project. These guidelines ensure the codebase remains scalable, highly maintainable, secure, and optimized for collaborative development in enterprise or open-source environments.

### General Guidelines

* For the full directory structure, data flow diagram, model design decisions, and config reference, see [ARCHITECTURE.md - Is not included in this repo].
* **Modular & Extensible Architecture:** The codebase is strictly decoupled into independent modules (e.g., data pipelines, model architectures, training configurations, and evaluation metrics). This high degree of modularity ensures that the system is highly extensible, allowing team members to effortlessly isolate variables, test new deep learning/machine learning architectures, and experiment with novel hyperparameters without introducing regressions into production-ready pipelines.
* **Industry-Standard Code Style:** To ensure seamless onboarding and readability for cross-functional engineering teams, all code adheres strictly to the **Google Style Guides** (including the Google Python Style Guide) and standard naming conventions (PascalCase for classes, snake_case for variables/functions, and UPPER_CASE for constants).
* **Static Analysis & Linting:** Code formatting and quality are maintained using automated linters and formatters aligned with Google’s strict readability requirements to guarantee consistent syntax and structure across the entire repository.

### Pull Request Instructions

To maintain a clean main branch and ensure thorough asynchronous code reviews, all contributions must follow a disciplined Pull Request (PR) workflow. PRs must be descriptive, self-documenting, and heavily informed by Google’s software engineering best practices.

#### 1. Pull Request Template & Criteria
Every PR description must explicitly detail the following three core sections to provide reviewers with full context:
* **Context / Purpose:** What is the underlying reason for this change? What specific business logic, user story, or research requirement does it address?
* **Solution Description:** What problem does this PR solve, and how does it solve it? Detail the architectural changes, algorithmic choices, or refactoring logic introduced.
* **Drawbacks & Trade-offs:** What are the potential drawbacks, side effects, or engineering trade-offs according to Google coding standards? (e.g., increased memory footprint, computational complexity, breaking changes, or technical debt incurred).

#### 2. Branch & PR Naming Conventions
To maintain a transparent and filterable commit history, branches and pull requests follow a strict structural prefix convention optimized for automated CI/CD parsing and semantic versioning:
* `feature/name-of-capability` — For implementing new features, components, or model layers.
* `bugfix/issue-description` — For resolving bugs, computational bottlenecks, or logical errors.
* `docs/documentation-update` — For modifications to READMEs, docstrings, inline comments, or wiki pages.
* `refactor/restructure-code` — For structural code modifications that neither fix bugs nor add features.
* `chore/dependency-upgrade` — For maintenance tasks, updating packages, or adjusting CI/CD workflows.

### Security for Public Repository

Repository hygiene and cryptographic secret security are foundational to our deployment workflow, especially when transitioning codebases from private development environments to public-facing platforms.

* **Secret Scoping & Environment Isolation:** Hardcoded credentials, API keys, private server paths, and sensitive configuration data are strictly prohibited within the codebase. Environment variables and secure configurations are utilized natively.
* **History Purging via `git-filter-repo`:** In the event that sensitive historical data, heavy debugging artifacts, or configuration files are accidentally tracked, standard `git rm` commands are insufficient as they leave data in the Git history. To permanently sanitize the repository, **`git-filter-repo`** is deployed.
* **Immutable History Rewriting:** `git-filter-repo` is used to systematically analyze the object graph, completely scrub sensitive files from all historical commits, and rewrite the Git history safely without corrupting tags or references. This ensures that the public repository contains zero historical leaks or traces of internal infrastructure details, making it fully production-safe and compliant with modern security audits.

---

## Libraries Used

**Python:** 3.10

Conda packages:
- numpy 1.24.4
- pandas 2.1.4
- scikit-learn 1.3.2
- pyarrow 14.0.2
- rdkit 2023.03.3
- matplotlib 3.8.4
- seaborn 0.13.2
- joblib 1.4.2
- scipy 1.9.3
- pytorch 1.13.1
- cudatoolkit 11.6
- pyyaml 6.0.3

PyG extension packages (pip, pinned to cu116 wheels):
- torch_scatter 2.1.1
- torch_sparse 0.6.17
- torch_cluster 1.6.1
- torch_spline_conv 1.2.2
- torch-geometric 2.3.1

---

## Quickstart

### 1. Clone
```bash
git clone <repo-url>
cd graph-amp-pred
```

### 2. Set up the environment (HPC — CUDA 11.6)

> The HPC NVIDIA driver is CUDA 11.6. Packages targeting CUDA 12+ will fail to load.

```bash
# Clean any corrupted cached packages
conda clean --packages -y

# Create environment with PyTorch 1.13.1 + CUDA 11.6
conda create -n amp_gnn python=3.10 \
  pytorch==1.13.1 cudatoolkit=11.6 \
  -c pytorch -c conda-forge -y

# Replace conda's (potentially CPU-only) PyTorch with the explicit CUDA wheel
pip install torch==1.13.1+cu116 \
  --extra-index-url https://download.pytorch.org/whl/cu116

# Install scientific and cheminformatics packages
conda install -n amp_gnn \
  numpy=1.24.4 pandas=2.1.4 \
  scikit-learn=1.3.2 \
  pyarrow=14.0.2 \
  rdkit=2023.03.3 \
  matplotlib=3.8.4 seaborn=0.13.2 \
  joblib=1.4.2 \
  scipy=1.9.3 \
  -c conda-forge -y

conda install -n amp_gnn -c conda-forge pyyaml=6.0.3 -y

# Install PyG extension packages (pinned to pre-built cu116 wheels)
pip install \
  "torch_scatter==2.1.1" \
  "torch_sparse==0.6.17" \
  "torch_cluster==1.6.1" \
  "torch_spline_conv==1.2.2" \
  -f https://data.pyg.org/whl/torch-1.13.1+cu116.html

pip install torch-geometric==2.3.1
```

**Verify the installation:**
```bash
python -c "import torch; print(torch.__version__, torch.version.cuda)"
# Expected: 1.13.1  11.6

python -c "import torch_geometric; print(torch_geometric.__version__)"
# Expected: 2.3.1

python -c "import rdkit; print(rdkit.__version__)"
# Expected: 2023.03.3

python -c "import scipy; print(scipy.__version__)"
# Expected: 1.9.3
```

### 3. Preprocess (run once)
```bash
python data_pipeline/preprocess.py
```
Reads raw CSVs from `data/raw/`, converts sequences to SMILES, and saves SMILES-enriched Parquet files to `data/processed/`. Safe to delete and regenerate.

### 4. Train
```bash
# Locally
python models/gcn/train.py --config models/gcn/configs/globaldescriptors_v3.yaml
python models/gin/train.py --config models/gin/configs/globaldescriptors_v2.yaml
python models/transfconv/train.py --config models/transfconv/configs/v1.yaml

# On HPC via SLURM
sbatch jobs/gcn_globaldescriptors_v3_job.sh
sbatch jobs/gin_globaldescriptors_v2_job.sh
sbatch jobs/transfconv_v1_job.sh
```
Checkpoints are saved to `checkpoints/<model>/` every 10 epochs. Two best-checkpoint files are kept per run: `<model>_best_bal_acc.pt` and `<model>_best_auprc.pt`. Results are appended to a CSV under `results/`.

### 5. Evaluate a saved checkpoint
```bash
python models/gcn/eval.py --config models/gcn/configs/globaldescriptors_v3.yaml
python models/gin/eval.py --config models/gin/configs/globaldescriptors_v2.yaml
python models/transfconv/eval.py --config models/transfconv/configs/v1.yaml
```

### 6. Run tests
```bash
pytest tests/gcn/
pytest tests/gin/
pytest tests/transfconv/
```
Test fixtures (small stratified Parquets) must be generated first by running `tests/gcn/create_test_fixtures.ipynb`.

---

## HPC / SLURM notes

- Use `conda run -n amp_gnn python <script>` instead of `conda activate` in job scripts — `conda activate` is unreliable in non-interactive SLURM shells.
- Always export `LD_LIBRARY_PATH` after `module load` (already done in all `jobs/*.sh`). The HPC system `libstdc++` only provides `GLIBCXX ≤ 3.4.28`, but scipy and PyG extensions require `GLIBCXX_3.4.29`:
  ```bash
  export LD_LIBRARY_PATH="$(conda run -n amp_gnn python -c 'import sys; print(sys.prefix)')/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
  ```
- See `ENVIRONMENT.md` for the full SLURM job script guidance.

---


