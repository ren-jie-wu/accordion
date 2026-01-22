# <img src="docs/assets/simba+_icon.webp" alt="simba+" height="110"/> **SIMBA+**

[![Ask DeepWiki](https://deepwiki.com/badge.svg)](https://deepwiki.com/ren-jie-wu/simba-plus)
[![CI](https://github.com/ren-jie-wu/simba-plus/actions/workflows/ci.yml/badge.svg)](https://github.com/ren-jie-wu/simba-plus/actions/workflows/ci.yml)

`SIMBA+`, a probabilistic graph framework that integrates **single-cell multiomics** with **GWAS** to:
1) **Map regulatory elements and disease variants to target genes** in specific cellular contexts through metapath analysis, and
2) **Decompose complex trait heritability** at **single-cell resolution**.

## Installation
```
git clone -b dev git@github.com:pinellolab/simba-plus.git
cd simba-plus
pip install .
```
## Tutorials
- [SNP-gene link prediction tutorial](notebooks/tutorial-eqtl.ipynb)
- [Element-gene link prediction tutorial](notebooks/tutorial-crispr.ipynb)

## Usage
See [CLI interface](docs/CLI.md) for running SIMBA+ on AnnData input.  
Also see [API reference](https://pinellolab.github.io/simba-plus/api/index.html).

## Contributing

1. Clone and fork the repository
2. Run
   ```
   pip install -e .
   ```
   every time making changes to the code. Then we can run
   ```
   simba+ <subcommand> ...
   ```
   to test the changes.
3. Commit and push the changes to the forked repository.
