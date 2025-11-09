# <img src="docs/assets/simba+_icon.webp" alt="simba+" height="110"/> **SIMBA+**

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