# <img src="docs/assets/accordion_icon.png" alt="accordion" height="110"/> **ACCORDION**

[![Ask DeepWiki](https://deepwiki.com/badge.svg)](https://deepwiki.com/ren-jie-wu/accordion)
[![CI](https://github.com/ren-jie-wu/accordion/actions/workflows/ci.yml/badge.svg)](https://github.com/ren-jie-wu/accordion/actions/workflows/ci.yml)

`ACCORDION` is a multi-condition extension of SIMBA+ (single-cell co-embedding framework) that
- produces a well-integrated cell embedding and interpretable feature-level outputs, while
- removing batch effects without erasing true biological signals, including both condition effects and within-condition cell-type structure, and
- providing diagnostics and recommendations for tuning alignment strength.

See the original SIMBA+ code [here](https://github.com/pinellolab/simba-plus).

## Installation
```
git clone -b dev git@github.com:ren-jie-wu/accordion.git
cd accordion
pip install .
```

## Usage
See [CLI interface](docs/CLI.md) for running ACCORDION via command line.

## Contributing

1. Clone and fork the repository
2. Run
   ```
   pip install -e .
   ```
   to make sure packages are installed and changes are reflected. Then run
   ```
   simba+ <subcommand> ...
   ```
   to test the changes.
3. Commit and push the changes to the forked repository.
