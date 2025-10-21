# GG24 Deep Funding Market Weights 🏗️

This repository contains the code for the initial weights and selection for the[Developer Tooling & Infrastructure Deep Funding Round](https://gitcoin.notion.site/GG24-Developer-Tooling-and-Infrastructure-Deep-Funding-Round-286f3309710d806bb97dfe25778f2afe) in GG24.

## ⚖️ How

The [candidate repositories](data/candidate_repositories.csv) are slighly expanded with [some metadata](generate_repository_context.py) and then, an Arbitron competition is run on top of that. The competition generates lots of pairwise comparisons that are then converted to a ranking with relative weights.
